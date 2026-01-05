import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from config import MODEL_DIR, DATA_DIR, RESONANCE_FREQS, WINDOW_SIZE, STRIDE


def load_svm_model(model_path, scaler_path):
    """SVM 모델 로드"""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


def butter_lowpass_filter(data, cutoff, fs, order=5):
    """저역 통과 필터"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def calculate_sample_rate(df, time_column='Time_us'):
    """샘플링 레이트 계산"""
    if time_column not in df.columns:
        time_column = df.columns[0]
    time_data = df[time_column].dropna().values
    time_diffs = np.diff(time_data)
    avg_time_diff_us = np.mean(time_diffs)
    sample_rate = 1 / (avg_time_diff_us / 1_000_000)
    return sample_rate


def extract_frequency_features(signal, sample_rate, target_bands=[30, 40, 50, 60]):
    """주파수 특징 추출"""
    n = len(signal)
    fft_vals = fft(signal)
    freqs = fftfreq(n, 1 / sample_rate)

    positive_mask = freqs > 0
    freqs = freqs[positive_mask]
    fft_vals = np.abs(fft_vals[positive_mask])

    features = []
    for target_freq in target_bands:
        band_mask = (freqs >= target_freq - 5) & (freqs <= target_freq + 5)
        band_energy = np.sum(fft_vals[band_mask] ** 2)
        features.append(band_energy)

    full_band_mask = (freqs >= 30) & (freqs <= 60)
    full_band_energy = np.sum(fft_vals[full_band_mask] ** 2)
    features.append(full_band_energy)

    dominant_freq = freqs[np.argmax(fft_vals)]
    features.append(dominant_freq)

    return np.array(features), dominant_freq


def assign_label(f_in, res_freqs=RESONANCE_FREQS):
    """입력 주파수를 가장 가까운 공진 주파수 클래스로 매핑"""
    res_array = np.array(res_freqs)
    return np.argmin(np.abs(res_array - f_in))


def sliding_window_split(signal, window_size=8192, stride=4096):
    """슬라이딩 윈도우 분할"""
    windows = []
    for start in range(0, len(signal) - window_size + 1, stride):
        window = signal[start:start + window_size]
        windows.append(window)
    return windows


def process_csv_for_boundary_test(file_path, window_size=8192, stride=4096):
    """CSV 파일에서 경계 영역 샘플만 추출"""
    df = pd.read_csv(file_path)

    if 'Accel_Z' not in df.columns:
        return [], [], []

    sample_rate = calculate_sample_rate(df)
    signal = df['Accel_Z'].dropna().values
    signal = butter_lowpass_filter(signal, cutoff=sample_rate / 4, fs=sample_rate)
    windows = sliding_window_split(signal, window_size=window_size, stride=stride)

    X_list = []
    y_list = []
    freq_list = []

    for window in windows:
        features, input_freq = extract_frequency_features(window, sample_rate)

        is_boundary = False
        for res_freq in RESONANCE_FREQS:
            if abs(input_freq - res_freq) <= 5:
                is_boundary = True
                break

        if is_boundary:
            label = assign_label(input_freq)
            X_list.append(features)
            y_list.append(label)
            freq_list.append(input_freq)

    return X_list, y_list, freq_list


def create_boundary_test_dataset(data_dir, window_size=8192, stride=4096):
    """모든 CSV 파일에서 경계 영역 테스트 데이터 생성"""
    data_path = Path(data_dir)
    csv_files = sorted(list(data_path.glob("*.csv")))

    if not csv_files:
        raise ValueError(f"CSV 파일을 찾을 수 없습니다: {data_dir}")

    print("=" * 70)
    print("CSV 파일에서 경계 영역 테스트 데이터 생성")
    print("=" * 70)
    print(f"윈도우 크기: {window_size}, 이동 간격: {stride}")
    print(f"총 CSV 파일: {len(csv_files)}개")
    print(f"경계 영역: 각 공진 주파수 +/- 5Hz")
    print()

    all_X = []
    all_y = []
    all_freqs = []

    for idx, file_path in enumerate(csv_files, 1):
        X_list, y_list, freq_list = process_csv_for_boundary_test(
            file_path, window_size=window_size, stride=stride
        )
        all_X.extend(X_list)
        all_y.extend(y_list)
        all_freqs.extend(freq_list)

    X = np.array(all_X)
    y = np.array(all_y)
    freqs = np.array(all_freqs)

    print(f"경계 영역 테스트 데이터 생성 완료: 총 {len(X)}개 샘플")
    print(f"입력 주파수 범위: {freqs.min():.2f} ~ {freqs.max():.2f} Hz")
    print()

    print("[클래스별 경계 샘플 분포]")
    for i, freq in enumerate(RESONANCE_FREQS):
        count = np.sum(y == i)
        if count > 0:
            class_freqs = freqs[y == i]
            print(f"  클래스 {i} ({freq}Hz): {count}개 (범위: {class_freqs.min():.2f}~{class_freqs.max():.2f}Hz)")
    print()

    return X, y, freqs


def analyze_boundary_performance(model, scaler, X, y, freqs):
    """경계 영역 성능 분석"""
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    results = []
    for i in range(len(X)):
        expected = RESONANCE_FREQS[y[i]]
        predicted = RESONANCE_FREQS[y_pred[i]]
        correct = (y[i] == y_pred[i])

        results.append({
            'Input_Frequency': freqs[i],
            'Expected_Class': y[i],
            'Expected_Frequency': expected,
            'Predicted_Class': y_pred[i],
            'Predicted_Frequency': predicted,
            'Error': predicted - expected,
            'Correct': correct
        })

    results_df = pd.DataFrame(results)
    accuracy = np.mean(results_df['Correct']) * 100

    print("=" * 70)
    print("테스트 결과")
    print("=" * 70)
    print(f"전체 정확도: {accuracy:.2f}% ({results_df['Correct'].sum()}/{len(results_df)})")
    print()

    print("[주파수 대역별 정확도]")
    for i, freq in enumerate(RESONANCE_FREQS):
        mask = results_df['Expected_Class'] == i
        if mask.sum() > 0:
            class_data = results_df[mask]
            class_acc = class_data['Correct'].mean() * 100
            print(f"  {freq}Hz: {class_acc:6.2f}% ({class_data['Correct'].sum():2d}/{len(class_data):2d})")
    print()

    # 전체 결과 (처음 10개, 마지막 10개만)
    print("[전체 테스트 결과 (처음 10개)]")
    print(results_df[['Input_Frequency', 'Expected_Frequency',
                      'Predicted_Frequency', 'Error', 'Correct']].head(10).to_string(index=False))
    print("...")
    print(results_df[['Input_Frequency', 'Expected_Frequency',
                      'Predicted_Frequency', 'Error', 'Correct']].tail(10).to_string(index=False))
    print()

    # 오분류 전체 출력
    errors = results_df[~results_df['Correct']]
    if len(errors) > 0:
        print("[오분류 상세 분석]")
        print(f"총 {len(errors)}개의 오분류 발생:")
        print(errors[['Input_Frequency', 'Expected_Frequency',
                      'Predicted_Frequency', 'Error']].to_string(index=False))
    else:
        print("[오분류 없음]")
    print()

    return results_df, accuracy


def boundary_test_from_csv():
    """CSV 파일에서 경계 영역 테스트 실행"""
    model_path = f"{MODEL_DIR}/svm_model.pkl"
    scaler_path = f"{MODEL_DIR}/svm_scaler.pkl"

    model, scaler = load_svm_model(model_path, scaler_path)
    print("모델 로드 완료")
    print()

    X, y, freqs = create_boundary_test_dataset(
        data_dir=DATA_DIR,
        window_size=WINDOW_SIZE,
        stride=STRIDE
    )

    if len(X) == 0:
        print("경계 영역 샘플이 없습니다.")
        return None, 0

    results_df, accuracy = analyze_boundary_performance(model, scaler, X, y, freqs)

    save_path = Path(MODEL_DIR)
    csv_path = save_path / "boundary_test_results.csv"
    results_df.to_csv(csv_path, index=False)

    print("=" * 70)
    print("최종 결과")
    print("=" * 70)
    print(f"경계 영역 정확도: {accuracy:.2f}%")
    print(f"결과 파일: {csv_path}")
    print("=" * 70)

    return results_df, accuracy


if __name__ == "__main__":
    results_df, accuracy = boundary_test_from_csv()