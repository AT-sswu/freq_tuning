import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.colors import ListedColormap
from imblearn.over_sampling import RandomOverSampler

# config 파일이 없을 경우를 대비한 기본값 설정
try:
    from config import MODEL_DIR, DATA_DIR, RESONANCE_FREQS, WINDOW_SIZE, STRIDE
except ImportError:
    MODEL_DIR = "./models"
    DATA_DIR = "./data"
    RESONANCE_FREQS = [40, 50, 60]
    WINDOW_SIZE = 8192
    STRIDE = 4096


# =========================================================
# [1] 기본 유틸리티 함수
# =========================================================

def load_svm_model(model_path, scaler_path):
    """SVM 모델 및 스케일러 로드"""
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

    return 1 / (avg_time_diff_us / 1_000_000)


def extract_frequency_features(signal, sample_rate, target_bands=RESONANCE_FREQS):
    """
    주파수 특징 추출
    - 각 공진 주파수 ±5Hz 에너지
    - 전체 밴드 에너지 (25~65Hz)
    - 지배 주파수
    """
    n = len(signal)
    fft_vals = np.abs(fft(signal))
    freqs = fftfreq(n, 1 / sample_rate)

    mask = freqs > 0
    freqs = freqs[mask]
    fft_vals = fft_vals[mask]

    features = []

    for f in target_bands:
        band = (freqs >= f - 5) & (freqs <= f + 5)
        features.append(np.sum(fft_vals[band] ** 2))

    full_band = (freqs >= 25) & (freqs <= 65)
    features.append(np.sum(fft_vals[full_band] ** 2))

    dominant_freq = freqs[np.argmax(fft_vals)]
    features.append(dominant_freq)

    return np.array(features), dominant_freq


def assign_label(freq, res_freqs=RESONANCE_FREQS):
    """가장 가까운 공진 주파수 클래스 할당"""
    return np.argmin(np.abs(np.array(res_freqs) - freq))


def sliding_window_split(signal, window_size, stride):
    """슬라이딩 윈도우 분할"""
    return [
        signal[i:i + window_size]
        for i in range(0, len(signal) - window_size + 1, stride)
    ]


# =========================================================
# [2] CSV 처리 및 데이터 증강
# =========================================================

def process_csv_for_boundary_test(file_path, window_size, stride, show_samples=True):
    """CSV → 경계 영역 샘플 추출"""
    df = pd.read_csv(file_path)
    if 'Accel_Z' not in df.columns:
        return [], [], []

    sample_rate = calculate_sample_rate(df)
    signal = butter_lowpass_filter(
        df['Accel_Z'].dropna().values,
        cutoff=sample_rate / 4,
        fs=sample_rate
    )

    windows = sliding_window_split(signal, window_size, stride)

    X, y, freqs = [], [], []
    samples_shown = 0

    for idx, window in enumerate(windows):
        features, input_freq = extract_frequency_features(window, sample_rate)

        if any(abs(input_freq - r) <= 5 for r in RESONANCE_FREQS):
            X.append(features)
            label = assign_label(input_freq)
            y.append(label)
            freqs.append(input_freq)

            if show_samples and samples_shown < 3:
                print(f"\n  샘플 #{samples_shown + 1} (Window {idx})")
                print(f"    입력 주파수: {input_freq:.2f} Hz")
                print(f"    할당된 클래스: {label} ({RESONANCE_FREQS[label]} Hz)")
                print(f"    특징 벡터 (처음 3개): [{features[0]:.2e}, {features[1]:.2e}, {features[2]:.2e}, ...]")
                samples_shown += 1

    return X, y, freqs


def generate_hard_samples(X, y, num_per_class, verbose=True):
    """경계 근처 어려운 샘플 생성"""
    X_hard, y_hard = [], []
    hard_sample_details = []

    if verbose:
        print("\n[어려운 샘플 생성]")
        print(f"  클래스당 목표 개수: {num_per_class}개")

    for class_idx in range(len(RESONANCE_FREQS)):
        class_samples = X[y == class_idx]
        if len(class_samples) == 0:
            continue

        indices = np.random.choice(
            len(class_samples),
            size=min(num_per_class, len(class_samples)),
            replace=True
        )

        for idx in indices:
            sample = class_samples[idx].copy()
            sample *= (1 + np.random.normal(0, 0.05, size=sample.shape))

            if class_idx < len(RESONANCE_FREQS) - 1:
                next_f = RESONANCE_FREQS[class_idx + 1]
                sample[-1] = (RESONANCE_FREQS[class_idx] + next_f) / 2 + np.random.uniform(-2, 2)
            else:
                sample[-1] += np.random.uniform(-3, 3)

            generated_freq = sample[-1]
            true_label = assign_label(generated_freq)

            X_hard.append(sample)
            y_hard.append(true_label)

            hard_sample_details.append({
                'original_class': class_idx,
                'original_freq': RESONANCE_FREQS[class_idx],
                'true_class': true_label,
                'true_freq': RESONANCE_FREQS[true_label],
                'generated_freq': generated_freq,
                'features': sample
            })

        if verbose:
            print(f"  클래스 {class_idx} ({RESONANCE_FREQS[class_idx]}Hz): {len(indices)}개 생성")

    return np.array(X_hard), np.array(y_hard), hard_sample_details


def balance_and_normalize_data(X, y, verbose=True):
    """오버샘플링 + 어려운 샘플 10% 추가"""
    print("\n" + "=" * 60)
    print("[데이터 밸런싱 및 증강 시작]")
    print("=" * 60)

    print("\n[원본 분포]")
    for i, freq in enumerate(RESONANCE_FREQS):
        print(f"  클래스 {i} ({freq}Hz): {np.sum(y == i)}개")

    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)

    print("\n[오버샘플링 후]")
    for i, freq in enumerate(RESONANCE_FREQS):
        print(f"  클래스 {i} ({freq}Hz): {np.sum(y_res == i)}개")

    samples_per_class = np.bincount(y_res)[0]
    hard_n = int(samples_per_class * 0.1)

    X_hard, y_hard, hard_details = generate_hard_samples(X_res, y_res, hard_n)

    X_final = np.vstack([X_res, X_hard])
    y_final = np.hstack([y_res, y_hard])

    print("\n[최종 분포]")
    for i, freq in enumerate(RESONANCE_FREQS):
        print(f"  클래스 {i} ({freq}Hz): {np.sum(y_final == i)}개")

    print(f"  총 샘플 수: {len(y_final)}개")

    return X_final, y_final, hard_details


# =========================================================
# [4] 메인 실행
# =========================================================

def boundary_test_from_csv():
    print("\n" + "#" * 60)
    print("SVM 경계 테스트 및 시각화")
    print("#" * 60 + "\n")

    model_file = Path(MODEL_DIR) / "svm_model.pkl"
    scaler_file = Path(MODEL_DIR) / "svm_scaler.pkl"

    if not model_file.exists():
        print(f"모델 파일을 찾을 수 없습니다: {model_file}")
        return

    model, scaler = load_svm_model(model_file, scaler_file)
    print(f"모델 로드 완료: kernel={model.kernel}, C={model.C}, gamma={model.gamma}")

    X_raw, y_raw = [], []
    for csv in Path(DATA_DIR).glob("*.csv"):
        X, y, _ = process_csv_for_boundary_test(csv, WINDOW_SIZE, STRIDE)
        X_raw.extend(X)
        y_raw.extend(y)

    if not X_raw:
        print("분석할 데이터가 없습니다.")
        return

    X = np.array(X_raw)
    y = np.array(y_raw)

    X_bal, y_bal, _ = balance_and_normalize_data(X, y)

    X_scaled = scaler.transform(X_bal)
    y_pred = model.predict(X_scaled)

    print("\n[Classification Report]")
    print(classification_report(y_bal, y_pred, target_names=[f"{f}Hz" for f in RESONANCE_FREQS]))


if __name__ == "__main__":
    boundary_test_from_csv()
