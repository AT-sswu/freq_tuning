import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from config import DATA_DIR, OUTPUT_DIR, RESONANCE_FREQS, WINDOW_SIZE, STRIDE, NUM_AUGMENTATIONS


def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def calculate_sample_rate(df, time_column='Time_us'):
    if time_column not in df.columns:
        time_column = df.columns[0]
    time_data = df[time_column].dropna().values
    time_diffs = np.diff(time_data)
    avg_time_diff_us = np.mean(time_diffs)
    sample_rate = 1 / (avg_time_diff_us / 1_000_000)
    return sample_rate


def extract_frequency_features(signal, sample_rate, target_bands=[30, 40, 50, 60]):
    """주파수 도메인 특징 추출"""
    n = len(signal)
    fft_vals = fft(signal)
    freqs = fftfreq(n, 1 / sample_rate)

    positive_mask = freqs > 0
    freqs = freqs[positive_mask]
    fft_vals = np.abs(fft_vals[positive_mask])

    features = []

    # 각 목표 주파수 대역(±5Hz)의 에너지
    for target_freq in target_bands:
        band_mask = (freqs >= target_freq - 5) & (freqs <= target_freq + 5)
        band_energy = np.sum(fft_vals[band_mask] ** 2)
        features.append(band_energy)

    # 30-60Hz 전체 에너지
    full_band_mask = (freqs >= 30) & (freqs <= 60)
    full_band_energy = np.sum(fft_vals[full_band_mask] ** 2)
    features.append(full_band_energy)

    # 지배 주파수 (입력 주파수)
    dominant_freq = freqs[np.argmax(fft_vals)]
    features.append(dominant_freq)

    return np.array(features), dominant_freq


def assign_label(f_in, res_freqs=RESONANCE_FREQS):
    """
    입력 주파수를 가장 가까운 공진 주파수 클래스로 매핑

    Args:
        f_in: 입력 주파수 (예: 39.8 Hz)
        res_freqs: 공진 주파수 리스트 [30, 40, 50, 60]

    Returns:
        클래스 레이블 (0, 1, 2, 3)
    """
    res_array = np.array(res_freqs)
    return np.argmin(np.abs(res_array - f_in))


def sliding_window_split(signal, window_size=8192, stride=4096):
    windows = []
    for start in range(0, len(signal) - window_size + 1, stride):
        window = signal[start:start + window_size]
        windows.append(window)
    return windows


def add_noise_augmentation(signal, noise_level=0.02):
    noise = np.random.normal(0, noise_level * np.std(signal), len(signal))
    return signal + noise


def scale_augmentation(signal, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return signal * scale


def time_shift_augmentation(signal, max_shift=100):
    shift = np.random.randint(-max_shift, max_shift)
    if shift > 0:
        return np.pad(signal[shift:], (0, shift), mode='edge')
    elif shift < 0:
        return np.pad(signal[:shift], (-shift, 0), mode='edge')
    return signal


def augment_window(window, sample_rate, num_augmentations=2):
    augmented = [window]
    for _ in range(num_augmentations):
        aug_window = window.copy()
        if np.random.rand() > 0.5:
            aug_window = add_noise_augmentation(aug_window)
        if np.random.rand() > 0.5:
            aug_window = scale_augmentation(aug_window)
        if np.random.rand() > 0.5:
            aug_window = time_shift_augmentation(aug_window)
        augmented.append(aug_window)
    return augmented


def process_single_file(file_path, window_size=8192, stride=4096, num_augmentations=2, apply_filter=True):
    df = pd.read_csv(file_path)

    if 'Accel_Z' not in df.columns:
        print(f"  [경고] Accel_Z 없음 - 스킵")
        return [], [], []

    sample_rate = calculate_sample_rate(df)
    signal = df['Accel_Z'].dropna().values

    print(f"  원본 신호 길이: {len(signal)} 샘플")

    if apply_filter:
        signal = butter_lowpass_filter(signal, cutoff=sample_rate / 4, fs=sample_rate)

    windows = sliding_window_split(signal, window_size=window_size, stride=stride)
    print(f"  Sliding Window 분할: {len(windows)}개 윈도우 생성")

    X_list = []
    y_list = []
    freq_list = []

    for window in windows:
        augmented_windows = augment_window(window, sample_rate, num_augmentations)

        for aug_window in augmented_windows:
            # 특징 추출 + 입력 주파수 추출
            features, input_freq = extract_frequency_features(aug_window, sample_rate)

            # 라벨 생성: 입력 주파수 → 가장 가까운 공진 주파수 클래스
            label = assign_label(input_freq)

            X_list.append(features)
            y_list.append(label)
            freq_list.append(input_freq)

    print(f"  증강 후 최종 샘플 수: {len(X_list)}개")

    return X_list, y_list, freq_list


def create_classification_dataset(data_dir, output_dir, window_size=8192, stride=4096, num_augmentations=2):
    data_path = Path(data_dir)
    csv_files = sorted(list(data_path.glob("*.csv")))

    if not csv_files:
        raise ValueError(f"CSV 파일을 찾을 수 없습니다: {data_dir}")

    print(f"\n{'=' * 60}")
    print(f"분류 데이터셋 생성 시작")
    print(f"{'=' * 60}")
    print(f"설정:")
    print(f"  윈도우 크기: {window_size} 샘플")
    print(f"  이동 간격: {stride} 샘플")
    print(f"  윈도우당 증강 수: {num_augmentations}개")
    print(f"  총 파일 수: {len(csv_files)}개")
    print(f"  공진 주파수 클래스: {RESONANCE_FREQS}")
    print(f"{'=' * 60}\n")

    all_X = []
    all_y = []
    all_freqs = []

    for idx, file_path in enumerate(csv_files, 1):
        print(f"[{idx}/{len(csv_files)}] 처리 중: {file_path.name}")

        try:
            X_list, y_list, freq_list = process_single_file(
                file_path,
                window_size=window_size,
                stride=stride,
                num_augmentations=num_augmentations,
                apply_filter=True
            )

            all_X.extend(X_list)
            all_y.extend(y_list)
            all_freqs.extend(freq_list)

        except Exception as e:
            print(f"  ✗ 오류: {e}")
            import traceback
            traceback.print_exc()
            continue

    X = np.array(all_X)
    y = np.array(all_y)
    freqs = np.array(all_freqs)

    print(f"\n{'=' * 60}")
    print(f"데이터셋 생성 완료")
    print(f"{'=' * 60}")
    print(f"  입력 형상: {X.shape}")
    print(f"  레이블 형상: {y.shape}")
    print(f"  입력 주파수 범위: {freqs.min():.2f} ~ {freqs.max():.2f} Hz")
    print(f"\n[클래스 분포]")
    for i, freq in enumerate(RESONANCE_FREQS):
        count = np.sum(y == i)
        print(f"  클래스 {i} ({freq}Hz): {count}개 ({count / len(y) * 100:.1f}%)")
    print(f"{'=' * 60}")

    # 저장
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / "classification_data.npz"
    np.savez(save_path, X=X, y=y, freqs=freqs, resonance_freqs=RESONANCE_FREQS)
    print(f"\n저장 완료: {save_path}")

    return X, y, freqs


if __name__ == "__main__":
    X, y, freqs = create_classification_dataset(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        num_augmentations=NUM_AUGMENTATIONS
    )