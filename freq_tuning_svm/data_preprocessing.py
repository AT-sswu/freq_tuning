import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split
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


def extract_frequency_features(signal, sample_rate, target_bands=RESONANCE_FREQS):
    """주파수 도메인 특징 추출"""
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


def augment_window(window, num_augmentations=2):
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


def process_single_signal(signal, sample_rate, window_size, stride, num_augmentations):
    """단일 신호 처리 (슬라이딩 윈도우 + 증강)"""
    windows = sliding_window_split(signal, window_size=window_size, stride=stride)

    X_list = []
    y_list = []
    freq_list = []

    for window in windows:
        augmented_windows = augment_window(window, num_augmentations)

        for aug_window in augmented_windows:
            features, input_freq = extract_frequency_features(aug_window, sample_rate)
            label = assign_label(input_freq)

            X_list.append(features)
            y_list.append(label)
            freq_list.append(input_freq)

    return X_list, y_list, freq_list


def load_all_signals(data_dir):
    """모든 CSV 파일을 신호 단위로 로드"""
    data_path = Path(data_dir)
    csv_files = sorted(list(data_path.glob("*.csv")))

    if not csv_files:
        raise ValueError(f"CSV 파일을 찾을 수 없습니다: {data_dir}")

    signals = []

    print(f"\n{'=' * 70}")
    print(f"{'원본 신호 로딩':^70}")
    print(f"{'=' * 70}")
    print(f"총 파일 수: {len(csv_files)}개\n")

    for idx, file_path in enumerate(csv_files):
        try:
            df = pd.read_csv(file_path)

            if 'Accel_Z' not in df.columns:
                print(f"  [{idx + 1}] {file_path.name} - Accel_Z 없음, 스킵")
                continue

            sample_rate = calculate_sample_rate(df)
            signal = df['Accel_Z'].dropna().values
            signal = butter_lowpass_filter(signal, cutoff=sample_rate / 4, fs=sample_rate)

            signals.append({
                'signal': signal,
                'sample_rate': sample_rate,
                'file_name': file_path.name,
                'signal_id': idx
            })

            print(f"  [{idx + 1}] {file_path.name} - 로드 완료 (길이: {len(signal)})")

        except Exception as e:
            print(f"  [{idx + 1}] {file_path.name} - 오류: {e}")
            continue
    print(f"\n총 {len(signals)}개 신호 로드 완료")
    return signals


def create_classification_dataset_fixed(data_dir, output_dir, window_size=8192,
                                        stride=4096, num_augmentations=2,
                                        test_size=0.2, random_state=42):
    """데이터 누수 방지: 신호 단위로 train/test 분할"""

    print(f"\n{'=' * 70}")
    print(f"{'분류 데이터셋 생성 (데이터 누수 방지)':^70}")
    print(f"{'=' * 70}")
    print(f"설정:")
    print(f"  윈도우 크기: {window_size} 샘플")
    print(f"  이동 간격: {stride} 샘플")
    print(f"  윈도우당 증강 수: {num_augmentations}개")
    print(f"  Test 비율: {test_size * 100}%")
    print(f"  공진 주파수 클래스: {RESONANCE_FREQS}")
    print(f"{'=' * 70}\n")

    signals = load_all_signals(data_dir)

    if len(signals) == 0:
        raise ValueError("로드된 신호가 없습니다")

    print(f"\n{'=' * 70}")
    print(f"{'신호 단위로 Train/Test 분할':^70}")
    print(f"{'=' * 70}")

    signal_indices = np.arange(len(signals))
    train_indices, test_indices = train_test_split(
        signal_indices,
        test_size=test_size,
        random_state=random_state
    )

    print(f"Train 신호: {len(train_indices)}개")
    print(f"Test 신호: {len(test_indices)}개")

    print(f"\n{'=' * 70}")
    print(f"{'Train 데이터 생성':^70}")
    print(f"{'=' * 70}")

    X_train_all, y_train_all, freq_train_all = [], [], []

    for idx in train_indices:
        signal_data = signals[idx]
        X_list, y_list, freq_list = process_single_signal(
            signal_data['signal'],
            signal_data['sample_rate'],
            window_size,
            stride,
            num_augmentations
        )
        X_train_all.extend(X_list)
        y_train_all.extend(y_list)
        freq_train_all.extend(freq_list)

    X_train = np.array(X_train_all)
    y_train = np.array(y_train_all)
    freq_train = np.array(freq_train_all)

    print(f"Train 데이터: {X_train.shape[0]}개 샘플 생성")

    print(f"\n{'=' * 70}")
    print(f"{'Test 데이터 생성':^70}")
    print(f"{'=' * 70}")

    X_test_all, y_test_all, freq_test_all = [], [], []

    for idx in test_indices:
        signal_data = signals[idx]
        X_list, y_list, freq_list = process_single_signal(
            signal_data['signal'],
            signal_data['sample_rate'],
            window_size,
            stride,
            num_augmentations
        )
        X_test_all.extend(X_list)
        y_test_all.extend(y_list)
        freq_test_all.extend(freq_list)

    X_test = np.array(X_test_all)
    y_test = np.array(y_test_all)
    freq_test = np.array(freq_test_all)

    print(f"Test 데이터: {X_test.shape[0]}개 샘플 생성")

    print(f"\n{'=' * 70}")
    print(f"{'데이터셋 생성 완료':^70}")
    print(f"{'=' * 70}")
    print(f"Train 형상: X={X_train.shape}, y={y_train.shape}")
    print(f"Test 형상: X={X_test.shape}, y={y_test.shape}")
    print(f"Train 주파수 범위: {freq_train.min():.2f} ~ {freq_train.max():.2f} Hz")
    print(f"Test 주파수 범위: {freq_test.min():.2f} ~ {freq_test.max():.2f} Hz")

    print(f"\n[Train 클래스 분포]")
    for i, freq in enumerate(RESONANCE_FREQS):
        count = np.sum(y_train == i)
        print(f"  클래스 {i} ({freq}Hz): {count}개 ({count / len(y_train) * 100:.1f}%)")

    print(f"\n[Test 클래스 분포]")
    for i, freq in enumerate(RESONANCE_FREQS):
        count = np.sum(y_test == i)
        print(f"  클래스 {i} ({freq}Hz): {count}개 ({count / len(y_test) * 100:.1f}%)")

    print(f"{'=' * 70}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    save_path = output_path / "classification_data.npz"
    np.savez(
        save_path,
        X_train=X_train, y_train=y_train, freq_train=freq_train,
        X_test=X_test, y_test=y_test, freq_test=freq_test,
        train_indices=train_indices,
        test_indices=test_indices,
        resonance_freqs=RESONANCE_FREQS
    )
    print(f"\n저장 완료: {save_path}")

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    X_train, y_train, X_test, y_test = create_classification_dataset_fixed(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        num_augmentations=NUM_AUGMENTATIONS,
        test_size=0.2,
        random_state=42
    )
