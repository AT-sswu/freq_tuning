import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq


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

    # 지배 주파수
    dominant_freq = freqs[np.argmax(fft_vals)]
    features.append(dominant_freq)

    return np.array(features)


def calculate_power_output(energy, target_freq):
    """에너지 → 전력 변환"""
    efficiency = {
        30: 0.65,
        40: 0.80,
        50: 0.85,
        60: 0.75
    }

    closest_freq = min(efficiency.keys(), key=lambda x: abs(x - target_freq))
    power = energy * efficiency[closest_freq] * 1000

    return power


def sliding_window_split(signal, window_size=8192, stride=4096):
    """
    Sliding Window로 신호 분할

    Args:
        signal: 원본 신호 (예: 100,000개 샘플)
        window_size: 윈도우 크기 (기본 8192)
        stride: 이동 간격 (기본 4096, 50% 오버랩)

    Returns:
        분할된 신호 리스트
    """
    windows = []

    for start in range(0, len(signal) - window_size + 1, stride):
        window = signal[start:start + window_size]
        windows.append(window)

    return windows


def add_noise_augmentation(signal, noise_level=0.02):
    """
    가우시안 노이즈 추가
    """
    noise = np.random.normal(0, noise_level * np.std(signal), len(signal))
    return signal + noise


def scale_augmentation(signal, scale_range=(0.9, 1.1)):
    """
    스케일링 (진폭 변화)
    """
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return signal * scale


def time_shift_augmentation(signal, max_shift=100):
    """
    시간축 이동
    """
    shift = np.random.randint(-max_shift, max_shift)
    if shift > 0:
        return np.pad(signal[shift:], (0, shift), mode='edge')
    elif shift < 0:
        return np.pad(signal[:shift], (-shift, 0), mode='edge')
    return signal


def augment_window(window, sample_rate, num_augmentations=3):
    """
    하나의 윈도우에 대해 여러 증강 적용

    Returns:
        원본 + 증강된 윈도우들 리스트
    """
    augmented = [window]  # 원본 포함

    for _ in range(num_augmentations):
        aug_window = window.copy()

        # 랜덤하게 증강 기법 선택 (2개 조합)
        if np.random.rand() > 0.5:
            aug_window = add_noise_augmentation(aug_window)

        if np.random.rand() > 0.5:
            aug_window = scale_augmentation(aug_window)

        if np.random.rand() > 0.5:
            aug_window = time_shift_augmentation(aug_window)

        augmented.append(aug_window)

    return augmented


def process_single_file_with_augmentation(
        file_path,
        window_size=8192,
        stride=4096,
        num_augmentations=2,
        apply_filter=True
):
    """
    단일 CSV 파일 처리: Sliding Window + Augmentation

    Args:
        file_path: CSV 파일 경로
        window_size: 윈도우 크기 (기본 8192 = 약 27초 @ 296Hz)
        stride: 이동 간격 (기본 4096 = 50% 오버랩)
        num_augmentations: 윈도우당 생성할 증강 데이터 수

    Returns:
        X_list, y_list (특징, 전력 값 리스트)
    """
    df = pd.read_csv(file_path)

    if 'Accel_Z' not in df.columns:
        print(f"  [경고] Accel_Z 없음 - 스킵")
        return [], []

    # 샘플링 레이트 계산
    sample_rate = calculate_sample_rate(df)

    # Accel_Z 데이터 추출
    signal = df['Accel_Z'].dropna().values

    print(f"  원본 신호 길이: {len(signal)} 샘플 ({len(signal) / sample_rate:.1f}초)")

    # 필터 적용 (선택)
    if apply_filter:
        signal = butter_lowpass_filter(signal, cutoff=sample_rate / 4, fs=sample_rate)

    # Sliding Window로 분할
    windows = sliding_window_split(signal, window_size=window_size, stride=stride)
    print(f"  Sliding Window 분할: {len(windows)}개 윈도우 생성")

    X_list = []
    y_list = []

    # 각 윈도우에 대해 증강 적용
    for window in windows:
        # 증강 (원본 + 증강 N개)
        augmented_windows = augment_window(window, sample_rate, num_augmentations)

        for aug_window in augmented_windows:
            # 특징 추출
            features = extract_frequency_features(aug_window, sample_rate)

            # 전력 계산
            target_freqs = [30, 40, 50, 60]
            powers = []

            for idx, target_freq in enumerate(target_freqs):
                band_energy = features[idx]
                power = calculate_power_output(band_energy, target_freq)
                powers.append(power)

            X_list.append(features)
            y_list.append(powers)

    print(f"  증강 후 최종 샘플 수: {len(X_list)}개")

    return X_list, y_list


def create_augmented_dataset(
        data_dir,
        output_dir,
        window_size=8192,
        stride=4096,
        num_augmentations=2,
        apply_filter=True
):
    """
    전체 데이터셋에 대해 증강 적용
    """
    data_path = Path(data_dir)
    csv_files = sorted(list(data_path.glob("*.csv")))

    if not csv_files:
        raise ValueError(f"CSV 파일을 찾을 수 없습니다: {data_dir}")

    print(f"\n{'=' * 60}")
    print(f"데이터 증강 시작")
    print(f"{'=' * 60}")
    print(f"설정:")
    print(f"  윈도우 크기: {window_size} 샘플")
    print(f"  이동 간격: {stride} 샘플 (오버랩 {(1 - stride / window_size) * 100:.0f}%)")
    print(f"  윈도우당 증강 수: {num_augmentations}개")
    print(f"  총 파일 수: {len(csv_files)}개")
    print(f"{'=' * 60}\n")

    all_X = []
    all_y = []

    for idx, file_path in enumerate(csv_files, 1):
        print(f"[{idx}/{len(csv_files)}] 처리 중: {file_path.name}")

        try:
            X_list, y_list = process_single_file_with_augmentation(
                file_path,
                window_size=window_size,
                stride=stride,
                num_augmentations=num_augmentations,
                apply_filter=apply_filter
            )

            all_X.extend(X_list)
            all_y.extend(y_list)

        except Exception as e:
            print(f"  ✗ 오류: {e}")
            continue

    # numpy 배열로 변환
    X = np.array(all_X)
    y = np.array(all_y)

    print(f"\n{'=' * 60}")
    print(f"데이터 증강 완료")
    print(f"{'=' * 60}")
    print(f"최종 데이터 형상:")
    print(f"  원본 파일 수: {len(csv_files)}개")
    print(f"  최종 샘플 수: {len(X)}개")
    print(f"  증가 비율: {len(X) / len(csv_files):.1f}배")
    print(f"  입력 형상: {X.shape}")
    print(f"  출력 형상: {y.shape}")
    print(f"{'=' * 60}")

    # 저장
    output_path = Path(output_dir) / "preprocessed_data_augmented.npz"
    np.savez(output_path, X=X, y=y, target_freqs=[30, 40, 50, 60])
    print(f"\n저장 완료: {output_path}")

    # 통계 정보
    print(f"\n[데이터 통계]")
    print(f"입력 특징 범위: {X.min():.2e} ~ {X.max():.2e}")
    print(f"출력 전력 범위: {y.min():.2f} ~ {y.max():.2f} mW")
    print(f"출력 전력 평균: {y.mean():.2f} mW")

    return X, y


if __name__ == "__main__":
    # 설정
    DATA_DIR = "/Users/seohyeon/AT_freq_tuning/data_v3"
    OUTPUT_DIR = "/Users/seohyeon/AT_freq_tuning/vibration_mlp/preprocess_results"

    # 증강 파라미터
    WINDOW_SIZE = 8192  # 약 27초 @ 296Hz
    STRIDE = 4096  # 50% 오버랩 (권장: 2048~4096)
    NUM_AUGMENTATIONS = 2  # 윈도우당 2개 증강 (원본 포함 총 3개)

    # 증강 실행
    X, y = create_augmented_dataset(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        num_augmentations=NUM_AUGMENTATIONS,
        apply_filter=True
    )
