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
    """
    주파수 도메인 특징 추출 및 로그 스케일 적용
    """
    n = len(signal)
    fft_vals = fft(signal - np.mean(signal))  # DC Offset 제거
    freqs = fftfreq(n, 1 / sample_rate)

    positive_mask = freqs > 0
    freqs = freqs[positive_mask]
    fft_vals = np.abs(fft_vals[positive_mask])

    features = []

    # 1. 각 목표 주파수 대역(±5Hz)의 에너지 계산 및 로그 변환
    for target_freq in target_bands:
        band_mask = (freqs >= target_freq - 5) & (freqs <= target_freq + 5)
        # 에너지 합산 후 log10(x + 1) 적용
        band_energy = np.sum(fft_vals[band_mask] ** 2)
        features.append(np.log10(band_energy + 1))

    # 2. 30-60Hz 전체 에너지 로그 변환
    full_band_mask = (freqs >= 30) & (freqs <= 60)
    full_band_energy = np.sum(fft_vals[full_band_mask] ** 2)
    features.append(np.log10(full_band_energy + 1))

    # 3. 지배 주파수 (주파수 자체는 로그 변환하지 않음)
    dominant_freq = freqs[np.argmax(fft_vals)]
    features.append(dominant_freq)

    return np.array(features)


def calculate_power_output(energy, target_freq):
    """
    에너지 → 전력 변환 (로그 변환 전의 raw 에너지 사용)
    """
    efficiency = {30: 0.65, 40: 0.80, 50: 0.85, 60: 0.75}
    closest_freq = min(efficiency.keys(), key=lambda x: abs(x - target_freq))

    # 전력 = 에너지 × 효율 * 1000 (mW)
    power = energy * efficiency[closest_freq] * 1000
    return power


def prepare_training_data(data_dir, output_dir):
    data_path = Path(data_dir)
    # 상위 폴더 생성
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    csv_files = sorted(list(data_path.glob("*.csv")))
    if not csv_files:
        raise ValueError(f"CSV 파일을 찾을 수 없습니다: {data_dir}")

    X_data = []
    y_data = []

    print(f"\n{'=' * 60}")
    print(f"로그 스케일 데이터 전처리 시작 ({len(csv_files)}개 파일)")
    print(f"{'=' * 60}")

    for idx, file_path in enumerate(csv_files, 1):
        try:
            df = pd.read_csv(file_path)
            if 'Accel_Z' not in df.columns: continue

            sample_rate = calculate_sample_rate(df)
            signal = df['Accel_Z'].dropna().values
            signal = butter_lowpass_filter(signal, cutoff=sample_rate / 4, fs=sample_rate)

            # 1. 입력 특징 추출 (이미 내부에서 로그 변환됨)
            features = extract_frequency_features(signal, sample_rate)

            # 2. 출력 전력 계산 및 로그 변환
            target_freqs = [30, 40, 50, 60]
            powers_log = []

            for target_freq in target_freqs:
                # features의 0~3번 인덱스는 로그 에너지가 저장되어 있음 -> 지수함수로 복원하여 전력 계산
                raw_energy = 10 ** (features[target_freqs.index(target_freq)]) - 1
                power = calculate_power_output(raw_energy, target_freq)

                # 계산된 전력값에 로그 변환 적용
                powers_log.append(np.log10(power + 1))

            X_data.append(features)
            y_data.append(powers_log)
            print(f"[{idx}/{len(csv_files)}] ✓ {file_path.name} 완료")

        except Exception as e:
            print(f"[{idx}/{len(csv_files)}] ✗ 오류: {e}")

    X = np.array(X_data)
    y = np.array(y_data)

    # 저장
    output_path = Path(output_dir) / "preprocessed_data_log.npz"
    np.savez(output_path, X=X, y=y, target_freqs=[30, 40, 50, 60])

    print(f"\n{'=' * 60}")
    print(f"[완료] 데이터 전처리 결과 (로그 스케일)")
    print(f"  입력 형상: {X.shape}, 출력 형상: {y.shape}")
    print(f"  입력 범위 (Log): {X.min():.2f} ~ {X.max():.2f}")
    print(f"  출력 범위 (Log): {y.min():.2f} ~ {y.max():.2f}")
    print(f"  저장 위치: {output_path}")
    print(f"{'=' * 60}")

    return X, y


if __name__ == "__main__":
    DATA_DIR = "/Users/seohyeon/AT_freq_tuning/data_v3"
    OUTPUT_DIR = "/Users/seohyeon/AT_freq_tuning/fft_analysis/results"
    X, y = prepare_training_data(DATA_DIR, OUTPUT_DIR)