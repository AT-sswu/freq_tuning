import pandas as pd
import numpy as np
from pathlib import Path
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split
from config import DATA_DIR, OUTPUT_DIR, RESONANCE_FREQS, WINDOW_SIZE, STRIDE, TEST_SIZE, RANDOM_STATE, AUGMENTATION_SNR_DB, ENABLE_AUGMENTATION

def add_gaussian_noise_with_snr(signal, snr_db):
    """SNR(dB)을 기준으로 가우시안 노이즈를 추가하여 신호 증강"""
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)
    noise = np.random.normal(0, noise_std, len(signal))
    augmented_signal = signal + noise
    return augmented_signal

def calculate_sample_rate(df, time_column='Time_us'):
    """데이터 기반 샘플링 주파수 계산"""
    if time_column not in df.columns:
        time_column = df.columns[0]
    time_data = df[time_column].dropna().values
    time_diffs = np.diff(time_data)
    avg_time_diff_us = np.mean(time_diffs)
    sample_rate = 1 / (avg_time_diff_us / 1_000_000)
    return sample_rate

def extract_frequency_features(signal, sample_rate, target_bands=RESONANCE_FREQS):
    """Q-factor 기반 주파수 특징 추출 (Q=10으로 고정, ±5% 범위)"""
    n = len(signal)
    fft_vals = fft(signal)
    freqs = fftfreq(n, 1 / sample_rate)
    
    positive_mask = freqs > 0
    freqs = freqs[positive_mask]
    fft_vals = np.abs(fft_vals[positive_mask])
    
    # 정규화: FFT 에너지 합을 1로 정규화하여 스케일 독립성 확보
    fft_energy_sum = np.sum(fft_vals ** 2)
    if fft_energy_sum > 0:
        fft_vals_normalized = fft_vals / np.sqrt(fft_energy_sum)
    else:
        fft_vals_normalized = fft_vals
    
    features = []
    
    # Q-factor 기반 대역폭 정의 (Q=10 고정: ±5% 범위)
    q_factor = 10
    
    # 각 공진 주파수별 특징 추출
    for target_freq in target_bands:
        # 대역폭 = target_freq / Q
        bandwidth = target_freq / q_factor
        
        # ±bandwidth/2 범위 정의
        band_mask = (freqs >= target_freq - bandwidth/2) & (freqs <= target_freq + bandwidth/2)
        band_energy = np.sum(fft_vals_normalized[band_mask] ** 2)
        features.append(band_energy)
    
    # 지배 주파수 (dominant frequency)
    peak_idx = np.argmax(fft_vals)
    peak_frequency = freqs[peak_idx]
    
    return np.array(features), peak_frequency

def assign_label(peak_frequency):
    """
    지배 주파수를 가장 가까운 공진 주파수로 매핑하여 클래스 할당
    
    관성 지배 영역 이론 적용:
    - 입력 주파수 > 고유 주파수: 관성 지배 영역 (w > ωn)
    - 입력 주파수 < 고유 주파수: 강성 지배 영역 (w < ωn)
    
    예:
    - 45Hz (40과 50의 중간) → 40Hz로 매핑 (첫번째 최솟값)
    - 55Hz (50과 60의 중간) → 50Hz로 매핑 (첫번째 최솟값)
    """
    res_array = np.array(RESONANCE_FREQS)
    label = np.argmin(np.abs(res_array - peak_frequency))
    return label

def calculate_augmentation_multiplier(class_counts, strategy='balanced'):
    """
    클래스별 증강 배수 계산 - 각 클래스를 동일한 샘플 개수로 맞춤
    
    Args:
        class_counts: 각 클래스의 샘플 개수
        strategy: 'balanced' (모든 클래스를 최대값에 맞춤)
    """
    multipliers = {}
    max_count = np.max(class_counts)
    
    for class_id, count in enumerate(class_counts):
        multipliers[class_id] = max_count / count
    
    return multipliers
    """주파수를 가장 가까운 공진 주파수로 매핑"""
    res_array = np.array(RESONANCE_FREQS)
    label = np.argmin(np.abs(res_array - peak_frequency))
    return label

def sliding_window_split(signal, window_size=512, stride=256):
    """윈도우 분할"""
    windows = [] 
    for start in range(0, len(signal) - window_size + 1, stride):
        window = signal[start:start + window_size]
        windows.append(window)
    return windows

def process_single_signal(signal, sample_rate, window_size, stride, file_name, snr_db=None):
    """신호 전처리 및 특징 추출"""
    windows = sliding_window_split(signal, window_size=window_size, stride=stride)
    X_list, y_list = [], []
    freq_list = []
    snr_list = []

    for window in windows:
        features, peak_frequency = extract_frequency_features(window, sample_rate)
        label = assign_label(peak_frequency)
        X_list.append(features)
        y_list.append(label)
        freq_list.append(peak_frequency)
        snr_list.append(snr_db)

    return X_list, y_list, freq_list, snr_list

def load_all_signals(data_dir):
    """CSV 파일 로드"""
    data_path = Path(data_dir)
    csv_files = sorted(list(data_path.glob("*.csv")))
    signals = []

    for idx, file_path in enumerate(csv_files):
        try:
            df = pd.read_csv(file_path)
            if 'Accel_Z' not in df.columns: continue
            sample_rate = calculate_sample_rate(df)
            signal = df['Accel_Z'].dropna().values
            signals.append({
                'signal': signal,
                'sample_rate': sample_rate,
                'file_name': file_path.name,
                'signal_id': idx,
                'is_augmented': False,
                'snr_db': None
            })
        except Exception:
            continue
    return signals

def calculate_augmentation_multiplier(class_counts, strategy='balanced'):
    """
    클래스 불균형 기반 증강 배수 계산 (더 강력한 버전)
    
    Args:
        class_counts: 각 클래스의 샘플 개수
        strategy: 'balanced' (모든 클래스를 최대값에 맞춤 - 더 공격적)
    """
    multipliers = {}
    max_count = np.max(class_counts)
    
    for class_id, count in enumerate(class_counts):
        # 모든 클래스를 최대값에 맞춤 (더 공격적인 오버샘플링)
        multipliers[class_id] = max_count / count
    
    return multipliers

def create_classification_dataset_fixed(data_dir, output_dir, window_size=WINDOW_SIZE,
                                        stride=STRIDE, test_size=TEST_SIZE, random_state=RANDOM_STATE):
                                        
    """클래스별 균등 증강을 적용한 학습/테스트 데이터셋 생성 함수"""
    signals = load_all_signals(data_dir)

    signal_indices = np.arange(len(signals))
    
    # 각 신호의 대표 주파수를 먼저 계산하여 클래스 레이블 결정 (Stratified Sampling용)
    signal_labels = []
    for signal_data in signals:
        windows = sliding_window_split(signal_data['signal'], window_size=window_size, stride=256)
        if len(windows) > 0:
            _, peak_freq = extract_frequency_features(windows[0], signal_data['sample_rate'])
            label = assign_label(peak_freq)
            signal_labels.append(int(label))
        else:
            signal_labels.append(0)
    
    # Stratified Sampling: 클래스별로 균형 있게 Train/Test 분리 (파일 개수가 너무 적으면 Random으로 변경)
    signal_labels_array = np.array(signal_labels, dtype=int)
    class_counts_signals = np.bincount(signal_labels_array)
    
    # 클래스별 파일이 2개 미만이면 Random Split 사용
    min_count = np.min(class_counts_signals)
    if min_count >= 2:
        train_indices, test_indices = train_test_split(
            signal_indices, test_size=test_size, random_state=random_state, stratify=signal_labels_array
        )
    else:
        # Random Split (Stratified 불가)
        print(f"경고: 일부 클래스의 파일이 2개 미만이어서 Random Split 사용합니다 (최소값: {min_count})")
        train_indices, test_indices = train_test_split(
            signal_indices, test_size=test_size, random_state=random_state
        )

    X_train_all, y_train_all = [], []
    freq_train_all = []
    snr_train_all = []
    
    # 1단계: 원본 훈련 데이터 수집
    print("[단계 1] 원본 훈련 데이터 수집 중...")
    for idx in train_indices:
        signal_data = signals[idx]
        X_list, y_list, freq_list, snr_list = process_single_signal(
            signal_data['signal'], signal_data['sample_rate'], window_size, stride, signal_data['file_name'], None
        )
        X_train_all.extend(X_list)
        y_train_all.extend(y_list)
        freq_train_all.extend(freq_list)
        snr_train_all.extend(snr_list)

    # 2단계: 클래스 불균형 분석
    y_train_temp = np.array(y_train_all)
    class_counts = np.bincount(y_train_temp, minlength=len(RESONANCE_FREQS))
    print(f"\n[증강 전 훈련 레이블 분포]")
    for class_id, count in enumerate(class_counts):
        print(f"  클래스 {class_id} ({RESONANCE_FREQS[class_id]}Hz): {count} 샘플")
    
    # 3단계: 클래스별 균등 증강 배수 계산
    augmentation_multipliers = calculate_augmentation_multiplier(class_counts, strategy='balanced')
    print(f"\n[클래스별 증강 배수 (모두 최대값으로 정렬)]")
    for class_id, multiplier in augmentation_multipliers.items():
        print(f"  클래스 {class_id} ({RESONANCE_FREQS[class_id]}Hz): {multiplier:.2f}배")
    
    # 4단계: SNR 기반 클래스별 균등 증강
    if ENABLE_AUGMENTATION:
        print(f"\n[단계 2] SNR 기반 클래스별 균등 증강 중...")
        
        # 훈련 신호들을 클래스별로 분류
        signal_class_map = {}
        for array_idx, file_idx in enumerate(train_indices):
            signal_data = signals[file_idx]
            X_list, y_list, _, _ = process_single_signal(
                signal_data['signal'], signal_data['sample_rate'], window_size, stride, 
                signal_data['file_name'], None
            )
            if len(y_list) > 0:
                signal_class = np.bincount(y_list).argmax()
                if signal_class not in signal_class_map:
                    signal_class_map[signal_class] = []
                signal_class_map[signal_class].append(file_idx)
        
        # 각 클래스별로 배수만큼 증강
        for class_id, file_indices in signal_class_map.items():
            multiplier = augmentation_multipliers[class_id]
            # 배수만큼 SNR 기반 증강 반복
            augment_count = int(np.ceil((multiplier - 1) * len(file_indices)))
            print(f"  클래스 {class_id} ({RESONANCE_FREQS[class_id]}Hz): {augment_count}개 파일 증강")
            
            for _ in range(augment_count):
                for file_idx in file_indices:
                    signal_data = signals[file_idx]
                    for snr_db in AUGMENTATION_SNR_DB:
                        augmented_signal = add_gaussian_noise_with_snr(signal_data['signal'], snr_db)
                        X_aug, y_aug, freq_aug, snr_aug = process_single_signal(
                            augmented_signal, signal_data['sample_rate'], window_size, stride,
                            f"{signal_data['file_name']}_aug_SNR{snr_db}dB",
                            snr_db
                        )
                        X_train_all.extend(X_aug)
                        y_train_all.extend(y_aug)
                        freq_train_all.extend(freq_aug)
                        snr_train_all.extend(snr_aug)

    # 테스트 데이터 (증강 없음)
    print(f"\n[단계 3] 테스트 데이터 수집 중...")
    X_test_all, y_test_all = [], []
    freq_test_all = []
    snr_test_all = []
    
    for idx in test_indices:
        signal_data = signals[idx]
        X_list, y_list, freq_list, snr_list = process_single_signal(
            signal_data['signal'], signal_data['sample_rate'], window_size, stride, 
            signal_data['file_name'], signal_data['snr_db']
        )
        X_test_all.extend(X_list)
        y_test_all.extend(y_list)
        freq_test_all.extend(freq_list)
        snr_test_all.extend(snr_list)

    # 최종 데이터 정리
    X_train, y_train = np.array(X_train_all), np.array(y_train_all)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test, y_test = np.array(X_test_all), np.array(y_test_all)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    freq_train = np.array(freq_train_all)
    freq_test = np.array(freq_test_all)
    snr_train = np.array(snr_train_all)
    snr_test = np.array(snr_test_all)
    
    # 최종 분포 확인
    print(f"\n[증강 후 최종 훈련 레이블 분포]")
    y_train_counts = np.bincount(y_train, minlength=len(RESONANCE_FREQS))
    for class_id, count in enumerate(y_train_counts):
        ratio = count / np.sum(y_train_counts) * 100
        print(f"  클래스 {class_id} ({RESONANCE_FREQS[class_id]}Hz): {count} 샘플 ({ratio:.1f}%)")
    
    print(f"\n[테스트 레이블 분포]")
    y_test_counts = np.bincount(y_test, minlength=len(RESONANCE_FREQS))
    for class_id, count in enumerate(y_test_counts):
        ratio = count / np.sum(y_test_counts) * 100
        print(f"  클래스 {class_id} ({RESONANCE_FREQS[class_id]}Hz): {count} 샘플 ({ratio:.1f}%)")
    
    return X_train, y_train, X_test, y_test, freq_train, freq_test, snr_train, snr_test


if __name__ == "__main__":
    print("=" * 70)
    print("강력한 클래스 불균형 처리 FFT 데이터 전처리 시작")
    print("=" * 70)
    
    X_train, y_train, X_test, y_test, freq_train, freq_test, snr_train, snr_test = \
        create_classification_dataset_fixed(DATA_DIR, OUTPUT_DIR, WINDOW_SIZE, STRIDE)
    
    print("\n[데이터셋 생성 완료]")
    print(f"훈련 데이터: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"테스트 데이터: X_test={X_test.shape}, y_test={y_test.shape}")
    print(f"입력 주파수: freq_train={freq_train.shape}, freq_test={freq_test.shape}")
