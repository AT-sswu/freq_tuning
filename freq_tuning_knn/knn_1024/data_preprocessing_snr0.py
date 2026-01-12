import pandas as pd
import numpy as np
from pathlib import Path
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split
from config import (
    DATA_DIR,
    OUTPUT_DIR,
    RESONANCE_FREQS,
    WINDOW_SIZE,
    STRIDE,
    TEST_SIZE,
    RANDOM_STATE,
    AUGMENTATION_SNR_DB,
    ENABLE_AUGMENTATION,
    Q_FACTOR,
)

def add_gaussian_noise_with_snr(signal, snr_db):
    """SNR(dB)을 기준으로 가우시안 노이즈를 추가하여 신호 증강"""
    # 신호 전력 계산
    signal_power = np.mean(signal ** 2)
    
    # SNR을 선형으로 변환
    snr_linear = 10 ** (snr_db / 10)
    
    # 노이즈 전력 계산
    noise_power = signal_power / snr_linear
    
    # 노이즈 표준편차
    noise_std = np.sqrt(noise_power)
    
    # 가우시안 노이즈 생성
    noise = np.random.normal(0, noise_std, len(signal))
    
    # 노이즈 추가
    augmented_signal = signal + noise
    
    return augmented_signal

def calculate_sample_rate(df, time_column='Time_us'):
    """데이터 기반 샘플링 주파수 계산: 센서 데이터의 실제 수집 간격을 파악함"""
    if time_column not in df.columns:
        time_column = df.columns[0]
    time_data = df[time_column].dropna().values  # 시간 데이터에서 결측치 제거 후 추출
    time_diffs = np.diff(time_data)  # 인접한 샘플 사이의 시간 간격(dt)들을 계산
    avg_time_diff_us = np.mean(time_diffs)  # 개별 샘플링 오차를 줄이기 위해 전체 간격의 평균을 사용
    sample_rate = 1 / (avg_time_diff_us / 1_000_000)  # 마이크로초(us) 단위를 초(s)로 환산하여 1초당 샘플 수(Hz) 도출
    return sample_rate


def extract_frequency_features(signal, sample_rate, target_bands=RESONANCE_FREQS):
    """FFT 기반 주파수 특징 추출: 목표 주파수와 그 인접 대역의 에너지를 추출"""
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
    
    # Q-factor 기반 대역폭 계산: BW = target_freq / Q_FACTOR
    for target_freq in target_bands:
        # Q-factor로부터 대역폭 계산
        bandwidth = target_freq / Q_FACTOR
        half_bw = bandwidth / 2
        
        # ±half_bw 범위로 에너지 계산
        band_mask = (freqs >= target_freq - half_bw) & (freqs <= target_freq + half_bw)
        band_energy = np.sum(fft_vals_normalized[band_mask] ** 2)
        features.append(band_energy)
    
    # 지배 주파수 (dominant frequency)
    peak_idx = np.argmax(fft_vals)
    peak_frequency = freqs[peak_idx]
    
    return np.array(features), peak_frequency


def assign_label(peak_frequency):
    """지배 주파수(peak_frequency)를 가장 가까운 공진 주파수로 매핑하여 클래스 할당"""
    res_array = np.array(RESONANCE_FREQS)
    label = np.argmin(np.abs(res_array - peak_frequency))
    return label

def sliding_window_split(signal, window_size=1024, stride=512):
    """연속 신호를 분석 단위인 윈도우로 분할: Stride를 작게 주어 윈도우 간 Overlap(중첩) 발생 가능"""
    windows = [] 
    # 신호의 시작부터 끝까지 지정된 보폭(Stride)만큼 건너뛰며 데이터를 슬라이싱함
    for start in range(0, len(signal) - window_size + 1, stride):
        window = signal[start:start + window_size]
        windows.append(window)
    return windows

def process_single_signal(signal, sample_rate, window_size, stride, file_name, snr_db=None):
    """단일 파일에서 읽어온 신호 전처리 및 특징 추출 파이프라인 수행"""
    windows = sliding_window_split(signal, window_size=window_size, stride=stride)

    X_list, y_list = [], []
    freq_list = []  # 입력 주파수 추적
    snr_list = []  # SNR 메타데이터 추적

    for window in windows:
        features, peak_frequency = extract_frequency_features(window, sample_rate)
        label = assign_label(peak_frequency)

        X_list.append(features)
        y_list.append(label)
        freq_list.append(peak_frequency)  # 입력 주파수 저장
        snr_list.append(snr_db)

    return X_list, y_list, freq_list, snr_list

def load_all_signals(data_dir, allowed_snrs=None):
    """지정된 경로의 모든 CSV 파일을 읽어와 물리적 의미가 있는 신호 객체로 변환

    allowed_snrs: 리스트로 전달하면 해당 SNR(또는 None=원본)만 포함
                 예: allowed_snrs=[-10] → 원본 + SNR -10dB 증강만
                 예: allowed_snrs=None → 원본만 (증강 없음)
    """
    data_path = Path(data_dir)
    csv_files = sorted(list(data_path.glob("*.csv")))

    signals = []

    for idx, file_path in enumerate(csv_files):
        try:
            df = pd.read_csv(file_path)
            if 'Accel_Z' not in df.columns: continue

            sample_rate = calculate_sample_rate(df)
            signal = df['Accel_Z'].dropna().values

            # 원본 신호 추가 (항상 포함)
            signals.append({
                'signal': signal,
                'sample_rate': sample_rate,
                'file_name': file_path.name,
                'signal_id': idx,
                'is_augmented': False,
                'snr_db': None
            })

            # 데이터 증강 적용 (SNR 기반 가우시안 노이즈)
            if ENABLE_AUGMENTATION and allowed_snrs is not None:
                for snr_db in AUGMENTATION_SNR_DB:
                    # allowed_snrs에 포함된 SNR만 생성
                    if snr_db in allowed_snrs:
                        augmented_signal = add_gaussian_noise_with_snr(signal, snr_db)
                        signals.append({
                            'signal': augmented_signal,
                            'sample_rate': sample_rate,
                            'file_name': f"{file_path.name}_aug_SNR{snr_db}dB",
                            'signal_id': idx,
                            'is_augmented': True,
                            'snr_db': snr_db
                        })
        except Exception:
            continue
    return signals

def create_classification_dataset_fixed(data_dir, output_dir, window_size=WINDOW_SIZE,
                                        stride=STRIDE, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                        allowed_snrs=None):
                                        
    """[핵심 로직] 데이터 누수 방지를 고려한 학습/테스트 데이터셋 통합 생성 함수
    
    수정 사항:
    - 훈련: 원본 + 해당 SNR 증강 데이터만 사용 (예: SNR -10dB 실험 → 원본 + SNR -10dB만)
    - 테스트: 원본 데이터만 사용
    - allowed_snrs: 사용할 SNR 레벨 리스트 (예: [-10] → 원본 + SNR -10dB만)
    """
    # 원본 + 지정된 SNR 증강만 로드
    signals = load_all_signals(data_dir, allowed_snrs=allowed_snrs)
    
    # 원본 신호만 추출하여 파일 ID 기준으로 Train/Test 분할
    original_signals = [s for s in signals if not s['is_augmented']]
    unique_signal_ids = list(set(s['signal_id'] for s in original_signals))
    
    # 파일 ID 기준으로 Train/Test 분할
    train_ids, test_ids = train_test_split(
        unique_signal_ids, test_size=test_size, random_state=random_state
    )
    
    train_ids_set = set(train_ids)
    test_ids_set = set(test_ids)

    # 훈련 데이터 - 원본 + 해당 SNR 증강만 사용
    X_train_all, y_train_all = [], []
    freq_train_all = []
    snr_train_all = []
    
    for signal_data in signals:
        if signal_data['signal_id'] in train_ids_set:
            # 훈련 파일에 속한 신호 (원본 + 해당 SNR 증강만)
            X_list, y_list, freq_list, snr_list = process_single_signal(
                signal_data['signal'], signal_data['sample_rate'], window_size, stride, 
                signal_data['file_name'], signal_data['snr_db']
            )
            X_train_all.extend(X_list)
            y_train_all.extend(y_list)
            freq_train_all.extend(freq_list)
            snr_train_all.extend(snr_list)

    # 테스트 데이터 - 원본만 사용
    X_test_all, y_test_all = [], []
    freq_test_all = []
    snr_test_all = []
    
    for signal_data in original_signals:
        if signal_data['signal_id'] in test_ids_set:
            # 테스트는 항상 원본만 사용
            X_list, y_list, freq_list, snr_list = process_single_signal(
                signal_data['signal'], signal_data['sample_rate'], window_size, stride, 
                signal_data['file_name'], signal_data['snr_db']
            )
            X_test_all.extend(X_list)
            y_test_all.extend(y_list)
            freq_test_all.extend(freq_list)
            snr_test_all.extend(snr_list)

    # 머신러닝 모델의 입력을 위해 최종 데이터를 넘파이 행렬 형태로 변환
    X_train, y_train = np.array(X_train_all), np.array(y_train_all)
    X_train = X_train.reshape(X_train.shape[0], -1)  # 2D로 강제 변환
    X_test, y_test = np.array(X_test_all), np.array(y_test_all)
    X_test = X_test.reshape(X_test.shape[0], -1)  # 2D로 강제 변환
    
    # 입력 주파수 및 SNR 메타데이터도 배열로 변환
    freq_train = np.array(freq_train_all)
    freq_test = np.array(freq_test_all)
    snr_train = np.array(snr_train_all)
    snr_test = np.array(snr_test_all)
    
    # 레이블 분포 확인 (디버깅용)
    print(f"훈련 레이블 분포 (증강 전): {np.bincount(y_train)}")
    print(f"테스트 레이블 분포: {np.bincount(y_test)}")
    
    # 클래스 불균형 해결: 각 클래스 샘플 수를 최대 클래스에 맞추도록 증강
    # (Oversampling: 소수 클래스를 반복 추가)
    classes, counts = np.unique(y_train, return_counts=True)
    max_count = np.max(counts)
    
    X_train_balanced = []
    y_train_balanced = []
    freq_train_balanced = []
    snr_train_balanced = []
    
    for cls in classes:
        mask = y_train == cls
        X_cls = X_train[mask]
        y_cls = y_train[mask]
        freq_cls = freq_train[mask]
        snr_cls = snr_train[mask]
        
        current_count = len(X_cls)
        
        if current_count < max_count:
            # 부족한 샘플 수 계산
            needed = max_count - current_count
            repeat_times = int(np.ceil(needed / current_count))
            
            # 부족분만 반복하여 추가
            indices_to_repeat = np.random.choice(current_count, size=needed, replace=True)
            X_cls = np.vstack([X_cls, X_cls[indices_to_repeat]])
            y_cls = np.concatenate([y_cls, y_cls[indices_to_repeat]])
            freq_cls = np.concatenate([freq_cls, freq_cls[indices_to_repeat]])
            snr_cls = np.concatenate([snr_cls, snr_cls[indices_to_repeat]])
        
        X_train_balanced.append(X_cls[:max_count])
        y_train_balanced.append(y_cls[:max_count])
        freq_train_balanced.append(freq_cls[:max_count])
        snr_train_balanced.append(snr_cls[:max_count])
    
    # 배열로 변환
    X_train = np.vstack(X_train_balanced)
    y_train = np.concatenate(y_train_balanced)
    freq_train = np.concatenate(freq_train_balanced)
    snr_train = np.concatenate(snr_train_balanced)
    
    print(f"훈련 레이블 분포 (증강 후): {np.bincount(y_train)}")
    
    return X_train, y_train, X_test, y_test, freq_train, freq_test, snr_train, snr_test


if __name__ == "__main__":
    print("=" * 70)
    print("FFT 기반 데이터 전처리 시작")
    print("=" * 70)
    
    # 데이터셋 생성
    X_train, y_train, X_test, y_test, freq_train, freq_test, snr_train, snr_test = \
        create_classification_dataset_fixed(DATA_DIR, OUTPUT_DIR)
    
    print("\n[데이터셋 생성 완료]")
    print(f"훈련 데이터: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"테스트 데이터: X_test={X_test.shape}, y_test={y_test.shape}")
    print(f"입력 주파수: freq_train={freq_train.shape}, freq_test={freq_test.shape}")
