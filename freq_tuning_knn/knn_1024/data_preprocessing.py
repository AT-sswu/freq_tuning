import pandas as pd
import numpy as np
from pathlib import Path
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split
from config import DATA_DIR, OUTPUT_DIR, RESONANCE_FREQS, WINDOW_SIZE, STRIDE, TEST_SIZE, RANDOM_STATE, AUGMENTATION_SNR_DB, ENABLE_AUGMENTATION

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
    
    # 각 공진 주파수별 특징 추출
    # 1) 좁은 대역 (±3Hz): 정확한 주파수 성분 감지
    # 2) 넓은 대역 (±5Hz): 주변 대역 고려
    # 3) 더 넓은 대역 (±10Hz): 광역 에너지 포함
    for target_freq in target_bands:
        # ±3Hz 대역
        band_mask_tight = (freqs >= target_freq - 3) & (freqs <= target_freq + 3)
        band_energy_tight = np.sum(fft_vals_normalized[band_mask_tight] ** 2)
        features.append(band_energy_tight)
        
        # ±5Hz 대역
        band_mask_mid = (freqs >= target_freq - 5) & (freqs <= target_freq + 5)
        band_energy_mid = np.sum(fft_vals_normalized[band_mask_mid] ** 2)
        features.append(band_energy_mid)
        
        # ±10Hz 대역
        band_mask_wide = (freqs >= target_freq - 10) & (freqs <= target_freq + 10)
        band_energy_wide = np.sum(fft_vals_normalized[band_mask_wide] ** 2)
        features.append(band_energy_wide)
    
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

def load_all_signals(data_dir):
    """지정된 경로의 모든 CSV 파일을 읽어와 물리적 의미가 있는 신호 객체로 변환"""
    data_path = Path(data_dir)
    csv_files = sorted(list(data_path.glob("*.csv")))

    signals = []

    for idx, file_path in enumerate(csv_files):
        try:
            df = pd.read_csv(file_path)
            if 'Accel_Z' not in df.columns: continue

            sample_rate = calculate_sample_rate(df)
            signal = df['Accel_Z'].dropna().values

            # 원본 신호 추가
            signals.append({
                'signal': signal,
                'sample_rate': sample_rate,
                'file_name': file_path.name,
                'signal_id': idx,  # 파일 단위 분할을 위해 고유 ID 부여
                'is_augmented': False,
                'snr_db': None
            })

            # 데이터 증강 적용 (SNR 기반 가우시안 노이즈)
            if ENABLE_AUGMENTATION:
                for snr_db in AUGMENTATION_SNR_DB:
                    augmented_signal = add_gaussian_noise_with_snr(signal, snr_db)
                    signals.append({
                        'signal': augmented_signal,
                        'sample_rate': sample_rate,
                        'file_name': f"{file_path.name}_aug_SNR{snr_db}dB",
                        'signal_id': idx,  # 동일 파일 그룹으로 유지 (데이터 누수 방지)
                        'is_augmented': True,
                        'snr_db': snr_db
                    })
        except Exception:
            continue
    return signals

def create_classification_dataset_fixed(data_dir, output_dir, window_size=WINDOW_SIZE,
                                        stride=STRIDE, test_size=TEST_SIZE, random_state=RANDOM_STATE):
                                        
    """[핵심 로직] 데이터 누수 방지를 고려한 학습/테스트 데이터셋 통합 생성 함수"""
    signals = load_all_signals(data_dir)

    # 데이터 누수(Data Leakage) 방지: 윈도우 단위로 섞으면 동일 파일의 파편이 Train/Test에 동시에 들어감
    # 이를 막기 위해 파일(Signal ID) 자체를 먼저 분리한 후, 분리된 그룹 내에서만 윈도우를 추출함
    signal_indices = np.arange(len(signals))
    train_indices, test_indices = train_test_split(
        signal_indices, test_size=test_size, random_state=random_state
    )

    X_train_all, y_train_all = [], []
    freq_train_all = []  # 입력 주파수 추적
    snr_train_all = []  # 훈련 데이터 SNR 메타데이터
    for idx in train_indices:
        signal_data = signals[idx]
        X_list, y_list, freq_list, snr_list = process_single_signal(
            signal_data['signal'], signal_data['sample_rate'], window_size, stride, signal_data['file_name'], signal_data['snr_db']
        )
        X_train_all.extend(X_list); y_train_all.extend(y_list)
        freq_train_all.extend(freq_list)
        snr_train_all.extend(snr_list)

    X_test_all, y_test_all = [], []
    freq_test_all = []  # 입력 주파수 추적
    snr_test_all = []  # 테스트 데이터 SNR 메타데이터
    for idx in test_indices:
        signal_data = signals[idx]

        # 테스트 셋은 학습 셋에서 보지 못한 독립적인 파일(신호)들로 구성됨
        X_list, y_list, freq_list, snr_list = process_single_signal(
            signal_data['signal'], signal_data['sample_rate'], window_size, stride, signal_data['file_name'], signal_data['snr_db']
        )
        X_test_all.extend(X_list); y_test_all.extend(y_list)
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
    print(f"훈련 레이블 분포: {np.bincount(y_train)}")
    print(f"테스트 레이블 분포: {np.bincount(y_test)}")
    
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

