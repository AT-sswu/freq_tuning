"""
60Hz 고정 SNR 실험 스크립트

각 SNR 레벨(-10, -5, 0, 5, 10dB)로만 훈련하고,
테스트는 항상 SNR 0dB로 평가하는 실험
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.fft import fft, fftfreq
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import pickle
import time

# 설정
DATA_DIR = "/Users/seohyeon/AT_freq_tuning/data_v3"
OUTPUT_DIR = "/Users/seohyeon/AT_freq_tuning/snr_experiment_results"
WINDOW_SIZE = 1024
STRIDE = 512
TEST_SIZE = 0.2
RANDOM_STATE = 42
TARGET_FREQ = 60  # 60Hz만 사용

# SNR 레벨 (각각 독립적으로 훈련)
SNR_LEVELS = [-10, -5, 0, 5, 10]

# 모델 설정
MODELS = {
    'SVM': SVC(kernel='rbf', C=10, gamma='scale', random_state=42, class_weight='balanced'),
    'RF': RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, class_weight='balanced'),
    'KNN': KNeighborsClassifier(n_neighbors=5, weights='distance'),
    'XGBoost': XGBClassifier(max_depth=5, learning_rate=0.1, n_estimators=100, 
                             objective='binary:logistic', random_state=42,
                             subsample=0.8, colsample_bytree=0.8)
}

def add_gaussian_noise_with_snr(signal, snr_db):
    """SNR(dB)을 기준으로 가우시안 노이즈를 추가"""
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power)
    noise = np.random.normal(0, noise_std, len(signal))
    return signal + noise

def calculate_sample_rate(df, time_column='Time_us'):
    """샘플링 주파수 계산"""
    if time_column not in df.columns:
        time_column = df.columns[0]
    time_data = df[time_column].dropna().values
    time_diffs = np.diff(time_data)
    avg_time_diff_us = np.mean(time_diffs)
    return 1 / (avg_time_diff_us / 1_000_000)

def extract_frequency_features(signal, sample_rate, target_freq=TARGET_FREQ):
    """Q-factor 기반 주파수 특징 추출 (단일 주파수)"""
    n = len(signal)
    fft_vals = fft(signal)
    freqs = fftfreq(n, 1 / sample_rate)
    
    positive_mask = freqs > 0
    freqs = freqs[positive_mask]
    fft_vals = np.abs(fft_vals[positive_mask])
    
    # 정규화
    fft_energy_sum = np.sum(fft_vals ** 2)
    if fft_energy_sum > 0:
        fft_vals_normalized = fft_vals / np.sqrt(fft_energy_sum)
    else:
        fft_vals_normalized = fft_vals
    
    # Q-factor 기반 대역폭 (Q=10)
    q_factor = 10
    bandwidth = target_freq / q_factor
    
    # 60Hz 대역 에너지
    band_mask = (freqs >= target_freq - bandwidth/2) & (freqs <= target_freq + bandwidth/2)
    band_energy = np.sum(fft_vals_normalized[band_mask] ** 2)
    
    # 피크 주파수
    peak_idx = np.argmax(fft_vals)
    peak_frequency = freqs[peak_idx]
    
    return band_energy, peak_frequency

def is_60hz_signal(peak_frequency, tolerance=5):
    """60Hz 신호인지 판단 (±5Hz 허용)"""
    return abs(peak_frequency - TARGET_FREQ) <= tolerance

def sliding_window_split(signal, window_size, stride):
    """슬라이딩 윈도우 분할"""
    windows = []
    for start in range(0, len(signal) - window_size + 1, stride):
        windows.append(signal[start:start + window_size])
    return windows

def load_60hz_signals_only(data_dir):
    """60Hz 신호만 필터링하여 로드"""
    data_path = Path(data_dir)
    csv_files = sorted(list(data_path.glob("*.csv")))
    signals_60hz = []
    
    print(f"전체 파일 수: {len(csv_files)}")
    
    for idx, file_path in enumerate(csv_files):
        try:
            df = pd.read_csv(file_path)
            if 'Accel_Z' not in df.columns:
                continue
            
            sample_rate = calculate_sample_rate(df)
            signal = df['Accel_Z'].dropna().values
            
            # 첫 번째 윈도우로 주파수 확인
            if len(signal) >= WINDOW_SIZE:
                first_window = signal[:WINDOW_SIZE]
                _, peak_freq = extract_frequency_features(first_window, sample_rate)
                
                # 60Hz 신호만 추가
                if is_60hz_signal(peak_freq):
                    signals_60hz.append({
                        'signal': signal,
                        'sample_rate': sample_rate,
                        'file_name': file_path.name,
                        'peak_freq': peak_freq
                    })
                    print(f"  ✓ {file_path.name}: {peak_freq:.1f}Hz (60Hz 신호)")
        except Exception as e:
            print(f"  ✗ {file_path.name}: 로드 실패 - {e}")
            continue
    
    print(f"\n60Hz 신호 파일 수: {len(signals_60hz)}")
    return signals_60hz

def create_snr_experiment_dataset(signals, snr_train, snr_test=0):
    """
    특정 SNR로 훈련 데이터 생성, SNR 0dB로 테스트 데이터 생성
    
    Args:
        signals: 60Hz 신호 리스트
        snr_train: 훈련 데이터에 적용할 SNR (dB)
        snr_test: 테스트 데이터에 적용할 SNR (dB, 기본값 0)
    """
    # Train/Test Split (파일 단위)
    train_signals, test_signals = train_test_split(
        signals, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    
    # 훈련 데이터: 특정 SNR 적용
    for sig_data in train_signals:
        signal_noisy = add_gaussian_noise_with_snr(sig_data['signal'], snr_train)
        windows = sliding_window_split(signal_noisy, WINDOW_SIZE, STRIDE)
        
        for window in windows:
            feature, peak_freq = extract_frequency_features(window, sig_data['sample_rate'])
            if is_60hz_signal(peak_freq):
                X_train_list.append([feature])
                y_train_list.append(1)  # 60Hz = 1
    
    # 테스트 데이터: SNR 0dB 고정
    for sig_data in test_signals:
        signal_noisy = add_gaussian_noise_with_snr(sig_data['signal'], snr_test)
        windows = sliding_window_split(signal_noisy, WINDOW_SIZE, STRIDE)
        
        for window in windows:
            feature, peak_freq = extract_frequency_features(window, sig_data['sample_rate'])
            if is_60hz_signal(peak_freq):
                X_test_list.append([feature])
                y_test_list.append(1)
    
    X_train = np.array(X_train_list).reshape(-1, 1)
    y_train = np.array(y_train_list)
    X_test = np.array(X_test_list).reshape(-1, 1)
    y_test = np.array(y_test_list)
    
    return X_train, y_train, X_test, y_test

def train_and_evaluate(model_name, model, X_train, y_train, X_test, y_test, snr_train):
    """모델 훈련 및 평가"""
    print(f"\n{'='*70}")
    print(f"모델: {model_name} | 훈련 SNR: {snr_train}dB | 테스트 SNR: 0dB")
    print(f"{'='*70}")
    
    # 스케일링 (k-NN용)
    if model_name == 'KNN':
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
    
    # 학습
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    train_time = time.time() - start_time
    
    # 예측
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # 평가
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred_test, average='binary', zero_division=0
    )
    
    print(f"훈련 정확도: {train_acc:.4f}")
    print(f"테스트 정확도: {test_acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"훈련 시간: {train_time:.2f}초")
    
    return {
        'model_name': model_name,
        'snr_train': snr_train,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'train_time': train_time
    }

def main():
    """메인 실험 루프"""
    print("="*70)
    print("60Hz 고정 SNR 실험 시작")
    print("="*70)
    
    # 출력 디렉토리 생성
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. 60Hz 신호만 로드
    print("\n[단계 1] 60Hz 신호 필터링 중...")
    signals_60hz = load_60hz_signals_only(DATA_DIR)
    
    if len(signals_60hz) == 0:
        print("오류: 60Hz 신호를 찾을 수 없습니다!")
        return
    
    # 2. 각 SNR × 각 모델 실험
    all_results = []
    
    for snr_train in SNR_LEVELS:
        print(f"\n{'='*70}")
        print(f"훈련 SNR: {snr_train}dB")
        print(f"{'='*70}")
        
        # 데이터셋 생성
        X_train, y_train, X_test, y_test = create_snr_experiment_dataset(
            signals_60hz, snr_train, snr_test=0
        )
        
        print(f"훈련 데이터: {X_train.shape[0]} 샘플")
        print(f"테스트 데이터: {X_test.shape[0]} 샘플")
        
        # 각 모델 훈련
        for model_name, model in MODELS.items():
            result = train_and_evaluate(
                model_name, model, X_train, y_train, X_test, y_test, snr_train
            )
            all_results.append(result)
    
    # 3. 결과 저장
    results_df = pd.DataFrame(all_results)
    results_path = output_path / "snr_experiment_60hz_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n결과 저장: {results_path}")
    
    # 4. 결과 요약
    print("\n" + "="*70)
    print("실험 결과 요약")
    print("="*70)
    print(results_df.to_string(index=False))
    
    # 5. 모델별 최고 성능 SNR
    print("\n" + "="*70)
    print("모델별 최고 테스트 정확도")
    print("="*70)
    best_by_model = results_df.loc[results_df.groupby('model_name')['test_acc'].idxmax()]
    print(best_by_model[['model_name', 'snr_train', 'test_acc']].to_string(index=False))

if __name__ == "__main__":
    main()
