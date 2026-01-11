import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
import joblib
from pathlib import Path
from data_preprocessing import create_classification_dataset_single_snr
from config import DATA_DIR, OUTPUT_DIR, MODEL_DIR

def train_knn_0db():
    """k-NN 모델 SNR 0dB 고정 학습 및 주파수별 성능 분석"""
    print("=" * 70)
    print("k-NN SNR 0dB 고정 학습 (주파수별 성능 분석)")
    print("=" * 70)
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    
    # SNR 0dB로 데이터셋 생성
    print("\nSNR 0dB 데이터셋 생성 중...")
    X_train, y_train, X_test, y_test = create_classification_dataset_single_snr(
        data_dir=DATA_DIR,
        output_dir=OUTPUT_DIR,
        snr_level=0
    )
    
    print(f"✓ 데이터: 훈련 {X_train.shape}, 테스트 {X_test.shape}")
    
    # 클래스 분포
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    
    print(f"\n훈련 클래스 분포:")
    freq_map = {0: '40Hz', 1: '50Hz', 2: '60Hz'}
    for cls, count in zip(unique_train, counts_train):
        print(f"  {freq_map[cls]}: {count}개 ({count/len(y_train)*100:.2f}%)")
    
    print(f"\n테스트 클래스 분포:")
    for cls, count in zip(unique_test, counts_test):
        print(f"  {freq_map[cls]}: {count}개 ({count/len(y_test)*100:.2f}%)")
    
    # 특징 정규화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # k-NN 학습
    print(f"\nk-NN 학습 중 (k=5, weights=distance)...")
    knn = KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
    knn.fit(X_train_scaled, y_train)
    
    # 예측
    y_pred_train = knn.predict(X_train_scaled)
    y_pred_test = knn.predict(X_test_scaled)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    print(f"\n훈련 정확도: {train_acc:.4f}")
    print(f"테스트 정확도: {test_acc:.4f}")
    
    # 혼동 행렬
    cm = confusion_matrix(y_test, y_pred_test)
    print(f"\n혼동 행렬:")
    print(f"              예측 40Hz  예측 50Hz  예측 60Hz")
    for i, freq in enumerate(['실제 40Hz', '실제 50Hz', '실제 60Hz']):
        print(f"{freq:12s}  {cm[i][0]:9d}  {cm[i][1]:9d}  {cm[i][2]:9d}")
    
    # 주파수별 상세 성능
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred_test, average=None, zero_division=0
    )
    
    print(f"\n주파수별 성능:")
    results = []
    for i, freq in enumerate(['40Hz', '50Hz', '60Hz']):
        print(f"\n{freq}:")
        print(f"  Precision: {precision[i]:.4f}")
        print(f"  Recall:    {recall[i]:.4f}")
        print(f"  F1-Score:  {f1[i]:.4f}")
        print(f"  Support:   {support[i]}")
        
        results.append({
            'model': 'k-NN',
            'snr_db': 0,
            'frequency': freq,
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
            'support': int(support[i]),
            'accuracy': test_acc
        })
    
    # 모델 저장
    model_filename = Path(MODEL_DIR) / "knn_snr_0db_fixed.pkl"
    joblib.dump({'model': knn, 'scaler': scaler}, model_filename)
    print(f"\n✓ 모델 저장: {model_filename}")
    
    # 결과 저장
    results_df = pd.DataFrame(results)
    results_csv = Path(OUTPUT_DIR) / "knn_0db_frequency_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"✓ 결과 저장: {results_csv}")
    
    return results_df

if __name__ == "__main__":
    train_knn_0db()
