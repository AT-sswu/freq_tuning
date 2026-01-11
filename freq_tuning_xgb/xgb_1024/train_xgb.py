import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import joblib
from pathlib import Path
from data_preprocessing import create_classification_dataset_single_snr
from config import DATA_DIR, OUTPUT_DIR, MODEL_DIR

SNR_LEVELS = [-10, -5, 0, 5, 10]

def train_xgboost_model():
    """XGBoost 모델 SNR별 학습 실험"""
    print("=" * 70)
    print("XGBoost SNR별 학습 실험 (각 SNR마다 독립적으로 데이터 생성)")
    print("=" * 70)
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for snr_level in SNR_LEVELS:
        print(f"\n{'='*70}")
        print(f"SNR {snr_level}dB 학습 시작")
        print(f"{'='*70}")
        
        # 해당 SNR로 독립적으로 데이터 생성
        X_train, y_train, X_test, y_test = create_classification_dataset_single_snr(
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            snr_level=snr_level
        )
        
        print(f"\n✓ SNR {snr_level}dB 데이터: 훈련 {X_train.shape}, 테스트 {X_test.shape}")
        
        # 클래스 분포 출력
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        
        print(f"\n훈련 클래스 분포:")
        for freq, count in zip(unique_train, counts_train):
            print(f"  {freq}Hz: {count}개 ({count/len(y_train)*100:.2f}%)")
        
        print(f"\n테스트 클래스 분포:")
        for freq, count in zip(unique_test, counts_test):
            print(f"  {freq}Hz: {count}개 ({count/len(y_test)*100:.2f}%)")
        
        # XGBoost 모델 학습
        print(f"\nXGBoost 모델 학습 중...")
        model = xgb.XGBClassifier(
            max_depth=5,
            learning_rate=0.1,
            n_estimators=100,
            objective='multi:softmax',
            num_class=3,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # 예측 및 평가
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        # 클래스별 성능
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred_test, average=None, zero_division=0
        )
        
        print(f"\n훈련 정확도: {train_acc:.4f}")
        print(f"테스트 정확도: {test_acc:.4f}")
        
        print(f"\n클래스별 성능:")
        for i, (freq, p, r, f, s) in enumerate(zip(unique_test, precision, recall, f1, support)):
            print(f"  {freq}Hz - Precision: {p:.4f}, Recall: {r:.4f}, F1: {f:.4f}, Support: {s}")
        
        # 모델 저장
        model_filename = Path(MODEL_DIR) / f"xgb_snr_{snr_level}db.pkl"
        joblib.dump(model, model_filename)
        print(f"\n✓ 모델 저장: {model_filename}")
        
        # 결과 기록
        result = {
            'snr_db': snr_level,
            'train_samples': len(y_train),
            'test_samples': len(y_test),
            'train_accuracy': train_acc,
            'test_accuracy': test_acc
        }
        
        for freq, p, r, f, s in zip(unique_test, precision, recall, f1, support):
            result[f'precision_{int(freq)}Hz'] = p
            result[f'recall_{int(freq)}Hz'] = r
            result[f'f1_{int(freq)}Hz'] = f
            result[f'support_{int(freq)}Hz'] = int(s)
        
        all_results.append(result)
    
    # 전체 결과 저장
    results_df = pd.DataFrame(all_results)
    results_csv = Path(OUTPUT_DIR) / "xgb_snr_experiment_results.csv"
    results_df.to_csv(results_csv, index=False)
    
    print(f"\n{'='*70}")
    print("모든 SNR 실험 완료!")
    print(f"{'='*70}")
    print(f"\n결과 저장: {results_csv}")
    print("\n전체 결과:")
    print(results_df[['snr_db', 'train_samples', 'test_samples', 'train_accuracy', 'test_accuracy']])

if __name__ == "__main__":
    train_xgboost_model()
