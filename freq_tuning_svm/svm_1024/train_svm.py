import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from pathlib import Path
from data_preprocessing import create_classification_dataset_single_snr
from config import DATA_DIR, OUTPUT_DIR, MODEL_DIR

# SNR 실험 레벨
SNR_LEVELS = [-10, -5, 0, 5, 10]

def train_svm_model():
    """SVM 모델 SNR별 학습 실험 (훈련 SNR = 테스트 SNR)"""
    print("=" * 70)
    print("SVM SNR별 학습 실험 (훈련 SNR = 테스트 SNR)")
    print("=" * 70)
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # 각 SNR 레벨로 독립적으로 데이터셋 생성 및 훈련
    for snr_level in SNR_LEVELS:
        print(f"\n{'='*70}")
        print(f"SNR {snr_level}dB로 훈련 및 테스트")
        print(f"{'='*70}")
        
        # 해당 SNR로 데이터셋 생성
        X_train, y_train, X_test, y_test = create_classification_dataset_single_snr(
            DATA_DIR, OUTPUT_DIR, snr_level
        )
        
        print(f"\n훈련 데이터: {X_train.shape[0]} 샘플")
        print(f"테스트 데이터: {X_test.shape[0]} 샘플")
        
        # 훈련 데이터 클래스 분포
        train_counts = np.bincount(y_train, minlength=3)
        test_counts = np.bincount(y_test, minlength=3)
        print(f"훈련 분포: 40Hz={train_counts[0]}, 50Hz={train_counts[1]}, 60Hz={train_counts[2]}")
        print(f"테스트 분포: 40Hz={test_counts[0]}, 50Hz={test_counts[1]}, 60Hz={test_counts[2]}")
        
        # SVM 모델 학습
        svm = SVC(C=10, gamma='scale', class_weight='balanced', kernel='rbf', random_state=42)
        print("\n모델 학습 중...")
        svm.fit(X_train, y_train)
        
        # 평가
        train_accuracy = svm.score(X_train, y_train)
        y_pred = svm.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n✓ 훈련 정확도: {train_accuracy:.4f}")
        print(f"✓ 테스트 정확도: {test_accuracy:.4f}")
        
        # 클래스별 지표
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )
        
        all_results.append({
            'model': 'SVM',
            'snr_db': snr_level,
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'train_acc': train_accuracy,
            'test_acc': test_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        print("\n분류 보고서:")
        print(classification_report(y_test, y_pred, target_names=['40Hz', '50Hz', '60Hz'], zero_division=0))
    
    # 결과 저장
    print(f"\n{'='*70}")
    print("SNR별 실험 결과 요약")
    print(f"{'='*70}")
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))
    
    results_csv_path = Path(OUTPUT_DIR) / "svm_snr_experiment_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    print(f"\n✓ 결과 저장: {results_csv_path}")

if __name__ == "__main__":
    train_svm_model()
