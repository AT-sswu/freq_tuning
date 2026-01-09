import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
import joblib
from pathlib import Path
from data_preprocessing import create_classification_dataset_fixed
from config import DATA_DIR, OUTPUT_DIR, MODEL_DIR, XGBOOST_CONFIG


def train_xgboost_model():
    """XGBoost 모델 학습 및 평가 파이프라인"""
    print("=" * 70)
    print("데이터셋 생성 중...")
    print("=" * 70)
    
    # 출력 디렉토리 미리 생성
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    
    # 데이터셋 생성
    X_train, y_train, X_test, y_test, freq_train, freq_test, snr_train, snr_test = \
        create_classification_dataset_fixed(data_dir=DATA_DIR, output_dir=OUTPUT_DIR)
    
    print(f"\n✓ 훈련 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")

    # XGBoost 모델 설정
    print(f"\n{'=' * 70}")
    print(f"XGBoost 모델 파라미터")
    print(f"{'=' * 70}")
    for key, val in XGBOOST_CONFIG.items():
        print(f"  {key}: {val}")
    print()
    
    # 클래스 가중치 자동 계산 (불균형 처리)
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', 
                                        classes=np.unique(y_train), 
                                        y=y_train)
    
    scale_pos_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(f"클래스 가중치 (불균형 보정): {scale_pos_weight_dict}\n")
    
    # XGBoost 모델 생성 및 학습
    print("모델 학습 중...")
    xgb_model = xgb.XGBClassifier(**XGBOOST_CONFIG)
    xgb_model.fit(X_train, y_train, verbose=False)
    
    # 모델 평가
    train_accuracy = xgb_model.score(X_train, y_train)
    print(f"\n✓ 훈련 정확도: {train_accuracy:.4f}")
    
    # 테스트 평가
    y_pred = xgb_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"✓ 테스트 정확도: {test_accuracy:.4f}\n")
    
    print("=" * 70)
    print("분류 보고서")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=['40Hz', '50Hz', '60Hz']))

    # 클래스별 상세 지표 계산
    print("=" * 70)
    print("클래스별 상세 지표 (따로 계산)")
    print("=" * 70)
    cm = confusion_matrix(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred)
    
    class_names = ['40Hz', '50Hz', '60Hz']
    print(f"\n{'클래스':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    
    for i, class_name in enumerate(class_names):
        # 클래스별 정확도 (그 클래스에 대해 맞게 분류한 비율)
        class_accuracy = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        print(f"{class_name:<10} {class_accuracy:<12.4f} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f} {support[i]:<10}")
    
    # 가중 평균
    weighted_accuracy = np.average([cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0 for i in range(3)], 
                                    weights=support)
    weighted_precision = np.average(precision, weights=support)
    weighted_recall = np.average(recall, weights=support)
    weighted_f1 = np.average(f1, weights=support)
    
    print("-" * 80)
    print(f"{'Weighted':<10} {weighted_accuracy:<12.4f} {weighted_precision:<12.4f} {weighted_recall:<12.4f} {weighted_f1:<12.4f}")
    
    # 혼동 행렬
    print("\n" + "=" * 70)
    print("혼동 행렬 (Confusion Matrix)")
    print("=" * 70)
    print(f"\n{'':>15} {'40Hz':>10} {'50Hz':>10} {'60Hz':>10}")
    print("-" * 50)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:>15} {cm[i, 0]:>10} {cm[i, 1]:>10} {cm[i, 2]:>10}")
    
    # 결과 저장
    model_path = MODEL_DIR / "xgb_model.pkl"
    joblib.dump(xgb_model, model_path)
    print(f"\n✓ 모델 저장됨: {model_path}")
    
    # 결과 로그 저장
    log_path = Path(__file__).parent / "xgb_training.log"
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("XGBoost 모델 학습 결과\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"훈련 데이터: {X_train.shape}\n")
        f.write(f"테스트 데이터: {X_test.shape}\n")
        f.write(f"훈련 정확도: {train_accuracy:.4f}\n")
        f.write(f"테스트 정확도: {test_accuracy:.4f}\n\n")
        f.write("=" * 70 + "\n")
        f.write("분류 보고서\n")
        f.write("=" * 70 + "\n")
        f.write(classification_report(y_test, y_pred, target_names=['40Hz', '50Hz', '60Hz']))
        f.write("\n" + "=" * 70 + "\n")
        f.write("클래스별 상세 지표\n")
        f.write("=" * 70 + "\n")
        f.write(f"{'클래스':<10} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 60 + "\n")
        for i, class_name in enumerate(class_names):
            class_accuracy = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
            f.write(f"{class_name:<10} {class_accuracy:<12.4f} {precision[i]:<12.4f} {recall[i]:<12.4f} {f1[i]:<12.4f}\n")
    
    print(f"✓ 로그 저장됨: {log_path}")
    
    print(f"\n{'=' * 70}")
    print("✓ XGBoost 모델 학습 완료!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    train_xgboost_model()
