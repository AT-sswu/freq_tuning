import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path
from data_preprocessing import create_classification_dataset_fixed
from config import DATA_DIR, OUTPUT_DIR, MODEL_DIR, RF_CONFIG
from visualization import calculate_snr_performance, plot_snr_performance, plot_confusion_matrix, plot_class_performance, plot_combined_performance


def train_rf_model():
    """RandomForest 모델 학습 파이프라인"""
    print("=" * 70)
    print("데이터셋 생성 중...")
    print("=" * 70)
    
    # 출력 디렉토리 미리 생성
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    
    # 데이터셋 생성 (8개 반환값)
    X_train, y_train, X_test, y_test, freq_train, freq_test, snr_train, snr_test = \
        create_classification_dataset_fixed(data_dir=DATA_DIR, output_dir=OUTPUT_DIR)
    
    print(f"\n✓ 훈련 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")

    # Raw 데이터 사용
    X_train_scaled = X_train
    X_test_scaled = X_test

    # RandomForest 모델
    print(f"\n{'=' * 70}")
    print(f"RandomForest 모델 학습")
    print(f"{'=' * 70}\n")
    
    rf = RandomForestClassifier(**RF_CONFIG, class_weight='balanced_subsample')
    print("모델 학습 중...")
    rf.fit(X_train_scaled, y_train)

    # 모델 평가
    train_accuracy = rf.score(X_train_scaled, y_train)
    print(f"\n✓ 훈련 정확도: {train_accuracy:.4f}")

    # 테스트 평가
    y_pred = rf.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"✓ 테스트 정확도: {test_accuracy:.4f}\n")
    
    print("=" * 70)
    print("분류 보고서")
    print("=" * 70)
    print(classification_report(y_test, y_pred, target_names=['40Hz', '50Hz', '60Hz']))

    # SNR별 성능 계산
    print("=" * 70)
    print("SNR별 성능 분석")
    print("=" * 70)
    snr_performance = calculate_snr_performance(y_test, y_pred, snr_test)
    if snr_performance:
        for snr, acc in sorted(snr_performance.items()):
            print(f"SNR {snr:>3}dB: {acc:.4f}")
        plot_snr_performance(snr_performance, OUTPUT_DIR)
    else:
        print("SNR 메타데이터 없음 (테스트 셋에 증강 데이터 미포함)")
    
    # 클래스별 성능 시각화
    print("\n" + "=" * 70)
    print("시각화 생성 중...")
    print("=" * 70)
    plot_confusion_matrix(y_test, y_pred, OUTPUT_DIR)
    plot_class_performance(y_test, y_pred, OUTPUT_DIR)
    plot_combined_performance(y_test, y_pred, OUTPUT_DIR)

    # 모델 저장
    model_path = Path(MODEL_DIR) / "rf_model.pkl"
    joblib.dump(rf, model_path)
    print(f"\n✓ 모델 저장: {model_path}")

    # 특징 중요도 저장
    feature_importance_path = Path(OUTPUT_DIR) / "feature_importance.txt"
    with open(feature_importance_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RandomForest 특징 중요도\n")
        f.write("=" * 70 + "\n\n")
        for i, importance in enumerate(rf.feature_importances_):
            f.write(f"특징 {i}: {importance:.4f}\n")
    print(f"✓ 특징 중요도 저장: {feature_importance_path}")

    # 결과 저장
    results_path = Path(OUTPUT_DIR) / "training_results.txt"
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RandomForest 모델 학습 결과\n")
        f.write("=" * 70 + "\n\n")
        f.write("[모델 파라미터]\n")
        for key, value in RF_CONFIG.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\n[성능 지표]\n")
        f.write(f"훈련 정확도: {train_accuracy:.4f}\n")
        f.write(f"테스트 정확도: {test_accuracy:.4f}\n\n")
        f.write("[분류 보고서]\n")
        f.write(classification_report(y_test, y_pred, target_names=['40Hz', '50Hz', '60Hz']))
        f.write("\n" + "=" * 70 + "\n")
    print(f"✓ 결과 저장: {results_path}\n")


if __name__ == "__main__":
    train_rf_model()
