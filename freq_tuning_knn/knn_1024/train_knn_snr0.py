import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
from data_preprocessing_snr0 import create_classification_dataset_fixed
from config import (
    DATA_DIR,
    OUTPUT_DIR,
    MODEL_DIR,
    KNN_CONFIG,
    AUGMENTATION_SNR_DB,
)


def _per_class_accuracy(y_true, y_pred):
    """클래스별 정확도(=해당 클래스의 리콜)를 반환"""
    acc = {}
    classes = np.unique(y_true)
    for cls in classes:
        mask = y_true == cls
        if np.sum(mask) == 0:
            acc[cls] = 0.0
        else:
            acc[cls] = np.mean(y_pred[mask] == y_true[mask])
    return acc


def _run_single_experiment(name, allowed_snrs, focus_class=None):
    print("=" * 80)
    print(f"[k-NN] 실험: {name} | 허용 SNR: {allowed_snrs}")
    print("=" * 80)

    # 데이터셋 생성 (SNR 필터 적용)
    X_train, y_train, X_test, y_test, freq_train, freq_test, snr_train, snr_test = \
        create_classification_dataset_fixed(
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            allowed_snrs=allowed_snrs,
        )

    print(f"훈련 데이터: {X_train.shape}, 테스트 데이터: {X_test.shape}")

    # 스케일 표준화 + k-NN 파이프라인
    knn_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(**KNN_CONFIG))
    ])
    knn_pipeline.fit(X_train, y_train)

    y_train_pred = knn_pipeline.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    y_pred = knn_pipeline.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    per_class_acc = _per_class_accuracy(y_test, y_pred)
    focus_acc = per_class_acc.get(focus_class, None) if focus_class is not None else None

    # 로그 출력
    print(f"훈련 정확도: {train_accuracy:.4f}")
    print(f"테스트 정확도: {test_accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=['40Hz', '50Hz', '60Hz']))
    print("클래스별 정확도:", {k: f"{v:.4f}" for k, v in per_class_acc.items()})
    if focus_acc is not None:
        print(f"→ 60Hz(클래스 2) 정확도: {focus_acc:.4f}")

    # 결과 저장
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    result_path = Path(OUTPUT_DIR) / f"training_results_{name}.txt"
    model_path = Path(MODEL_DIR) / f"knn_model_{name}.pkl"
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(f"실험명: {name}\n")
        f.write(f"허용 SNR: {allowed_snrs}\n")
        f.write(f"훈련 정확도: {train_accuracy:.4f}\n")
        f.write(f"테스트 정확도: {test_accuracy:.4f}\n")
        f.write("클래스별 정확도:\n")
        for cls_id, acc in per_class_acc.items():
            f.write(f"  클래스 {cls_id} ( {['40Hz','50Hz','60Hz'][cls_id]} ): {acc:.4f}\n")
        f.write("\n[분류 보고서]\n")
        f.write(classification_report(y_test, y_pred, target_names=['40Hz', '50Hz', '60Hz']))
    joblib.dump(knn_pipeline, model_path)

    return {
        'name': name,
        'allowed_snrs': allowed_snrs,
        'train_acc': train_accuracy,
        'test_acc': test_accuracy,
        'per_class': per_class_acc,
        'focus_acc': focus_acc,
    }


def run_all_experiments():
    summaries = []

    # 실험 1: 주파수 기준 (SNR=0dB 고정)
    summaries.append(_run_single_experiment(name="freq_snr0", allowed_snrs=[0]))

    # 실험 2: SNR 기준 (60Hz 성능 관찰, focus_class=2)
    for snr in AUGMENTATION_SNR_DB:
        summaries.append(_run_single_experiment(name=f"snr_{snr}dB", allowed_snrs=[snr], focus_class=2))

    # 요약 저장
    summary_path = Path(OUTPUT_DIR) / "experiment_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("실험 요약\n")
        f.write("=" * 40 + "\n")
        for s in summaries:
            f.write(f"{s['name']} | SNR {s['allowed_snrs']} | 전체 {s['test_acc']:.4f}")
            if s['focus_acc'] is not None:
                f.write(f" | 60Hz {s['focus_acc']:.4f}")
            f.write("\n")
    print(f"요약 저장: {summary_path}")


if __name__ == "__main__":
    run_all_experiments()
