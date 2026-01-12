import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support, confusion_matrix
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
    
    # precision, recall, f1-score 계산
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average=None, labels=[0, 1, 2])
    class_names = ['40Hz', '50Hz', '60Hz']
    
    # 혼동 행렬 계산
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    # 로그 출력
    print(f"훈련 정확도: {train_accuracy:.4f}")
    print(f"테스트 정확도: {test_accuracy:.4f}")
    print(classification_report(y_test, y_pred, target_names=class_names))
    print("클래스별 정확도:", {k: f"{v:.4f}" for k, v in per_class_acc.items()})
    print("\n클래스별 Precision, Recall, F1-Score:")
    for i, cls_name in enumerate(class_names):
        print(f"  {cls_name}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}")
    if focus_acc is not None:
        print(f"\n→ 60Hz(클래스 2) 정확도: {focus_acc:.4f}")
        print(f"→ 60Hz(클래스 2) Precision: {precision[2]:.4f}, Recall: {recall[2]:.4f}, F1: {f1[2]:.4f}")

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
    
    # 예측 주파수 저장 (클래스 인덱스 -> 주파수)
    freq_map = {0: 40, 1: 50, 2: 60}
    y_pred_freq = np.array([freq_map[int(p)] for p in y_pred])
    y_test_freq = np.array([freq_map[int(t)] for t in y_test])
    
    predictions_path = Path(OUTPUT_DIR) / f"predictions_{name}.npz"
    np.savez(predictions_path,
             y_pred=y_pred,
             y_test=y_test,
             y_pred_freq=y_pred_freq,
             y_test_freq=y_test_freq,
             freq_test=freq_test,
             snr_test=snr_test)

    return {
        'name': name,
        'allowed_snrs': allowed_snrs,
        'train_shape': X_train.shape,
        'test_shape': X_test.shape,
        'train_acc': train_accuracy,
        'test_acc': test_accuracy,
        'per_class_acc': per_class_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'confusion_matrix': cm,
        'focus_acc': focus_acc,
        'y_pred_freq': y_pred_freq,
    }


def run_all_experiments():
    summaries = []

    # 실험 1: 주파수 기준 (SNR=0dB 고정)
    summaries.append(_run_single_experiment(name="freq_snr0", allowed_snrs=[0], focus_class=2))

    # 실험 2: SNR 기준 (60Hz 성능 관찰, focus_class=2) - 0dB 제외
    for snr in AUGMENTATION_SNR_DB:
        if snr != 0:  # 0dB는 이미 freq_snr0에서 실행했으므로 제외
            summaries.append(_run_single_experiment(name=f"snr_{snr}dB", allowed_snrs=[snr], focus_class=2))

    # 상세 요약 저장
    summary_path = Path(OUTPUT_DIR) / "experiment_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("k-NN 모델 실험 종합 요약\n")
        f.write("=" * 100 + "\n\n")
        
        # 모델 설정 정보
        f.write("[모델 설정]\n")
        f.write(f"  알고리즘: k-Nearest Neighbors (k-NN)\n")
        f.write(f"  하이퍼파라미터: {KNN_CONFIG}\n")
        f.write(f"  전처리: StandardScaler 적용\n\n")
        
        # 실험별 상세 정보
        for idx, s in enumerate(summaries, 1):
            f.write("=" * 100 + "\n")
            f.write(f"실험 {idx}: {s['name']}\n")
            f.write("=" * 100 + "\n")
            f.write(f"허용 SNR: {s['allowed_snrs']}\n")
            f.write(f"훈련 데이터 shape: {s['train_shape']}\n")
            f.write(f"테스트 데이터 shape: {s['test_shape']}\n")
            f.write(f"훈련 정확도: {s['train_acc']:.4f}\n")
            f.write(f"테스트 정확도: {s['test_acc']:.4f}\n\n")
            
            # 클래스별 상세 지표
            f.write("[클래스별 성능 지표]\n")
            class_names = ['40Hz', '50Hz', '60Hz']
            f.write(f"{'클래스':<10} {'정확도':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}\n")
            f.write("-" * 80 + "\n")
            for i, cls_name in enumerate(class_names):
                f.write(f"{cls_name:<10} {s['per_class_acc'][i]:<12.4f} {s['precision'][i]:<12.4f} "
                       f"{s['recall'][i]:<12.4f} {s['f1'][i]:<12.4f} {s['support'][i]:<10}\n")
            f.write("\n")
            
            # 혼동 행렬
            f.write("[혼동 행렬 (Confusion Matrix)]\n")
            f.write(f"{'':>12} {'예측 40Hz':>12} {'예측 50Hz':>12} {'예측 60Hz':>12}\n")
            f.write("-" * 60 + "\n")
            for i, cls_name in enumerate(class_names):
                f.write(f"실제 {cls_name:>6}")
                for j in range(3):
                    f.write(f"{s['confusion_matrix'][i][j]:>12}")
                f.write("\n")
            f.write("\n\n")
        
        # 요약 1: SNR = 0 dB일 때, 40/50/60 Hz 각각의 성능
        f.write("=" * 100 + "\n")
        f.write("요약 1: SNR = 0 dB에서 주파수별 성능 비교\n")
        f.write("=" * 100 + "\n")
        snr0_result = [s for s in summaries if 'freq_snr0' in s['name']][0]
        f.write(f"전체 테스트 정확도: {snr0_result['test_acc']:.4f}\n\n")
        f.write(f"{'주파수':<10} {'정확도':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 60 + "\n")
        class_names = ['40Hz', '50Hz', '60Hz']
        for i, cls_name in enumerate(class_names):
            f.write(f"{cls_name:<10} {snr0_result['per_class_acc'][i]:<12.4f} {snr0_result['precision'][i]:<12.4f} "
                   f"{snr0_result['recall'][i]:<12.4f} {snr0_result['f1'][i]:<12.4f}\n")
        f.write("\n\n")
        
        # 요약 2: 60 Hz일 때, SNR별 성능
        f.write("=" * 100 + "\n")
        f.write("요약 2: 60 Hz에서 SNR별 성능 비교\n")
        f.write("=" * 100 + "\n")
        f.write(f"{'SNR (dB)':<12} {'정확도':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-" * 60 + "\n")
        
        # SNR별로 정렬하여 출력 (freq_snr0를 0dB로 포함)
        snr_results = []
        for s in summaries:
            if s['name'] == 'freq_snr0':
                snr_results.append((0, s))  # freq_snr0를 0dB로 처리
            elif 'snr_' in s['name']:
                snr_val = s['allowed_snrs'][0]
                snr_results.append((snr_val, s))
        
        # SNR 값으로 정렬
        snr_results.sort(key=lambda x: x[0])
        
        for snr_val, s in snr_results:
            f.write(f"{snr_val:<12} {s['per_class_acc'][2]:<12.4f} {s['precision'][2]:<12.4f} "
                   f"{s['recall'][2]:<12.4f} {s['f1'][2]:<12.4f}\n")
        f.write("\n")
        
    print(f"✓ 상세 요약 저장: {summary_path}")


if __name__ == "__main__":
    run_all_experiments()
