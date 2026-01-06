import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from config import OUTPUT_DIR, MODEL_DIR, RESONANCE_FREQS


def load_classification_data_fixed(data_path):
    """수정된 데이터 로드 (train/test 분리됨)"""
    data = np.load(data_path)
    X_train = data['X_train']
    y_train = data['y_train']
    freq_train = data['freq_train']
    X_test = data['X_test']
    y_test = data['y_test']
    freq_test = data['freq_test']
    return X_train, y_train, freq_train, X_test, y_test, freq_test


def balance_dataset(X, y, strategy='auto', random_state=42):
    """클래스 균형화"""
    print("\n[클래스 균형화 전]")
    for i, freq in enumerate(RESONANCE_FREQS):
        count = np.sum(y == i)
        print(f"  클래스 {i} ({freq}Hz): {count}개 ({count / len(y) * 100:.1f}%)")

    if strategy == 'auto':
        class_counts = [np.sum(y == i) for i in range(len(RESONANCE_FREQS))]
        target_count = int(np.median(class_counts))
        print(f"\n전략: SMOTE + UnderSampling (목표: {target_count}개)")

        smote = SMOTE(random_state=random_state, k_neighbors=min(3, min(class_counts) - 1))
        X_temp, y_temp = smote.fit_resample(X, y)

        sampling_strategy = {i: target_count for i in range(len(RESONANCE_FREQS))}
        under_sampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
        X_balanced, y_balanced = under_sampler.fit_resample(X_temp, y_temp)

    elif strategy == 'smote_only':
        print(f"\n전략: SMOTE only")
        class_counts = [np.sum(y == i) for i in range(len(RESONANCE_FREQS))]
        smote = SMOTE(random_state=random_state, k_neighbors=min(3, min(class_counts) - 1))
        X_balanced, y_balanced = smote.fit_resample(X, y)

    elif strategy == 'none':
        print(f"\n전략: 균형화 없음")
        X_balanced, y_balanced = X, y

    print("\n[클래스 균형화 후]")
    for i, freq in enumerate(RESONANCE_FREQS):
        count = np.sum(y_balanced == i)
        print(f"  클래스 {i} ({freq}Hz): {count}개 ({count / len(y_balanced) * 100:.1f}%)")

    return X_balanced, y_balanced


def analyze_frequency_mapping_loss(y_true, y_pred, freq_true, save_dir):
    """✅ 각 클래스별 경계 영역 매핑 손실 분석"""

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    actual_freqs = np.array([RESONANCE_FREQS[y] for y in y_true])
    predicted_freqs = np.array([RESONANCE_FREQS[y] for y in y_pred])

    print(f"\n{'=' * 70}")
    print(f"{'클래스별 경계 영역 매핑 손실 분석':^70}")
    print(f"{'=' * 70}\n")

    for i, target_freq in enumerate(RESONANCE_FREQS):
        ax = axes[i]

        # 해당 클래스 데이터 추출
        class_mask = y_true == i
        class_input_freqs = freq_true[class_mask]
        class_actual = actual_freqs[class_mask]
        class_predicted = predicted_freqs[class_mask]
        class_errors = class_predicted - class_actual

        if len(class_input_freqs) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=16)
            ax.set_title(f'{target_freq}Hz Class - No Data', fontsize=14, fontweight='bold')
            continue

        # 경계 영역별 분석 (±5Hz 범위)
        boundary_ranges = [
            (target_freq - 5, target_freq - 2.5, f'{target_freq - 5}~{target_freq - 2.5}Hz'),
            (target_freq - 2.5, target_freq, f'{target_freq - 2.5}~{target_freq}Hz'),
            (target_freq, target_freq + 2.5, f'{target_freq}~{target_freq + 2.5}Hz'),
            (target_freq + 2.5, target_freq + 5, f'{target_freq + 2.5}~{target_freq + 5}Hz'),
        ]

        accuracies = []
        labels = []
        colors_map = []

        print(f"[{target_freq}Hz 클래스]")

        for low, high, label in boundary_ranges:
            range_mask = (class_input_freqs >= low) & (class_input_freqs < high)

            if range_mask.sum() > 0:
                range_errors = class_errors[range_mask]
                accuracy = (range_errors == 0).mean() * 100
                accuracies.append(accuracy)
                labels.append(label)

                # 색상 결정
                if accuracy >= 90:
                    colors_map.append('green')
                elif accuracy >= 70:
                    colors_map.append('orange')
                else:
                    colors_map.append('red')

                print(f"  {label:20s}: {accuracy:6.2f}% ({(range_errors == 0).sum():3d}/{range_mask.sum():3d})")
            else:
                accuracies.append(0)
                labels.append(label)
                colors_map.append('gray')
                print(f"  {label:20s}: No data")

        # 전체 정확도
        overall_acc = (class_errors == 0).mean() * 100
        print(f"  {'전체':20s}: {overall_acc:6.2f}% ({(class_errors == 0).sum():3d}/{len(class_errors):3d})\n")

        # 막대 그래프
        bars = ax.bar(range(len(labels)), accuracies, color=colors_map,
                      alpha=0.7, edgecolor='black', linewidth=1.5)

        # 정확도 표시
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{acc:.1f}%',
                        ha='center', va='bottom',
                        fontsize=11, fontweight='bold')

        # 90% 기준선
        ax.axhline(y=90, color='blue', linestyle='--', linewidth=2,
                   label='90% Target', alpha=0.7)

        ax.set_xlabel('Frequency Range (Hz)', fontsize=12)
        ax.set_ylabel('Mapping Accuracy (%)', fontsize=12)
        ax.set_title(f'{target_freq}Hz Class - Boundary Mapping Loss',
                     fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # 저장
    plot_path = save_path / "class_boundary_mapping_loss.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ 클래스별 매핑 손실 그래프 저장: {plot_path}\n")
    plt.close()


def train_svm_model_fixed(X_train, y_train, freq_train, X_test, y_test, freq_test,
                          balance_strategy='auto', random_state=42):
    """✅ 이미 분리된 train/test 데이터로 학습 + Loss 분석"""
    print(f"\n{'=' * 70}")
    print(f"{'SVM 모델 학습 (데이터 누수 방지)':^70}")
    print(f"{'=' * 70}")
    print(f"Train 데이터: X={X_train.shape}, y={y_train.shape}")
    print(f"Test 데이터: X={X_test.shape}, y={y_test.shape}")

    # 클래스 균형화 (Train만)
    X_train_balanced, y_train_balanced = balance_dataset(
        X_train, y_train,
        strategy=balance_strategy,
        random_state=random_state
    )

    print(f"\n균형화 후 Train 데이터: {len(X_train_balanced)}개")

    # 정규화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)

    # 하이퍼파라미터 튜닝
    print(f"\n{'[하이퍼파라미터 탐색]':^70}")
    print(f"{'-' * 70}")

    best_accuracy = 0
    best_params = None
    best_model = None

    param_grid = [
        {'kernel': 'rbf', 'C': 1, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 100, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 10, 'gamma': 0.1},
        {'kernel': 'rbf', 'C': 10, 'gamma': 0.01},
    ]

    for params in param_grid:
        model = SVC(**params, class_weight='balanced', random_state=random_state)
        model.fit(X_train_scaled, y_train_balanced)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"  {params} → 정확도: {accuracy * 100:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            best_model = model

    print(f"\n✓ 최적 파라미터: {best_params}")
    print(f"✓ 최고 정확도: {best_accuracy * 100:.2f}%")

    # 최종 평가
    y_pred = best_model.predict(X_test_scaled)

    print(f"\n{'=' * 70}")
    print(f"{'최종 모델 성능':^70}")
    print(f"{'=' * 70}")
    print(f"정확도: {best_accuracy * 100:.2f}%\n")

    print("분류 리포트:")
    target_names = [f"{freq}Hz" for freq in RESONANCE_FREQS]
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("\n혼동 행렬:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # 클래스별 정확도
    print(f"\n{'[클래스별 정확도]':^70}")
    print(f"{'-' * 70}")
    for i, freq in enumerate(RESONANCE_FREQS):
        mask = y_test == i
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == y_test[mask]).mean() * 100
            total = mask.sum()
            correct = (y_pred[mask] == y_test[mask]).sum()
            status = "✓" if class_acc >= 80 else "✗"
            print(f"  {status} {freq}Hz: {class_acc:.2f}% ({correct}/{total})")

    return best_model, scaler, best_accuracy, y_pred


def analyze_prediction_errors(y_test, y_pred, freq_test, save_dir):
    """✅ 예측 오차 분석 및 시각화"""

    # 실제/예측 주파수 계산
    actual_freqs = np.array([RESONANCE_FREQS[y] for y in y_test])
    predicted_freqs = np.array([RESONANCE_FREQS[y] for y in y_pred])
    errors = predicted_freqs - actual_freqs

    print(f"\n{'=' * 70}")
    print(f"{'예측 오차 분석':^70}")
    print(f"{'=' * 70}")

    # 클래스별 오차 통계
    for i, freq in enumerate(RESONANCE_FREQS):
        mask = y_test == i
        if mask.sum() > 0:
            class_errors = errors[mask]
            mean_error = np.mean(class_errors)
            std_error = np.std(class_errors)
            mae = np.mean(np.abs(class_errors))

            print(f"\n{freq}Hz 클래스:")
            print(f"  평균 오차: {mean_error:+.2f} Hz")
            print(f"  표준편차: {std_error:.2f} Hz")
            print(f"  절대 오차: {mae:.2f} Hz")
            print(f"  정확도: {(class_errors == 0).mean() * 100:.1f}%")

    # 전체 통계
    print(f"\n전체:")
    print(f"  평균 절대 오차(MAE): {np.mean(np.abs(errors)):.2f} Hz")
    print(f"  RMSE: {np.sqrt(np.mean(errors ** 2)):.2f} Hz")
    print(f"  정확도: {(errors == 0).mean() * 100:.2f}%")

    # 시각화
    create_error_plots(y_test, y_pred, freq_test, errors, save_dir)

    return errors


def create_error_plots(y_test, y_pred, freq_test, errors, save_dir):
    """✅ 오차 시각화 (4개 그래프)"""

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 12))

    # 실제/예측 주파수
    actual_freqs = np.array([RESONANCE_FREQS[y] for y in y_test])
    predicted_freqs = np.array([RESONANCE_FREQS[y] for y in y_pred])

    # 1. 클래스별 평균 오차
    ax1 = plt.subplot(2, 2, 1)
    class_errors = []
    class_labels = []

    for i, freq in enumerate(RESONANCE_FREQS):
        mask = y_test == i
        if mask.sum() > 0:
            mean_error = np.mean(errors[mask])
            class_errors.append(mean_error)
            class_labels.append(f'{freq}Hz')

    colors = ['green' if abs(e) < 5 else 'orange' if abs(e) < 10 else 'red'
              for e in class_errors]
    bars = ax1.bar(class_labels, class_errors, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel('Frequency Class (Hz)', fontsize=12)
    ax1.set_ylabel('Average Prediction Error (Hz)', fontsize=12)
    ax1.set_title('Average Prediction Error by Frequency Class', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # 오차 값 표시
    for bar, error in zip(bars, class_errors):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{error:+.2f}Hz',
                 ha='center', va='bottom' if height > 0 else 'top',
                 fontsize=10, fontweight='bold')

    # 2. 오차 분포 히스토그램
    ax2 = plt.subplot(2, 2, 2)

    for i, freq in enumerate(RESONANCE_FREQS):
        mask = y_test == i
        if mask.sum() > 0:
            class_errors_data = errors[mask]
            ax2.hist(class_errors_data, bins=15, alpha=0.6, label=f'{freq}Hz',
                     edgecolor='black', linewidth=0.5)

    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    ax2.set_xlabel('Prediction Error (Hz)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Error Distribution by Frequency Class', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # 3. 실제 vs 예측 주파수 산점도
    ax3 = plt.subplot(2, 2, 3)

    for i, freq in enumerate(RESONANCE_FREQS):
        mask = y_test == i
        if mask.sum() > 0:
            ax3.scatter(actual_freqs[mask], predicted_freqs[mask],
                        alpha=0.6, s=50, label=f'{freq}Hz',
                        edgecolors='black', linewidths=0.5)

    # 완벽한 예측 라인
    ax3.plot([25, 65], [25, 65], 'r--', linewidth=2, label='Perfect Prediction')
    ax3.set_xlabel('Actual Frequency (Hz)', fontsize=12)
    ax3.set_ylabel('Predicted Frequency (Hz)', fontsize=12)
    ax3.set_title('Actual vs Predicted Frequency', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(25, 65)
    ax3.set_ylim(25, 65)

    # 4. 혼동 행렬 (정규화)
    ax4 = plt.subplot(2, 2, 4)

    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    im = ax4.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(RESONANCE_FREQS))
    ax4.set_xticks(tick_marks)
    ax4.set_yticks(tick_marks)
    ax4.set_xticklabels([f'{f}Hz' for f in RESONANCE_FREQS])
    ax4.set_yticklabels([f'{f}Hz' for f in RESONANCE_FREQS])

    # 셀에 값 표시
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax4.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j] * 100:.1f}%)',
                     ha="center", va="center",
                     color="white" if cm_normalized[i, j] > thresh else "black",
                     fontsize=9)

    ax4.set_xlabel('Predicted Frequency', fontsize=12)
    ax4.set_ylabel('Actual Frequency', fontsize=12)
    ax4.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')

    plt.tight_layout()

    # 저장
    plot_path = save_path / "prediction_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 분석 그래프 저장: {plot_path}")
    plt.close()


def save_model(model, scaler, save_dir):
    """모델 저장"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model_path = save_path / "svm_model.pkl"
    scaler_path = save_path / "svm_scaler.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"\n✓ 모델 저장: {model_path}")
    print(f"✓ 스케일러 저장: {scaler_path}")


if __name__ == "__main__":
    # 데이터 로드
    data_path = f"{OUTPUT_DIR}/classification_data.npz"
    X_train, y_train, freq_train, X_test, y_test, freq_test = load_classification_data_fixed(data_path)

    # 여러 전략 시도
    strategies = ['auto', 'smote_only', 'none']
    results = {}

    print(f"\n{'=' * 70}")
    print(f"{'여러 균형화 전략 비교':^70}")
    print(f"{'=' * 70}")

    best_strategy = None
    best_model_overall = None
    best_scaler_overall = None
    best_y_pred = None

    for strategy in strategies:
        print(f"\n\n{'#' * 70}")
        print(f"{'전략: ' + strategy:^70}")
        print(f"{'#' * 70}")

        try:
            model, scaler, accuracy, y_pred = train_svm_model_fixed(
                X_train, y_train, freq_train, X_test, y_test, freq_test,
                balance_strategy=strategy,
                random_state=42
            )
            results[strategy] = accuracy

            if best_strategy is None or accuracy > results[best_strategy]:
                best_strategy = strategy
                best_model_overall = model
                best_scaler_overall = scaler
                best_y_pred = y_pred

        except Exception as e:
            print(f"✗ 오류 발생: {e}")
            import traceback

            traceback.print_exc()
            results[strategy] = 0

    # 최고 전략 선택
    print(f"\n{'=' * 70}")
    print(f"{'전략 비교 결과':^70}")
    print(f"{'=' * 70}")

    for strategy, accuracy in results.items():
        print(f"  {strategy:20s}: {accuracy * 100:.2f}%")

    print(f"\n✓ 최적 전략: {best_strategy} (정확도: {results[best_strategy] * 100:.2f}%)")

    # 최적 모델 저장
    save_model(best_model_overall, best_scaler_overall, MODEL_DIR)

    # 오차 분석 및 시각화
    errors = analyze_prediction_errors(y_test, best_y_pred, freq_test, MODEL_DIR)

    # ✅ 클래스별 경계 영역 매핑 손실 분석 추가
    analyze_frequency_mapping_loss(y_test, best_y_pred, freq_test, MODEL_DIR)

    print(f"\n{'=' * 70}")
    print(f"✓ 학습 완료! 최종 정확도: {results[best_strategy] * 100:.2f}%")
    print(f"{'=' * 70}")