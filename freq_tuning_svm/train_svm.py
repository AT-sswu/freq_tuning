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


# =========================================================
# [1] 데이터 로드 및 전처리 함수
# =========================================================

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
    """클래스 균형화 - 3가지 전략 지원"""
    print("\n[클래스 균형화 전]")
    class_counts = [np.sum(y == i) for i in range(len(RESONANCE_FREQS))]
    for i, freq in enumerate(RESONANCE_FREQS):
        count = class_counts[i]
        print(f"  클래스 {i} ({freq}Hz): {count}개 ({count / len(y) * 100:.1f}%)")

    min_count = min(class_counts)
    max_count = max(class_counts)
    median_count = int(np.median(class_counts))

    if strategy == 'auto':
        target_count = median_count
        print(f"\n전략: SMOTE + UnderSampling (목표: {target_count}개)")

        k_neighbors = min(3, min_count - 1) if min_count > 1 else 1
        smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
        X_temp, y_temp = smote.fit_resample(X, y)

        sampling_strategy = {i: target_count for i in range(len(RESONANCE_FREQS))}
        under_sampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
        X_balanced, y_balanced = under_sampler.fit_resample(X_temp, y_temp)

    elif strategy == 'smote_only':
        print(f"\n전략: SMOTE only (목표: {max_count}개)")
        k_neighbors = min(3, min_count - 1) if min_count > 1 else 1
        sampling_strategy = {i: max_count for i in range(len(RESONANCE_FREQS))}
        smote = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors
        )
        X_balanced, y_balanced = smote.fit_resample(X, y)

    elif strategy == 'downsample':
        print(f"\n전략: Downsample (목표: {min_count}개)")
        sampling_strategy = {i: min_count for i in range(len(RESONANCE_FREQS))}
        under_sampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
        X_balanced, y_balanced = under_sampler.fit_resample(X, y)

    elif strategy == 'none':
        print("\n전략: 균형화 없음")
        X_balanced, y_balanced = X, y

    else:
        raise ValueError(
            f"지원하지 않는 전략: {strategy}. "
            "'auto', 'smote_only', 'downsample', 'none' 중 선택하세요."
        )

    print("\n[클래스 균형화 후]")
    for i, freq in enumerate(RESONANCE_FREQS):
        count = np.sum(y_balanced == i)
        print(f"  클래스 {i} ({freq}Hz): {count}개 ({count / len(y_balanced) * 100:.1f}%)")

    return X_balanced, y_balanced


# =========================================================
# [2] 시각화 및 분석 함수
# =========================================================

def plot_training_loss_curve(param_grid, acc_history, save_dir):
    """하이퍼파라미터 변화에 따른 정확도 및 손실(100-Acc) 추이 그래프"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    params_labels = [f"C={p['C']}\ng={p['gamma']}" for p in param_grid]
    accuracies = [a * 100 for a in acc_history]
    losses = [100 - a for a in accuracies]

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Hyperparameter Settings (RBF Kernel)', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', color='tab:blue', fontsize=12, fontweight='bold')
    ax1.bar(params_labels, accuracies, color='tab:blue', alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 110)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Empirical Loss (100 - Accuracy %)', color='tab:red', fontsize=12, fontweight='bold')
    ax2.plot(params_labels, losses, color='tab:red', marker='o', linewidth=3, markersize=8)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('SVM Model Optimization: Accuracy vs Loss', fontsize=15, pad=20)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    for i, acc in enumerate(accuracies):
        ax1.text(i, acc + 1, f'{acc:.1f}%', ha='center', color='blue', fontweight='bold')

    fig.tight_layout()
    plt.savefig(save_path / "training_loss_curve.png", dpi=300)
    plt.close()


def analyze_frequency_mapping_loss(y_true, y_pred, freq_true, save_dir):
    """각 클래스별 경계 영역 매핑 손실 분석"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes = axes.flatten()

    actual_freqs = np.array([RESONANCE_FREQS[y] for y in y_true])
    predicted_freqs = np.array([RESONANCE_FREQS[y] for y in y_pred])

    for i, target_freq in enumerate(RESONANCE_FREQS):
        ax = axes[i]
        class_mask = y_true == i
        class_input_freqs = freq_true[class_mask]
        class_actual = actual_freqs[class_mask]
        class_predicted = predicted_freqs[class_mask]
        class_errors = class_predicted - class_actual

        if len(class_input_freqs) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=16)
            continue

        boundary_ranges = [
            (target_freq - 5, target_freq - 2.5, f'{target_freq - 5}~{target_freq - 2.5}Hz'),
            (target_freq - 2.5, target_freq, f'{target_freq - 2.5}~{target_freq}Hz'),
            (target_freq, target_freq + 2.5, f'{target_freq}~{target_freq + 2.5}Hz'),
            (target_freq + 2.5, target_freq + 5, f'{target_freq + 2.5}~{target_freq + 5}Hz'),
        ]

        accuracies = []
        labels = []
        colors_map = []

        for low, high, label in boundary_ranges:
            range_mask = (class_input_freqs >= low) & (class_input_freqs < high)
            if range_mask.sum() > 0:
                accuracy = (class_errors[range_mask] == 0).mean() * 100
                accuracies.append(accuracy)
                labels.append(label)
                colors_map.append(
                    'green' if accuracy >= 90 else
                    'orange' if accuracy >= 70 else
                    'red'
                )
            else:
                accuracies.append(0)
                labels.append(label)
                colors_map.append('gray')

        ax.bar(range(len(labels)), accuracies, color=colors_map, alpha=0.7, edgecolor='black')
        ax.set_title(f'{target_freq}Hz Boundary Loss')
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(save_path / "class_boundary_mapping_loss.png", dpi=300)
    plt.close()


def analyze_prediction_errors(y_test, y_pred, freq_test, save_dir):
    """예측 오차 분석"""
    actual_freqs = np.array([RESONANCE_FREQS[y] for y in y_test])
    predicted_freqs = np.array([RESONANCE_FREQS[y] for y in y_pred])

    fig = plt.figure(figsize=(12, 10))

    ax1 = plt.subplot(2, 2, 1)
    ax1.scatter(actual_freqs, predicted_freqs, alpha=0.5)
    ax1.plot([35, 65], [35, 65], 'r--')
    ax1.set_title("Actual vs Predicted Freq")
    ax1.set_xlabel("Actual Frequency (Hz)")
    ax1.set_ylabel("Predicted Frequency (Hz)")

    ax2 = plt.subplot(2, 2, 2)
    cm = confusion_matrix(y_test, y_pred)
    im = ax2.imshow(cm, cmap='Blues')

    # 혼동 행렬에 숫자 표시
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax2.text(j, i, str(cm[i, j]),
                     ha='center', va='center',
                     color=text_color, fontsize=12, fontweight='bold')

    ax2.set_title("Confusion Matrix")
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")

    # 클래스 레이블 추가
    ax2.set_xticks(range(len(RESONANCE_FREQS)))
    ax2.set_yticks(range(len(RESONANCE_FREQS)))
    ax2.set_xticklabels([f'{freq}Hz' for freq in RESONANCE_FREQS])
    ax2.set_yticklabels([f'{freq}Hz' for freq in RESONANCE_FREQS])

    plt.colorbar(im, ax=ax2)

    plt.tight_layout()
    plt.savefig(Path(save_dir) / "prediction_analysis.png", dpi=300)
    plt.close()


# =========================================================
# [3] SVM 학습 핵심 엔진
# =========================================================

def train_svm_model_fixed(
        X_train, y_train, freq_train,
        X_test, y_test, freq_test,
        balance_strategy='auto',
        random_state=42
):
    """학습 + 최적화 + 시각화"""
    X_train_balanced, y_train_balanced = balance_dataset(
        X_train, y_train, strategy=balance_strategy
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)

    param_grid = [
        {'kernel': 'rbf', 'C': 1, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 100, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 10, 'gamma': 0.1},
        {'kernel': 'rbf', 'C': 10, 'gamma': 0.01},
    ]

    best_accuracy = 0
    best_model = None
    acc_history = []

    print("\n[하이퍼파라미터 탐색 중...]")
    for params in param_grid:
        model = SVC(**params, class_weight='balanced', random_state=random_state)
        model.fit(X_train_scaled, y_train_balanced)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        acc_history.append(accuracy)

        print(f"  {params} -> 정확도: {accuracy * 100:.2f}%")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    plot_training_loss_curve(param_grid, acc_history, MODEL_DIR)

    return best_model, scaler, best_accuracy, best_model.predict(X_test_scaled)


# =========================================================
# [4] 메인 실행부
# =========================================================

def save_model(model, scaler, save_dir):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "svm_model.pkl", 'wb') as f:
        pickle.dump(model, f)
    with open(save_path / "svm_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    print("\n모델 및 스케일러 저장 완료.")


def compare_balancing_strategies(
        X_train, y_train, freq_train,
        X_test, y_test, freq_test
):
    """세 가지 균형화 전략 비교 실험"""
    strategies = ['auto', 'smote_only', 'downsample']
    results = {}

    print("\n" + "=" * 70)
    print("클래스 균형화 전략 비교 실험")
    print("=" * 70)

    for strategy in strategies:
        print("\n" + "-" * 70)
        print(f"전략: {strategy.upper()}")
        print("-" * 70)

        model, scaler, acc, y_pred = train_svm_model_fixed(
            X_train, y_train, freq_train,
            X_test, y_test, freq_test,
            balance_strategy=strategy
        )

        results[strategy] = {
            'accuracy': acc,
            'model': model,
            'scaler': scaler,
            'predictions': y_pred
        }

        print(f"\n{strategy} 전략 정확도: {acc * 100:.2f}%")

    best_strategy = max(results, key=lambda k: results[k]['accuracy'])
    print("\n" + "=" * 70)
    print(
        f"최고 성능 전략: {best_strategy.upper()} "
        f"(정확도: {results[best_strategy]['accuracy'] * 100:.2f}%)"
    )
    print("=" * 70)

    return results, best_strategy


if __name__ == "__main__":
    data_path = f"{OUTPUT_DIR}/classification_data.npz"
    X_train, y_train, freq_train, X_test, y_test, freq_test = load_classification_data_fixed(data_path)

    results, best_strategy = compare_balancing_strategies(
        X_train, y_train, freq_train,
        X_test, y_test, freq_test
    )

    best_model = results[best_strategy]['model']
    best_scaler = results[best_strategy]['scaler']
    best_predictions = results[best_strategy]['predictions']
    best_accuracy = results[best_strategy]['accuracy']

    save_model(best_model, best_scaler, MODEL_DIR)
    analyze_prediction_errors(y_test, best_predictions, freq_test, MODEL_DIR)
    analyze_frequency_mapping_loss(y_test, best_predictions, freq_test, MODEL_DIR)

    print("\n" + "=" * 70)
    print(f"최종 학습 완료 (전략: {best_strategy})")
    print(f"최종 정확도: {best_accuracy * 100:.2f}%")
    print(f"그래프 경로: {MODEL_DIR}/training_loss_curve.png")
    print("=" * 70)