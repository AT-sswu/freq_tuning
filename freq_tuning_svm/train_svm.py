import numpy as np
import pickle
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from config import OUTPUT_DIR, MODEL_DIR, RESONANCE_FREQS


def load_classification_data(data_path):
    """분류 데이터 로드"""
    data = np.load(data_path)
    X = data['X']
    y = data['y']
    freqs = data['freqs']
    return X, y, freqs


def advanced_balance_dataset(X, y, strategy='auto', random_state=42):
    """
    고급 클래스 균형화 전략

    1. 과대 샘플링 (SMOTE) - 소수 클래스 증강
    2. 과소 샘플링 (RandomUnderSampler) - 다수 클래스 축소
    """
    print("\n[클래스 균형화 전]")
    for i, freq in enumerate(RESONANCE_FREQS):
        count = np.sum(y == i)
        print(f"  클래스 {i} ({freq}Hz): {count}개 ({count / len(y) * 100:.1f}%)")

    # 전략 1: SMOTE + UnderSampling
    if strategy == 'auto':
        # 목표: 중간값으로 맞추기
        class_counts = [np.sum(y == i) for i in range(len(RESONANCE_FREQS))]
        target_count = int(np.median(class_counts))

        print(f"\n전략: SMOTE + UnderSampling (목표: {target_count}개)")

        # Step 1: SMOTE로 소수 클래스 증강
        smote = SMOTE(random_state=random_state, k_neighbors=3)
        X_temp, y_temp = smote.fit_resample(X, y)

        # Step 2: 다수 클래스 축소
        sampling_strategy = {i: target_count for i in range(len(RESONANCE_FREQS))}
        under_sampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
        X_balanced, y_balanced = under_sampler.fit_resample(X_temp, y_temp)

    elif strategy == 'smote_only':
        # 전략 2: SMOTE만 사용
        print(f"\n전략: SMOTE only")
        smote = SMOTE(random_state=random_state, k_neighbors=3)
        X_balanced, y_balanced = smote.fit_resample(X, y)

    elif strategy == 'downsample':
        # 전략 3: 다운샘플링만 사용
        class_counts = [np.sum(y == i) for i in range(len(RESONANCE_FREQS))]
        min_count = min(class_counts)

        print(f"\n전략: 다운샘플링 only (목표: {min_count}개)")
        sampling_strategy = {i: min_count for i in range(len(RESONANCE_FREQS))}
        under_sampler = RandomUnderSampler(
            sampling_strategy=sampling_strategy,
            random_state=random_state
        )
        X_balanced, y_balanced = under_sampler.fit_resample(X, y)

    print("\n[클래스 균형화 후]")
    for i, freq in enumerate(RESONANCE_FREQS):
        count = np.sum(y_balanced == i)
        print(f"  클래스 {i} ({freq}Hz): {count}개 ({count / len(y_balanced) * 100:.1f}%)")

    return X_balanced, y_balanced


def train_svm_model(X, y, balance_strategy='auto', random_state=42):
    """SVM 모델 학습"""
    print(f"\n{'=' * 70}")
    print(f"{'SVM 모델 학습 (고급 전략)':^70}")
    print(f"{'=' * 70}")
    print(f"원본 데이터 형상: X={X.shape}, y={y.shape}")

    # 클래스 균형화
    X_balanced, y_balanced = advanced_balance_dataset(X, y, strategy=balance_strategy, random_state=random_state)

    # Train-Test 분할 (층화 샘플링)
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced,
        test_size=0.2,
        random_state=random_state,
        stratify=y_balanced
    )

    print(f"\n학습 데이터: {len(X_train)}개")
    print(f"테스트 데이터: {len(X_test)}개")

    # 정규화
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SVM 하이퍼파라미터 튜닝
    print(f"\n{'[하이퍼파라미터 탐색]':^70}")
    print(f"{'-' * 70}")

    best_accuracy = 0
    best_params = None
    best_model = None

    # 그리드 서치 (간단 버전)
    param_grid = [
        {'kernel': 'rbf', 'C': 1, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 100, 'gamma': 'scale'},
        {'kernel': 'rbf', 'C': 10, 'gamma': 0.1},
        {'kernel': 'rbf', 'C': 10, 'gamma': 0.01},
    ]

    for params in param_grid:
        model = SVC(**params, class_weight='balanced', random_state=random_state)
        model.fit(X_train_scaled, y_train)
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

    return best_model, scaler, best_accuracy


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
    X, y, freqs = load_classification_data(data_path)

    # 여러 전략 시도
    strategies = ['auto', 'smote_only', 'downsample']
    results = {}

    print(f"\n{'=' * 70}")
    print(f"{'여러 균형화 전략 비교':^70}")
    print(f"{'=' * 70}")

    for strategy in strategies:
        print(f"\n\n{'#' * 70}")
        print(f"{'전략: ' + strategy:^70}")
        print(f"{'#' * 70}")

        try:
            model, scaler, accuracy = train_svm_model(
                X, y,
                balance_strategy=strategy,
                random_state=42
            )
            results[strategy] = accuracy
        except Exception as e:
            print(f"✗ 오류 발생: {e}")
            results[strategy] = 0

    # 최고 전략 선택
    print(f"\n{'=' * 70}")
    print(f"{'전략 비교 결과':^70}")
    print(f"{'=' * 70}")

    for strategy, accuracy in results.items():
        print(f"  {strategy:20s}: {accuracy * 100:.2f}%")

    best_strategy = max(results, key=results.get)
    print(f"\n✓ 최적 전략: {best_strategy} (정확도: {results[best_strategy] * 100:.2f}%)")

    # 최적 전략으로 최종 학습 및 저장
    print(f"\n{'=' * 70}")
    print(f"{'최종 모델 학습 및 저장':^70}")
    print(f"{'=' * 70}")

    model, scaler, accuracy = train_svm_model(
        X, y,
        balance_strategy=best_strategy,
        random_state=42
    )

    save_model(model, scaler, MODEL_DIR)

    print(f"\n{'=' * 70}")
    print(f"✓ 학습 완료! 최종 정확도: {accuracy * 100:.2f}%")
    print(f"{'=' * 70}")