import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mlp_model import create_model


def normalize_power_values(y_true, y_pred):
    """
    전력값을 0-100% 범위로 정규화

    Args:
        y_true: 실제 전력 (N, 4)
        y_pred: 예측 전력 (N, 4)

    Returns:
        정규화된 y_true, y_pred (각 샘플의 최대값을 100으로)
    """
    # 각 샘플별로 정규화 (행별 정규화)
    y_true_normalized = np.zeros_like(y_true)
    y_pred_normalized = np.zeros_like(y_pred)

    for i in range(len(y_true)):
        # 실제값 정규화 (해당 샘플의 최대값을 100으로)
        max_true = np.max(y_true[i])
        if max_true > 0:
            y_true_normalized[i] = (y_true[i] / max_true) * 100
        else:
            # 음수일 경우 절대값 기준
            max_true = np.max(np.abs(y_true[i]))
            if max_true > 0:
                y_true_normalized[i] = (y_true[i] / max_true) * 100

        # 예측값 정규화 (해당 샘플의 최대값을 100으로)
        max_pred = np.max(y_pred[i])
        if max_pred > 0:
            y_pred_normalized[i] = (y_pred[i] / max_pred) * 100
        else:
            # 음수일 경우 절대값 기준
            max_pred = np.max(np.abs(y_pred[i]))
            if max_pred > 0:
                y_pred_normalized[i] = (y_pred[i] / max_pred) * 100

    return y_true_normalized, y_pred_normalized


def plot_power_vs_frequency(y_true, y_pred, target_freqs, save_dir, normalize=True):
    """
    주파수별 출력 전력 그래프 생성

    Args:
        y_true: 실제 전력 값 (N, 4)
        y_pred: 예측 전력 값 (N, 4)
        target_freqs: 공진 주파수 리스트 [30, 40, 50, 60]
        save_dir: 저장 디렉토리
        normalize: True이면 0-100% 정규화, False이면 원본값
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 정규화 옵션
    if normalize:
        y_true_plot, y_pred_plot = normalize_power_values(y_true, y_pred)
        ylabel = "Relative Power (%)"
        title_suffix = "(Normalized)"
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
        ylabel = "Power Output (Relative)"
        title_suffix = ""

    # 1. 평균 전력 vs 주파수
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) 실제 전력의 평균
    ax1 = axes[0, 0]
    avg_true = y_true_plot.mean(axis=0)
    std_true = y_true_plot.std(axis=0)
    ax1.plot(target_freqs, avg_true, 'o-', linewidth=2, markersize=8, label='Actual Power')
    ax1.fill_between(target_freqs, avg_true - std_true, avg_true + std_true, alpha=0.3)
    ax1.set_xlabel('Frequency (Hz)', fontsize=12)
    ax1.set_ylabel(ylabel, fontsize=12)
    ax1.set_title(f'Average Actual Power vs Frequency {title_suffix}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    if normalize:
        ax1.set_ylim(0, 100)

    # (2) 예측 전력의 평균
    ax2 = axes[0, 1]
    avg_pred = y_pred_plot.mean(axis=0)
    std_pred = y_pred_plot.std(axis=0)
    ax2.plot(target_freqs, avg_pred, 's-', linewidth=2, markersize=8, color='orange', label='Predicted Power')
    ax2.fill_between(target_freqs, avg_pred - std_pred, avg_pred + std_pred, alpha=0.3, color='orange')
    ax2.set_xlabel('Frequency (Hz)', fontsize=12)
    ax2.set_ylabel(ylabel, fontsize=12)
    ax2.set_title(f'Average Predicted Power vs Frequency {title_suffix}', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    if normalize:
        ax2.set_ylim(0, 100)

    # (3) 실제 vs 예측 비교
    ax3 = axes[1, 0]
    ax3.plot(target_freqs, avg_true, 'o-', linewidth=2, markersize=8, label='Actual', color='blue')
    ax3.plot(target_freqs, avg_pred, 's-', linewidth=2, markersize=8, label='Predicted', color='red')
    ax3.set_xlabel('Frequency (Hz)', fontsize=12)
    ax3.set_ylabel(ylabel, fontsize=12)
    ax3.set_title(f'Actual vs Predicted Power {title_suffix}', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    if normalize:
        ax3.set_ylim(0, 100)

    # (4) 오차율
    ax4 = axes[1, 1]
    error_percent = np.abs(avg_true - avg_pred) / (np.abs(avg_true) + 1e-8) * 100
    ax4.bar(target_freqs, error_percent, color=['green', 'blue', 'orange', 'red'], alpha=0.7)
    ax4.set_xlabel('Frequency (Hz)', fontsize=12)
    ax4.set_ylabel('Error (%)', fontsize=12)
    ax4.set_title('Prediction Error by Frequency', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    # 오차율 값 표시
    for i, (freq, err) in enumerate(zip(target_freqs, error_percent)):
        ax4.text(freq, err + 1, f'{err:.1f}%', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    filename = "power_vs_frequency_normalized.png" if normalize else "power_vs_frequency_analysis.png"
    plot_path = save_path / filename
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"✓ 전력-주파수 분석 그래프 저장: {plot_path}")

    # 2. 샘플별 상세 그래프 (처음 10개 샘플)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    num_samples = min(10, len(y_true_plot))

    for i in range(num_samples):
        ax = axes[i]
        ax.plot(target_freqs, y_true_plot[i], 'o-', label='Actual', linewidth=2, markersize=6)
        ax.plot(target_freqs, y_pred_plot[i], 's--', label='Predicted', linewidth=2, markersize=6)
        ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f'Sample {i + 1}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        if normalize:
            ax.set_ylim(0, 100)
        if i == 0:
            ax.legend()

    # 남은 subplot 숨기기
    for i in range(num_samples, 10):
        axes[i].axis('off')

    plt.tight_layout()
    filename = "sample_power_normalized.png" if normalize else "sample_power_vs_frequency.png"
    plot_path = save_path / filename
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"✓ 샘플별 전력-주파수 그래프 저장: {plot_path}")

    # 3. 최적 주파수 분포
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 실제 최적 주파수
    ax1 = axes[0]
    best_freq_true = np.array([target_freqs[np.argmax(y)] for y in y_true])
    unique_true, counts_true = np.unique(best_freq_true, return_counts=True)
    ax1.bar(unique_true, counts_true, color='blue', alpha=0.7, width=5)
    ax1.set_xlabel('Optimal Frequency (Hz)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Distribution of Actual Optimal Frequency', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # 카운트 값 표시
    for freq, count in zip(unique_true, counts_true):
        ax1.text(freq, count + 0.5, f'{count}', ha='center', va='bottom', fontsize=11)

    # 예측 최적 주파수
    ax2 = axes[1]
    best_freq_pred = np.array([target_freqs[np.argmax(y)] for y in y_pred])
    unique_pred, counts_pred = np.unique(best_freq_pred, return_counts=True)
    ax2.bar(unique_pred, counts_pred, color='red', alpha=0.7, width=5)
    ax2.set_xlabel('Optimal Frequency (Hz)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Distribution of Predicted Optimal Frequency', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 카운트 값 표시
    for freq, count in zip(unique_pred, counts_pred):
        ax2.text(freq, count + 0.5, f'{count}', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plot_path = save_path / "optimal_frequency_distribution.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    print(f"✓ 최적 주파수 분포 그래프 저장: {plot_path}")

    # 4. 통계 요약 출력
    print(f"\n{'=' * 60}")
    print(f"전력-주파수 분석 결과 {'(정규화됨)' if normalize else ''}")
    print(f"{'=' * 60}")

    for i, freq in enumerate(target_freqs):
        print(f"\n[{freq}Hz]")
        if normalize:
            print(f"  실제 평균: {avg_true[i]:.2f}% ± {std_true[i]:.2f}%")
            print(f"  예측 평균: {avg_pred[i]:.2f}% ± {std_pred[i]:.2f}%")
        else:
            print(f"  실제 평균: {avg_true[i]:.4f} ± {std_true[i]:.4f}")
            print(f"  예측 평균: {avg_pred[i]:.4f} ± {std_pred[i]:.4f}")
        print(f"  오차율: {error_percent[i]:.2f}%")

    print(f"\n[최적 주파수 예측 정확도]")
    accuracy = np.mean(best_freq_true == best_freq_pred) * 100
    print(f"  정확도: {accuracy:.2f}%")

    print(f"{'=' * 60}\n")

    return accuracy


def visualize_model_results(model_path, data_path, save_dir, normalize=True):
    """
    학습된 모델로 예측하고 시각화

    Args:
        model_path: 저장된 모델 경로 (.pth)
        data_path: 데이터 경로 (.npz)
        save_dir: 그래프 저장 디렉토리
        normalize: True이면 전력값 정규화 (0-100%), False이면 원본값
    """
    print(f"\n{'=' * 60}")
    print(f"모델 결과 시각화 시작")
    print(f"정규화: {'적용 (0-100%)' if normalize else '미적용 (원본값)'}")
    print(f"{'=' * 60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 디바이스: {device}")

    # 1. 체크포인트 로드
    print(f"\n[1/4] 모델 로드 중...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    X_mean = checkpoint['X_mean']
    X_std = checkpoint['X_std']
    y_mean = checkpoint['y_mean']
    y_std = checkpoint['y_std']
    target_freqs = checkpoint.get('target_freqs', [30, 40, 50, 60])

    print(f"  ✓ 체크포인트 로드 완료")
    print(f"  - Epoch: {checkpoint['epoch']}")
    print(f"  - Train Loss: {checkpoint['train_loss']:.4f}")
    print(f"  - Val Loss: {checkpoint['val_loss']:.4f}")

    # 2. 데이터 로드
    print(f"\n[2/4] 데이터 로드 중...")
    data = np.load(data_path)
    X = data['X']
    y = data['y']

    print(f"  ✓ 데이터 로드 완료")
    print(f"  - 입력 형상: {X.shape}")
    print(f"  - 출력 형상: {y.shape}")
    print(f"  - 공진 주파수: {target_freqs}")
    print(f"  - 출력 전력 범위: {y.min():.4f} ~ {y.max():.4f}")

    # 3. 정규화 및 예측
    print(f"\n[3/4] 예측 수행 중...")
    X_normalized = (X - X_mean) / (X_std + 1e-8)

    # 모델 생성 및 가중치 로드
    model = create_model(input_dim=X.shape[1], output_dim=y.shape[1])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # 예측
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_normalized).to(device)
        y_pred_normalized = model(X_tensor).cpu().numpy()

    # 역정규화
    y_pred = y_pred_normalized * y_std + y_mean

    print(f"  ✓ 예측 완료")
    print(f"  - 실제 범위: {y.min():.4f} ~ {y.max():.4f}")
    print(f"  - 예측 범위: {y_pred.min():.4f} ~ {y_pred.max():.4f}")

    # 4. 시각화
    print(f"\n[4/4] 그래프 생성 중...")
    accuracy = plot_power_vs_frequency(y, y_pred, target_freqs, save_dir, normalize=normalize)

    print(f"\n{'=' * 60}")
    print(f"시각화 완료!")
    print(f"{'=' * 60}")
    print(f"저장 위치: {save_dir}")
    print(f"최적 주파수 예측 정확도: {accuracy:.2f}%")
    print(f"{'=' * 60}\n")

    return y, y_pred, accuracy


if __name__ == "__main__":
    # 경로 설정
    MODEL_PATH = "/Users/seohyeon/AT_freq_tuning/model_results/best_model.pth"
    DATA_PATH = "/freq_tuning_mlp/preprocess_results/preprocessed_data_augmented_log.npz"
    SAVE_DIR = "/Users/seohyeon/AT_freq_tuning/model_results"

    # 정규화 옵션 선택
    NORMALIZE = True  # True: 0-100% 정규화, False: 원본값 (로그×효율)

    # 시각화 실행
    y_true, y_pred, accuracy = visualize_model_results(
        model_path=MODEL_PATH,
        data_path=DATA_PATH,
        save_dir=SAVE_DIR,
        normalize=NORMALIZE
    )

    print("완료! 생성된 그래프:")
    if NORMALIZE:
        print("  1. power_vs_frequency_normalized.png")
        print("  2. sample_power_normalized.png")
    else:
        print("  1. power_vs_frequency_analysis.png")
        print("  2. sample_power_vs_frequency.png")
    print("  3. optimal_frequency_distribution.png")