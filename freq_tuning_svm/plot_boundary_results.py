import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from config import MODEL_DIR, RESONANCE_FREQS


def plot_boundary_results():
    """경계값 테스트 결과 시각화"""

    # CSV 파일 로드
    csv_path = Path(MODEL_DIR) / "boundary_test_results.csv"

    if not csv_path.exists():
        print(f"결과 파일이 없습니다: {csv_path}")
        print("먼저 boundary_test_from_csv.py를 실행하세요.")
        return

    results_df = pd.read_csv(csv_path)

    print("=" * 70)
    print("그래프 생성 중...")
    print("=" * 70)

    x = results_df['Input_Frequency'].values
    expected = results_df['Expected_Frequency'].values
    predicted = results_df['Predicted_Frequency'].values
    correct = results_df['Correct'].values
    error = results_df['Error'].values

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 예측 결과 산점도
    ax1 = axes[0, 0]
    colors = ['green' if c else 'red' for c in correct]

    ax1.scatter(x, expected, label='Expected', alpha=0.4, s=30,
                marker='o', edgecolors='black', linewidths=0.5)
    ax1.scatter(x, predicted, label='Predicted', alpha=0.6, s=30,
                marker='x', c=colors, linewidths=2)

    for freq in RESONANCE_FREQS:
        ax1.axhline(y=freq, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        ax1.axvspan(freq - 5, freq + 5, alpha=0.05, color='blue')

    ax1.set_xlabel('Input Frequency (Hz)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Output Frequency (Hz)', fontsize=12, fontweight='bold')
    ax1.set_title('Boundary Region Performance', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 2. 오차 분포
    ax2 = axes[0, 1]
    ax2.scatter(x, error, c=colors, alpha=0.5, s=30, edgecolors='black', linewidths=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=2)
    ax2.fill_between([x.min() - 2, x.max() + 2], -5, 5, alpha=0.1, color='green')

    ax2.set_xlabel('Input Frequency (Hz)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Prediction Error (Hz)', fontsize=12, fontweight='bold')
    ax2.set_title('Error Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. 클래스별 정확도
    ax3 = axes[1, 0]
    class_acc = []
    class_counts = []

    for freq in RESONANCE_FREQS:
        mask = results_df['Expected_Frequency'] == freq
        if mask.sum() > 0:
            acc = results_df[mask]['Correct'].mean() * 100
            count = mask.sum()
        else:
            acc = 0
            count = 0
        class_acc.append(acc)
        class_counts.append(count)

    colors_bar = ['green' if acc >= 80 else 'orange' if acc >= 60 else 'red'
                  for acc in class_acc]
    bars = ax3.bar(RESONANCE_FREQS, class_acc, color=colors_bar, alpha=0.7,
                   edgecolor='black', linewidth=2)

    for bar, acc, count in zip(bars, class_acc, class_counts):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 2,
                 f'{acc:.1f}%\n(n={count})',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax3.axhline(y=80, color='green', linestyle='--', alpha=0.5, linewidth=2)
    ax3.set_xlabel('Resonance Frequency (Hz)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 105)
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 입력 주파수 분포
    ax4 = axes[1, 1]

    for i, freq in enumerate(RESONANCE_FREQS):
        mask = results_df['Expected_Frequency'] == freq
        if mask.sum() > 0:
            class_freqs = results_df[mask]['Input_Frequency'].values
            ax4.hist(class_freqs, bins=20, alpha=0.5,
                     label=f'{freq}Hz (n={len(class_freqs)})',
                     edgecolor='black', linewidth=0.5)

    ax4.set_xlabel('Input Frequency (Hz)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax4.set_title('Input Frequency Distribution', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # 저장
    save_path = Path(MODEL_DIR) / "boundary_test_results.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"그래프 저장: {save_path}")
    print("=" * 70)

    plt.close()


if __name__ == "__main__":
    plot_boundary_results()