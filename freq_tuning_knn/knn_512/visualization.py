import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from pathlib import Path
import seaborn as sns

def calculate_snr_performance(y_true, y_pred, snr_values):
    """SNR별 분류 성능 계산"""
    unique_snrs = np.unique(snr_values[snr_values != None])
    performance = {}
    
    for snr in unique_snrs:
        mask = snr_values == snr
        if mask.sum() > 0:
            acc = (y_true[mask] == y_pred[mask]).sum() / mask.sum()
            performance[snr] = acc
    
    return performance

def plot_snr_performance(snr_performance, output_dir):
    """SNR별 성능 시각화"""
    if not snr_performance:
        print("SNR 데이터가 없습니다.")
        return
    
    snrs = sorted(snr_performance.keys())
    accuracies = [snr_performance[snr] for snr in snrs]
    
    plt.figure(figsize=(10, 6))
    plt.plot(snrs, accuracies, 'o-', linewidth=2, markersize=8, color='#2E86AB')
    plt.xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('SNR별 분류 성능', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = Path(output_dir) / "snr_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"SNR 성능 그래프 저장: {output_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """혼동 행렬 시각화"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['40Hz', '50Hz', '60Hz'],
                yticklabels=['40Hz', '50Hz', '60Hz'],
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"혼동 행렬 저장: {output_path}")
    plt.close()

def plot_class_performance(y_true, y_pred, output_dir):
    """클래스별 성능 지표 시각화"""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    class_names = ['40Hz', '50Hz', '60Hz']
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8, color='#A23B72')
    ax.bar(x, recall, width, label='Recall', alpha=0.8, color='#F18F01')
    ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8, color='#C73E1D')
    
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('클래스별 분류 성능', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "class_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"클래스 성능 그래프 저장: {output_path}")
    plt.close()

def plot_combined_performance(y_true, y_pred, output_dir):
    """통합 성능 분석 시각화"""
    cm = confusion_matrix(y_true, y_pred)
    accuracy = (y_true == y_pred).sum() / len(y_true)
    
    fig = plt.figure(figsize=(12, 8))
    
    # 혼동 행렬
    ax1 = plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax1,
                xticklabels=['40Hz', '50Hz', '60Hz'],
                yticklabels=['40Hz', '50Hz', '60Hz'])
    ax1.set_title('Confusion Matrix', fontweight='bold')
    
    # 정확도
    ax2 = plt.subplot(2, 2, 2)
    ax2.text(0.5, 0.5, f'Accuracy\n{accuracy:.2%}', 
             ha='center', va='center', fontsize=20, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='#2E86AB', alpha=0.3))
    ax2.axis('off')
    ax2.set_title('Overall Performance', fontweight='bold')
    
    # 클래스별 정확도
    ax3 = plt.subplot(2, 2, 3)
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    bars = ax3.bar(['40Hz', '50Hz', '60Hz'], class_accuracies, color=['#A23B72', '#F18F01', '#C73E1D'], alpha=0.7)
    ax3.set_ylabel('Accuracy', fontweight='bold')
    ax3.set_title('Class-wise Accuracy', fontweight='bold')
    ax3.set_ylim([0, 1.1])
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}', ha='center', va='bottom', fontweight='bold')
    
    # 샘플 분포
    ax4 = plt.subplot(2, 2, 4)
    class_counts = np.bincount(y_true)
    ax4.pie(class_counts, labels=['40Hz', '50Hz', '60Hz'], autopct='%1.1f%%',
            colors=['#A23B72', '#F18F01', '#C73E1D'], startangle=90)
    ax4.set_title('Class Distribution (Test Set)', fontweight='bold')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "combined_performance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"통합 성능 분석 그래프 저장: {output_path}")
    plt.close()
