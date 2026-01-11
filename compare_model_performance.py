import matplotlib.pyplot as plt
import numpy as np

# 데이터 정의
models = ['SVM', 'Random Forest', 'k-NN', 'XGBoost']

# 전체 Test Accuracy
overall_accuracy = [0.8943, 0.9098, 0.9098, 0.9201]

# 그래프 생성
fig, ax = plt.subplots(figsize=(10, 6))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 막대 그래프 (width=0.5로 얇게)
bars = ax.bar(models, overall_accuracy, width=0.5, color=colors, alpha=0.8, edgecolor='black')

ax.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
ax.set_xlabel('Models', fontsize=13, fontweight='bold')
ax.set_title('Test Accuracy Comparison (Window 1024)', fontsize=15, fontweight='bold')
ax.set_ylim([0.85, 0.95])
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 값 표시
for bar, acc in zip(bars, overall_accuracy):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.002,
            f'{acc:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/seohyeon/AT_freq_tuning/model_comparison_1024.png', 
            dpi=300, bbox_inches='tight')
print("✓ 그래프 저장 완료: /Users/seohyeon/AT_freq_tuning/model_comparison_1024.png")

plt.show()
