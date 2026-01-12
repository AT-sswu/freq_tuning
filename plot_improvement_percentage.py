"""
VEH Power Improvement Percentage - Single Graph
모델별 전력 개선율 비교 그래프
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def plot_improvement_percentage():
    """
    모델별 전력 개선율 비교 그래프 생성
    """
    # CSV 파일에서 데이터 로드
    csv_path = Path("/Users/seohyeon/AT_freq_tuning/veh_power_improvement_results_1024.csv")
    df = pd.read_csv(csv_path)
    
    # Pre-Mapping 제외하고 모델들만 추출
    df_models = df[df['Model'] != 'Pre-Mapping'].copy()
    
    # 데이터 추출
    models = df_models['Model'].values
    improvements = df_models['Percentage_Improvement'].values
    
    # 색상 설정
    colors = {
        'RF': '#3498db',
        'SVM': '#e74c3c', 
        'kNN': '#f39c12',
        'XGBoost': '#9b59b6'
    }
    
    bar_colors = [colors[model] for model in models]
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 막대 그래프 (넓이 줄임: 0.5 -> 0.4)
    x_pos = np.arange(len(models))
    bars = ax.bar(x_pos, improvements, width=0.4, color=bar_colors, 
                   alpha=0.85, edgecolor='black', linewidth=2)
    
    # 축 설정
    ax.set_ylabel('Improvement (%)', fontsize=18, fontweight='bold')
    ax.set_xlabel('Model', fontsize=18, fontweight='bold')
    ax.set_title('Power Improvement % by Model', fontsize=20, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=16, fontweight='bold')
    
    # y축 범위 설정 (0~40%)
    ax.set_ylim(0, 40)
    ax.yaxis.set_tick_params(labelsize=15)
    
    # 그리드 추가
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    
    # 기준선 (0%)
    ax.axhline(0, color='black', linestyle='-', linewidth=1.5)
    
    # Pre-mapping 대비 몇 배 증가했는지 계산 및 표시
    pre_mapping_mean = df[df['Model'] == 'Pre-Mapping']['Mean_Power_uW'].values[0]
    
    for i, (model, improvement) in enumerate(zip(models, improvements)):
        post_mapping_mean = df_models[df_models['Model'] == model]['Mean_Power_uW'].values[0]
        
        # 증가 배율 계산 (예: 1.35배 = 135%)
        multiplier = post_mapping_mean / pre_mapping_mean
        multiplier_pct = multiplier * 100
        
        # 막대 위에 텍스트 표시
        # 첫 번째 줄: 개선율
        # 두 번째 줄: 증가 배율
        text_str = f'+{improvement:.1f}%\n({multiplier_pct:.1f}%)'
        
        ax.text(i, improvement + 1.5, text_str, 
                ha='center', va='bottom', fontsize=13, 
                fontweight='bold', color=colors[model],
                linespacing=1.3)
    
    plt.tight_layout()
    
    # 저장
    output_path = Path("/Users/seohyeon/AT_freq_tuning") / "power_improvement_percentage_only.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n{'='*100}")
    print(f"✓ Power improvement percentage graph saved: {output_path}")
    print(f"{'='*100}\n")
    
    # 결과 출력
    print("\n" + "="*100)
    print("MODEL IMPROVEMENT SUMMARY")
    print("="*100 + "\n")
    
    for i, model in enumerate(models):
        post_mean = df_models[df_models['Model'] == model]['Mean_Power_uW'].values[0]
        multiplier = post_mean / pre_mapping_mean
        
        print(f"{model:8s}:")
        print(f"  ├─ Improvement:     +{improvements[i]:.2f}%")
        print(f"  ├─ Multiplier:      {multiplier:.3f}x (original)")
        print(f"  └─ As Percentage:   {multiplier*100:.1f}%")
        print()
    
    print("="*100 + "\n")
    
    plt.close()


if __name__ == "__main__":
    plot_improvement_percentage()
