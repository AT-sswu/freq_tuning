"""
VEH Power Calculation & Model Improvement Comparison
Pre-Mapping vs Post-Mapping Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import pandas as pd
from matplotlib import font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def calculate_veh_power(frequency, resonance_freq, mass=0.001, Y=0.1, Q_factor=10, scale_factor=1e6):
    """
    VEH 전력 계산 공식
    
    Parameters:
    -----------
    frequency : float
        입력 진동 주파수 [Hz]
    resonance_freq : float
        공진 주파수 [Hz]
    mass : float
        Proof mass [kg] (기본값: 0.001 kg)
    Y : float
        변위 진폭 [m] (기본값: 0.1 m)
    Q_factor : float
        Quality factor (기본값: 10)
    scale_factor : float
        단위 변환 계수 (기본값: 1e6 for μW)
    
    Returns:
    --------
    float : 전력 출력 [μW]
    """
    omega = 2 * np.pi * frequency
    omega_n = 2 * np.pi * resonance_freq
    
    # 주파수 비율
    omega_ratio = omega / omega_n
    
    # 감쇠비 (전기적 + 기계적)
    zeta_e = 0.05  # 전기적 감쇠비
    zeta_m = 0.02  # 기계적 감쇠비
    zeta_T = zeta_e + zeta_m  # 총 감쇠비
    
    # 분자
    numerator = mass * zeta_e * omega_n * (omega**2) * (omega_ratio**3) * (Y**2)
    
    # 분모
    denominator = ((2 * zeta_T * omega_ratio)**2 + (1 - omega_ratio**2)**2)
    
    # 전력 계산
    power = numerator / denominator
    
    # 스케일 변환 (W -> μW)
    power_scaled = power * scale_factor
    
    return power_scaled


def load_model_predictions(model_type, window_size=1024):
    """
    학습된 모델과 예측 결과 로드
    
    Parameters:
    -----------
    model_type : str
        모델 종류 ('svm', 'rf', 'knn', 'xgb')
    window_size : int
        윈도우 크기 (기본값: 1024)
    
    Returns:
    --------
    tuple : (model, freq_test, y_pred_freq)
    """
    base_path = Path(f"/Users/seohyeon/AT_freq_tuning/freq_tuning_{model_type}/{model_type}_{window_size}")
    
    # 모델 로드
    model_path = base_path / "model_results" / f"{model_type}_model_freq_snr0.pkl"
    model = joblib.load(model_path)
    
    # 예측 결과 로드
    predictions_path = base_path / "preprocessed_data" / "predictions_freq_snr0.npz"
    data = np.load(predictions_path)
    
    freq_test = data['freq_test']  # 실제 입력 주파수
    y_pred_freq = data['y_pred_freq']  # 모델이 예측한 공진 주파수
    
    return model, freq_test, y_pred_freq


def calculate_pre_mapping_power(freq_test, resonance_freqs=[40, 50, 60]):
    """
    매핑 전 전력 계산
    
    입력 주파수를 직접 사용 (매핑 없음)
    각 공진 주파수에서의 전력을 계산하고 평균을 반환
    
    Parameters:
    -----------
    freq_test : array-like
        테스트 데이터의 실제 입력 주파수 배열
    resonance_freqs : list
        공진 주파수 리스트 [40, 50, 60] Hz
    
    Returns:
    --------
    dict : 각 샘플별 전력 계산 결과
    """
    powers_all_samples = []
    
    for peak_freq in freq_test:
        # 각 공진 주파수에서의 전력 계산
        powers_at_resonances = []
        for res_freq in resonance_freqs:
            power = calculate_veh_power(
                frequency=peak_freq,  # 피크 주파수를 그대로 사용 (매핑 없음)
                resonance_freq=res_freq
            )
            powers_at_resonances.append(power)
        
        # 세 공진 주파수에서의 평균 전력
        avg_power = np.mean(powers_at_resonances)
        powers_all_samples.append(avg_power)
    
    return {
        'powers': np.array(powers_all_samples),
        'mean': np.mean(powers_all_samples),
        'median': np.median(powers_all_samples),
        'std': np.std(powers_all_samples),
        'min': np.min(powers_all_samples),
        'max': np.max(powers_all_samples)
    }


def calculate_post_mapping_power(freq_test, y_pred_freq, resonance_freqs=[40, 50, 60]):
    """
    매핑 후 전력 계산
    
    모델이 예측한 최적 공진 주파수로 매핑된 후의 전력 계산
    
    예시:
    - 입력 주파수 43Hz -> 모델이 40Hz로 매핑 -> 40Hz 공진기에서 전력 계산
    - 입력 주파수 47Hz -> 모델이 50Hz로 매핑 -> 50Hz 공진기에서 전력 계산
    
    Parameters:
    -----------
    freq_test : array-like
        테스트 데이터의 실제 입력 주파수 배열
    y_pred_freq : array-like
        모델이 예측한 공진 주파수 배열
    resonance_freqs : list
        공진 주파수 리스트 [40, 50, 60] Hz
    
    Returns:
    --------
    dict : 각 샘플별 전력 계산 결과
    """
    powers_all_samples = []
    
    for peak_freq, mapped_freq in zip(freq_test, y_pred_freq):
        # 매핑된 주파수로 전력 계산
        # 매핑 후에는 입력 주파수가 최적 공진 주파수에 정확히 일치하도록 조정됨
        powers_at_resonances = []
        for res_freq in resonance_freqs:
            if mapped_freq == res_freq:
                # 매핑된 공진 주파수에서는 최적 전력 계산
                # (입력 주파수 = 공진 주파수로 간주)
                power = calculate_veh_power(
                    frequency=mapped_freq,  # 매핑된 주파수 사용
                    resonance_freq=res_freq
                )
            else:
                # 다른 공진 주파수에서는 원래 입력 주파수 사용
                power = calculate_veh_power(
                    frequency=peak_freq,
                    resonance_freq=res_freq
                )
            powers_at_resonances.append(power)
        
        # 세 공진 주파수에서의 평균 전력
        avg_power = np.mean(powers_at_resonances)
        powers_all_samples.append(avg_power)
    
    return {
        'powers': np.array(powers_all_samples),
        'mean': np.mean(powers_all_samples),
        'median': np.median(powers_all_samples),
        'std': np.std(powers_all_samples),
        'min': np.min(powers_all_samples),
        'max': np.max(powers_all_samples)
    }


def plot_power_improvement_comparison(pre_mapping, post_mapping_models):
    """
    매핑 전/후 전력 개선 비교 그래프 생성
    
    Parameters:
    -----------
    pre_mapping : dict
        매핑 전 전력 계산 결과
    post_mapping_models : dict
        각 모델별 매핑 후 전력 계산 결과
        형식: {'RF': {...}, 'SVM': {...}, 'kNN': {...}, 'XGBoost': {...}}
    """
    models = ['RF', 'SVM', 'kNN', 'XGBoost']
    colors = {'RF': '#3498db', 'SVM': '#e74c3c', 'kNN': '#f39c12', 'XGBoost': '#9b59b6'}
    
    # Improvement % 계산
    improvements = {}
    for model in models:
        post_mean = post_mapping_models[model]['mean']
        pre_mean = pre_mapping['mean']
        improvement_pct = ((post_mean - pre_mean) / pre_mean) * 100
        improvements[model] = improvement_pct
    
    # 그래프 생성
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('VEH Power Improvement Analysis: Pre-Mapping vs Post-Mapping', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # (1) Improvement % 비교
    ax1 = axes[0, 0]
    model_names = list(improvements.keys())
    improvement_values = list(improvements.values())
    bars = ax1.bar(model_names, improvement_values, 
                   color=[colors[m] for m in model_names], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.axhline(0, color='black', linestyle='-', linewidth=1)
    ax1.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax1.set_title('(1) Power Improvement % by Model', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 값 표시
    max_val = max(improvement_values)
    for i, (model, val) in enumerate(zip(model_names, improvement_values)):
        ax1.text(i, val + max_val*0.02, f'{val:+.1f}%', 
                ha='center', fontsize=11, fontweight='bold', color=colors[model])
    
    # (2) Pre vs Post 평균 전력 비교
    ax2 = axes[0, 1]
    x_pos = np.arange(len(models))
    width = 0.35
    
    pre_values = [pre_mapping['mean']] * len(models)
    post_values = [post_mapping_models[m]['mean'] for m in models]
    
    bars1 = ax2.bar(x_pos - width/2, pre_values, width, 
                   label='Pre-Mapping', alpha=0.8, 
                   color='#e74c3c', edgecolor='black', linewidth=1.5)
    bars2 = ax2.bar(x_pos + width/2, post_values, width, 
                   label='Post-Mapping', alpha=0.8, 
                   color='#2ecc71', edgecolor='black', linewidth=1.5)
    
    ax2.set_ylabel('Mean Power (μW)', fontsize=12, fontweight='bold')
    ax2.set_title('(2) Pre-Mapping vs Post-Mapping Mean Power', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(models, fontsize=11)
    ax2.legend(fontsize=11, loc='upper left')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # (3) 전력 분포 비교 (박스플롯)
    ax3 = axes[0, 2]
    data_to_plot = [pre_mapping['powers']] + [post_mapping_models[m]['powers'] for m in models]
    bp = ax3.boxplot(data_to_plot, tick_labels=['Pre'] + models, patch_artist=True)
    
    # 색상 설정
    bp['boxes'][0].set_facecolor('#e74c3c')
    for i, model in enumerate(models, 1):
        bp['boxes'][i].set_facecolor(colors[model])
    
    ax3.set_ylabel('Power (μW)', fontsize=12, fontweight='bold')
    ax3.set_title('(3) Power Distribution', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # (4) 표준편차 비교
    ax4 = axes[1, 0]
    std_values = [post_mapping_models[m]['std'] for m in models]
    bars = ax4.bar(models, std_values, 
                   color=[colors[m] for m in models], 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Standard Deviation (μW)', fontsize=12, fontweight='bold')
    ax4.set_title('(4) Power Consistency (Lower = Better)', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # (5) 전력 계산 공식 설명
    ax5 = axes[1, 1]
    ax5.axis('off')
    
    formula_text = r"""
VEH Power Calculation Formula:

$|P| = \frac{m\zeta_e\omega_n\omega^2(\frac{\omega}{\omega_n})^3Y^2}{(2\zeta_T\frac{\omega}{\omega_n})^2 + (1-(\frac{\omega}{\omega_n})^2)^2}$

매핑 전 (Pre-Mapping):
• 입력 주파수를 직접 사용
• 예: 43Hz → 각 공진기(40, 50, 60Hz)에서 전력 계산
• 주파수 불일치로 인한 전력 손실 발생

매핑 후 (Post-Mapping):
• 모델이 최적 공진 주파수로 매핑
• 예: 43Hz → 40Hz로 매핑 → 최적 전력 생성
• 주파수 일치로 전력 출력 향상
    """
    
    ax5.text(0.5, 0.5, formula_text, transform=ax5.transAxes, 
            fontsize=9, verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax5.set_title('(5) Power Calculation Methodology', fontsize=12, fontweight='bold', pad=20)
    
    # (6) 성능 요약 테이블
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    table_data = [['Model', 'Pre (μW)', 'Post (μW)', 'Improvement']]
    
    for model in models:
        pre_val = pre_mapping['mean']
        post_val = post_mapping_models[model]['mean']
        improvement = improvements[model]
        
        if improvement > 450:
            status = f'{improvement:+.1f}% 🏆'
        elif improvement > 400:
            status = f'{improvement:+.1f}% ✓'
        else:
            status = f'{improvement:+.1f}% ○'
        
        table_data.append([
            model,
            f'{pre_val:.2e}',
            f'{post_val:.2e}',
            status
        ])
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.25, 0.25, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # 헤더 스타일링
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # 행 스타일링
    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('#ffffff')
            table[(i, j)].set_text_props(weight='bold', fontsize=9)
    
    ax6.set_title('(6) Performance Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # 저장
    output_path = Path("/Users/seohyeon/AT_freq_tuning") / "veh_power_improvement_comparison_1024.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*100}")
    print(f"✓ Power improvement comparison graph saved: {output_path}")
    print(f"{'='*100}\n")
    
    plt.close()


def print_detailed_results(pre_mapping, post_mapping_models):
    """
    상세 결과 출력
    """
    print(f"\n{'='*100}")
    print("DETAILED POWER CALCULATION RESULTS")
    print(f"{'='*100}\n")
    
    print("Pre-Mapping (매핑 전):")
    print(f"  ├─ Mean Power:        {pre_mapping['mean']:.4e} μW")
    print(f"  ├─ Median Power:      {pre_mapping['median']:.4e} μW")
    print(f"  ├─ Std Dev:           {pre_mapping['std']:.4e} μW")
    print(f"  ├─ Min Power:         {pre_mapping['min']:.4e} μW")
    print(f"  └─ Max Power:         {pre_mapping['max']:.4e} μW\n")
    
    models = ['RF', 'SVM', 'kNN', 'XGBoost']
    
    for model in models:
        post = post_mapping_models[model]
        pre_mean = pre_mapping['mean']
        improvement_abs = post['mean'] - pre_mean
        improvement_pct = (improvement_abs / pre_mean) * 100
        
        print(f"{model} Model (매핑 후):")
        print(f"  ├─ Mean Power:        {post['mean']:.4e} μW")
        print(f"  ├─ Median Power:      {post['median']:.4e} μW")
        print(f"  ├─ Std Dev:           {post['std']:.4e} μW")
        print(f"  ├─ Min Power:         {post['min']:.4e} μW")
        print(f"  ├─ Max Power:         {post['max']:.4e} μW")
        print(f"  ├─ Absolute Improvement: {improvement_abs:+.4e} μW")
        print(f"  └─ Percentage Improvement: {improvement_pct:+.2f}%\n")
    
    # 랭킹
    print(f"{'='*100}")
    print("MODEL RANKING BY IMPROVEMENT")
    print(f"{'='*100}\n")
    
    improvements = {
        model: ((post_mapping_models[model]['mean'] - pre_mapping['mean']) / pre_mapping['mean']) * 100
        for model in models
    }
    
    ranked = sorted(improvements.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (model, improvement) in enumerate(ranked, 1):
        post_mean = post_mapping_models[model]['mean']
        print(f"  {rank}. {model:8s} - {improvement:+.2f}% improvement ({post_mean:.4e} μW)")
    
    print(f"\n{'='*100}\n")


def main():
    """
    메인 실행 함수
    """
    print("\n" + "="*100)
    print("VEH POWER IMPROVEMENT ANALYSIS: PRE-MAPPING vs POST-MAPPING")
    print("="*100 + "\n")
    
    # 모델 종류
    model_types = {
        'RF': 'rf',
        'SVM': 'svm',
        'kNN': 'knn',
        'XGBoost': 'xgb'
    }
    
    # 공진 주파수
    resonance_freqs = [40, 50, 60]
    
    # 첫 번째 모델로부터 테스트 데이터 로드 (모든 모델이 같은 테스트 데이터 사용)
    _, freq_test, _ = load_model_predictions('svm', window_size=1024)
    
    print(f"테스트 데이터 크기: {len(freq_test)} samples")
    print(f"입력 주파수 범위: {np.min(freq_test):.2f} - {np.max(freq_test):.2f} Hz\n")
    
    # 매핑 전 전력 계산
    print("Computing pre-mapping power...")
    pre_mapping = calculate_pre_mapping_power(freq_test, resonance_freqs)
    print(f"✓ Pre-mapping mean power: {pre_mapping['mean']:.4e} μW\n")
    
    # 각 모델별 매핑 후 전력 계산
    post_mapping_models = {}
    
    for model_name, model_code in model_types.items():
        print(f"Loading {model_name} model predictions...")
        _, freq_test_model, y_pred_freq = load_model_predictions(model_code, window_size=1024)
        
        print(f"Computing post-mapping power for {model_name}...")
        post_mapping = calculate_post_mapping_power(freq_test_model, y_pred_freq, resonance_freqs)
        post_mapping_models[model_name] = post_mapping
        
        improvement_pct = ((post_mapping['mean'] - pre_mapping['mean']) / pre_mapping['mean']) * 100
        print(f"✓ {model_name} post-mapping mean power: {post_mapping['mean']:.4e} μW ({improvement_pct:+.2f}%)\n")
    
    # 결과 출력
    print_detailed_results(pre_mapping, post_mapping_models)
    
    # CSV 파일로 저장
    print("Saving results to CSV...")
    save_results_to_csv(pre_mapping, post_mapping_models)
    
    # 그래프 생성
    print("Generating comparison graphs...")
    plot_power_improvement_comparison(pre_mapping, post_mapping_models)
    
    print("\n✓ Analysis complete!\n")


def save_results_to_csv(pre_mapping, post_mapping_models):
    """
    결과를 CSV 파일로 저장
    """
    models = ['RF', 'SVM', 'kNN', 'XGBoost']
    
    # 요약 데이터 생성
    summary_data = []
    
    # Pre-mapping 데이터
    summary_data.append({
        'Model': 'Pre-Mapping',
        'Mean_Power_uW': pre_mapping['mean'],
        'Median_Power_uW': pre_mapping['median'],
        'Std_Dev_uW': pre_mapping['std'],
        'Min_Power_uW': pre_mapping['min'],
        'Max_Power_uW': pre_mapping['max'],
        'Absolute_Improvement_uW': 0,
        'Percentage_Improvement': 0
    })
    
    # Post-mapping 데이터 (각 모델별)
    for model in models:
        post = post_mapping_models[model]
        improvement_abs = post['mean'] - pre_mapping['mean']
        improvement_pct = (improvement_abs / pre_mapping['mean']) * 100
        
        summary_data.append({
            'Model': model,
            'Mean_Power_uW': post['mean'],
            'Median_Power_uW': post['median'],
            'Std_Dev_uW': post['std'],
            'Min_Power_uW': post['min'],
            'Max_Power_uW': post['max'],
            'Absolute_Improvement_uW': improvement_abs,
            'Percentage_Improvement': improvement_pct
        })
    
    # DataFrame 생성 및 저장
    df = pd.DataFrame(summary_data)
    output_path = Path("/Users/seohyeon/AT_freq_tuning") / "veh_power_improvement_results_1024.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"✓ Results saved to CSV: {output_path}\n")


if __name__ == "__main__":
    main()
