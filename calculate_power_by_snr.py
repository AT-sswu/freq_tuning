"""
VEH Power Calculation by SNR Level
각 SNR 레벨별 전력 개선 분석 (-10, -5, 0, 5, 10 dB)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from matplotlib import font_manager as fm

# 폰트 설정 - FFT 그래프와 동일하게
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'AppleGothic']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'


def calculate_veh_power(frequency, resonance_freq, mass=0.001, Y=0.1, Q_factor=10, scale_factor=1e6):
    """VEH 전력 계산"""
    omega = 2 * np.pi * frequency
    omega_n = 2 * np.pi * resonance_freq
    omega_ratio = omega / omega_n
    
    zeta_e = 0.05
    zeta_m = 0.02
    zeta_T = zeta_e + zeta_m
    
    numerator = mass * zeta_e * omega_n * (omega**2) * (omega_ratio**3) * (Y**2)
    denominator = ((2 * zeta_T * omega_ratio)**2 + (1 - omega_ratio**2)**2)
    power = numerator / denominator
    
    return power * scale_factor


def load_predictions_by_snr(model_type, snr_db, window_size=1024):
    """특정 SNR 레벨의 예측 결과 로드"""
    base_path = Path(f"/Users/seohyeon/AT_freq_tuning/freq_tuning_{model_type}/{model_type}_{window_size}")
    
    # 예측 결과 로드
    predictions_path = base_path / "preprocessed_data" / f"predictions_snr_{snr_db}dB.npz"
    
    if not predictions_path.exists():
        print(f"⚠️  Warning: {predictions_path} does not exist!")
        return None, None
    
    data = np.load(predictions_path)
    freq_test = data['freq_test']
    y_pred_freq = data['y_pred_freq']
    
    return freq_test, y_pred_freq


def calculate_pre_mapping_power(freq_test, resonance_freqs=[40, 50, 60]):
    """매핑 전 전력 계산"""
    powers_all_samples = []
    
    for peak_freq in freq_test:
        powers_at_resonances = []
        for res_freq in resonance_freqs:
            power = calculate_veh_power(frequency=peak_freq, resonance_freq=res_freq)
            powers_at_resonances.append(power)
        
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
    """매핑 후 전력 계산"""
    powers_all_samples = []
    
    for peak_freq, mapped_freq in zip(freq_test, y_pred_freq):
        powers_at_resonances = []
        for res_freq in resonance_freqs:
            if mapped_freq == res_freq:
                power = calculate_veh_power(frequency=mapped_freq, resonance_freq=res_freq)
            else:
                power = calculate_veh_power(frequency=peak_freq, resonance_freq=res_freq)
            powers_at_resonances.append(power)
        
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


def plot_snr_comparison(snr_db, pre_mapping, post_mapping_models):
    """특정 SNR 레벨의 전력 개선율 그래프"""
    models = ['RF', 'SVM', 'kNN', 'XGBoost']
    colors = {'RF': '#3498db', 'SVM': '#e74c3c', 'kNN': '#f39c12', 'XGBoost': '#9b59b6'}
    
    # 개선율 계산
    improvements = []
    multipliers = []
    pre_mean = pre_mapping['mean']
    
    for model in models:
        if post_mapping_models[model] is not None:
            post_mean = post_mapping_models[model]['mean']
            improvement_pct = ((post_mean - pre_mean) / pre_mean) * 100
            multiplier_pct = (post_mean / pre_mean) * 100
        else:
            improvement_pct = 0
            multiplier_pct = 100
        
        improvements.append(improvement_pct)
        multipliers.append(multiplier_pct)
    
    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 4))
    
    x_pos = np.arange(len(models))
    bars = ax.bar(x_pos, improvements, width=0.25, 
                   color=[colors[m] for m in models], 
                   alpha=0.85, edgecolor='black', linewidth=2)
    
    # 축 설정
    ax.set_ylabel('Improvement (%)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Model', fontsize=16, fontweight='bold')
    ax.set_title(f'Power Improvement % (SNR {snr_db} dB, 60 Hz)', 
                 fontsize=18, fontweight='bold', pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=14)
    ax.set_ylim(0, 40)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.yaxis.set_tick_params(labelsize=14)
    
    # 그리드
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
    ax.set_axisbelow(True)
    ax.axhline(0, color='black', linestyle='-', linewidth=1.5)
    
    # 값 표시
    for i, (model, improvement, multiplier) in enumerate(zip(models, improvements, multipliers)):
        if post_mapping_models[model] is not None:
            text_str = f'+{improvement:.1f}%\n({multiplier:.1f}%)'
            ax.text(i, improvement + 0.8, text_str, 
                   ha='center', va='bottom', fontsize=11, 
                   fontweight='bold', color=colors[model],
                   linespacing=1.3)
    
    plt.tight_layout()
    
    # 저장
    output_path = Path("/Users/seohyeon/AT_freq_tuning") / f"power_improvement_snr{snr_db}db.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Graph saved: {output_path}")
    plt.close()


def analyze_snr_level(snr_db):
    """특정 SNR 레벨 분석"""
    print(f"\n{'='*100}")
    print(f"ANALYZING SNR = {snr_db} dB")
    print(f"{'='*100}\n")
    
    model_types = {'RF': 'rf', 'SVM': 'svm', 'kNN': 'knn', 'XGBoost': 'xgb'}
    resonance_freqs = [40, 50, 60]
    
    # 첫 번째 모델로부터 테스트 데이터 로드
    freq_test, _ = load_predictions_by_snr('svm', snr_db, window_size=1024)
    
    if freq_test is None:
        print(f"⚠️  Skipping SNR = {snr_db} dB (no data found)")
        return None
    
    print(f"Test data size: {len(freq_test)} samples")
    print(f"Frequency range: {np.min(freq_test):.2f} - {np.max(freq_test):.2f} Hz\n")
    
    # 매핑 전 전력 계산
    print("Computing pre-mapping power...")
    pre_mapping = calculate_pre_mapping_power(freq_test, resonance_freqs)
    print(f"✓ Pre-mapping mean power: {pre_mapping['mean']:.4e} μW\n")
    
    # 각 모델별 매핑 후 전력 계산
    post_mapping_models = {}
    results_data = []
    
    # Pre-mapping 데이터 추가
    results_data.append({
        'SNR_dB': snr_db,
        'Model': 'Pre-Mapping',
        'Mean_Power_uW': pre_mapping['mean'],
        'Median_Power_uW': pre_mapping['median'],
        'Std_Dev_uW': pre_mapping['std'],
        'Min_Power_uW': pre_mapping['min'],
        'Max_Power_uW': pre_mapping['max'],
        'Absolute_Improvement_uW': 0,
        'Percentage_Improvement': 0
    })
    
    for model_name, model_code in model_types.items():
        print(f"Loading {model_name} model predictions...")
        freq_test_model, y_pred_freq = load_predictions_by_snr(model_code, snr_db, window_size=1024)
        
        if freq_test_model is None or y_pred_freq is None:
            print(f"⚠️  {model_name}: No data found\n")
            post_mapping_models[model_name] = None
            continue
        
        print(f"Computing post-mapping power for {model_name}...")
        post_mapping = calculate_post_mapping_power(freq_test_model, y_pred_freq, resonance_freqs)
        post_mapping_models[model_name] = post_mapping
        
        improvement_abs = post_mapping['mean'] - pre_mapping['mean']
        improvement_pct = (improvement_abs / pre_mapping['mean']) * 100
        
        print(f"✓ {model_name} post-mapping mean power: {post_mapping['mean']:.4e} μW ({improvement_pct:+.2f}%)\n")
        
        # 결과 데이터 추가
        results_data.append({
            'SNR_dB': snr_db,
            'Model': model_name,
            'Mean_Power_uW': post_mapping['mean'],
            'Median_Power_uW': post_mapping['median'],
            'Std_Dev_uW': post_mapping['std'],
            'Min_Power_uW': post_mapping['min'],
            'Max_Power_uW': post_mapping['max'],
            'Absolute_Improvement_uW': improvement_abs,
            'Percentage_Improvement': improvement_pct
        })
    
    # CSV 저장
    df = pd.DataFrame(results_data)
    csv_path = Path("/Users/seohyeon/AT_freq_tuning") / f"power_results_snr{snr_db}db.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✓ CSV saved: {csv_path}\n")
    
    # 그래프 생성
    print("Generating graph...")
    plot_snr_comparison(snr_db, pre_mapping, post_mapping_models)
    
    # 결과 요약 출력
    print(f"\n{'='*100}")
    print(f"SUMMARY: SNR = {snr_db} dB")
    print(f"{'='*100}\n")
    
    for model_name in model_types.keys():
        if post_mapping_models[model_name] is not None:
            post = post_mapping_models[model_name]
            improvement = ((post['mean'] - pre_mapping['mean']) / pre_mapping['mean']) * 100
            multiplier = post['mean'] / pre_mapping['mean']
            
            print(f"{model_name:8s}: {post['mean']:.4e} μW | +{improvement:.2f}% | {multiplier:.3f}x")
    
    print(f"\n{'='*100}\n")
    
    return results_data


def main():
    """모든 SNR 레벨 분석"""
    snr_levels = [-10, -5, 0, 5, 10]
    
    print("\n" + "="*100)
    print("VEH POWER ANALYSIS BY SNR LEVEL")
    print("="*100)
    
    all_results = []
    
    for snr_db in snr_levels:
        results = analyze_snr_level(snr_db)
        if results:
            all_results.extend(results)
    
    # 전체 결과 통합 CSV 저장
    if all_results:
        df_all = pd.DataFrame(all_results)
        csv_all_path = Path("/Users/seohyeon/AT_freq_tuning") / "power_results_all_snr_levels.csv"
        df_all.to_csv(csv_all_path, index=False, encoding='utf-8-sig')
        print(f"\n{'='*100}")
        print(f"✓ Combined CSV saved: {csv_all_path}")
        print(f"{'='*100}\n")
    
    print("\n✓ All SNR level analysis complete!\n")


if __name__ == "__main__":
    main()
