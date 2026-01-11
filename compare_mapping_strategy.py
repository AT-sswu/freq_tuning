"""
Energy Harvester Simulation: Generalized Linear Array VEH Power Comparison

Compares pre-mapping (raw frequency) vs post-mapping (ML model predicted frequency)
using 1-DOF VEH power formula with frequency response function.

Power Formula:
P(ω_input, ω_res) = [m * Y² * ω_input⁵ / (4 * ζ_tot² * ω_res⁴)] * |H(ω_input/ω_res)|²
where |H(r)| = 1 / √[(1-r²)² + (2ζ r)²], r = ω_input/ω_res
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft, fftfreq
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')


def calculate_veh_power(frequency, resonance_freq, 
                        mass=0.001, Y=0.1, Q_factor=10, scale_factor=1e6):
    """
    Generalized 1-DOF VEH Power Calculation using Frequency Response Function
    
    Power Formula:
    P(ω_input, ω_res) = [m * Y² * ω_input⁵ / (4 * ζ_tot² * ω_res⁴)] * |H(r)|²
    
    where:
    - |H(r)| = 1 / √[(1-r²)² + (2ζ r)²]
    - r = ω_input / ω_res (frequency ratio)
    - ζ_tot = 1 / (2 * Q_factor) (derived from Q factor)
    
    Parameters:
    -----------
    frequency : float
        Input frequency (Hz)
    resonance_freq : float
        Harvester resonance frequency (Hz)
    mass : float
        Harvester mass (kg), default 1g
    Y : float
        Displacement amplitude (m), default 0.1m
    Q_factor : float
        Quality factor (dimensionless), default 10
    scale_factor : float
        Unit conversion factor (1e6 → μW)
    
    Returns:
    --------
    power : float
        Output power (μW)
    """
    if frequency <= 0 or resonance_freq <= 0:
        return 0.0
    
    # Angular frequencies
    omega = 2 * np.pi * frequency
    omega_n = 2 * np.pi * resonance_freq
    
    # Damping ratio from Q factor
    zeta_tot = 1 / (2 * Q_factor)
    
    # Frequency ratio
    r = omega / omega_n
    
    # Frequency response magnitude |H(r)|
    numerator = (1 - r**2)**2 + (2 * zeta_tot * r)**2
    if numerator == 0:
        return 0.0
    
    H_magnitude_squared = 1 / numerator
    
    # Power formula: P = [m * Y² * ω^5 / (4 * ζ²_tot * ω_n^4)] * |H(r)|²
    power_coefficient = (mass * Y**2 * omega**5) / (4 * zeta_tot**2 * omega_n**4)
    power = power_coefficient * H_magnitude_squared
    
    return power * scale_factor  # convert to μW


def load_ml_models():
    """
    Load trained ML models (RF, SVM, KNN, XGB) from workspace using pickle or joblib
    Returns dict of {model_name: (model, scaler)}
    """
    models = {}
    workspace_root = Path(__file__).parent
    
    model_configs = {
        'SVM': [
            workspace_root / 'freq_tuning_svm' / 'svm_512' / 'model_results' / 'svm_model.joblib',
            workspace_root / 'freq_tuning_svm' / 'svm_512' / 'model_results' / 'svm_model.pkl',
        ],
        'RF': [
            workspace_root / 'freq_tuning_rf' / 'rf_512' / 'model_results' / 'rf_model.joblib',
            workspace_root / 'freq_tuning_rf' / 'rf_512' / 'model_results' / 'rf_model.pkl',
        ],
        'KNN': [
            workspace_root / 'freq_tuning_knn' / 'knn_512' / 'model_results' / 'knn_model.joblib',
            workspace_root / 'freq_tuning_knn' / 'knn_512' / 'model_results' / 'knn_model.pkl',
        ],
        'XGB': [
            workspace_root / 'freq_tuning_xgb' / 'xgb_512' / 'model_results' / 'xgb_model.joblib',
            workspace_root / 'freq_tuning_xgb' / 'xgb_512' / 'model_results' / 'xgb_model.pkl',
        ],
    }
    
    scaler_configs = {
        'SVM': [
            workspace_root / 'freq_tuning_svm' / 'svm_512' / 'model_results' / 'scaler.joblib',
            workspace_root / 'freq_tuning_svm' / 'svm_512' / 'model_results' / 'scaler.pkl',
        ],
        'RF': [
            workspace_root / 'freq_tuning_rf' / 'rf_512' / 'model_results' / 'scaler.joblib',
            workspace_root / 'freq_tuning_rf' / 'rf_512' / 'model_results' / 'scaler.pkl',
        ],
        'KNN': [
            workspace_root / 'freq_tuning_knn' / 'knn_512' / 'model_results' / 'scaler.joblib',
            workspace_root / 'freq_tuning_knn' / 'knn_512' / 'model_results' / 'scaler.pkl',
        ],
        'XGB': [
            workspace_root / 'freq_tuning_xgb' / 'xgb_512' / 'model_results' / 'scaler.joblib',
            workspace_root / 'freq_tuning_xgb' / 'xgb_512' / 'model_results' / 'scaler.pkl',
        ],
    }
    
    for model_name, model_paths in model_configs.items():
        model = None
        model_file = None
        
        # Try to find and load model
        for path in model_paths:
            if path.exists():
                try:
                    model = joblib.load(path)
                    model_file = path
                    break
                except:
                    continue
        
        if model is None:
            print(f"  ⊘ {model_name} model not found in any location")
            continue
        
        # Try to load scaler
        scaler = None
        for path in scaler_configs[model_name]:
            if path.exists():
                try:
                    scaler = joblib.load(path)
                    break
                except:
                    continue
        
        models[model_name] = (model, scaler)
        print(f"  ✓ Loaded {model_name} model from {model_file.name}")
    
    return models


def predict_resonance_frequency(features, model, scaler, resonance_freqs=[40, 50, 60]):
    """
    Predict resonance frequency using ML model
    """
    try:
        features_2d = features.reshape(1, -1)
        
        if scaler is not None:
            features_2d = scaler.transform(features_2d)
        
        # Predict class index
        predicted_idx = model.predict(features_2d)[0]
        predicted_freq = resonance_freqs[int(predicted_idx)]
        
        return predicted_freq
    except Exception as e:
        # Fallback: return nearest resonance frequency
        return resonance_freqs[np.argmax(features)]


def extract_frequency_features_with_peak(signal, sample_rate, target_bands=[40, 50, 60]):
    """
    FFT-based frequency feature extraction + peak frequency return
    """
    n = len(signal)
    fft_vals = fft(signal)
    freqs = fftfreq(n, 1 / sample_rate)
    
    positive_mask = freqs > 0
    freqs = freqs[positive_mask]
    fft_vals = np.abs(fft_vals[positive_mask])
    
    # normalization
    fft_energy_sum = np.sum(fft_vals ** 2)
    if fft_energy_sum > 0:
        fft_vals_normalized = fft_vals / np.sqrt(fft_energy_sum)
    else:
        fft_vals_normalized = fft_vals
    
    # peak frequency (dominant frequency)
    peak_idx = np.argmax(fft_vals)
    peak_frequency = freqs[peak_idx]
    
    # energy features for each resonance frequency
    features = []
    for target_freq in target_bands:
        band_mask = (freqs >= target_freq - 10) & (freqs <= target_freq + 10)
        band_energy = np.sum(fft_vals_normalized[band_mask] ** 2)
        features.append(band_energy)
    
    return np.array(features), peak_frequency


def calculate_sample_rate(df, time_column='Time_us'):
    """Calculate sampling frequency from data"""
    if time_column not in df.columns:
        time_column = df.columns[0]
    time_data = df[time_column].dropna().values
    time_diffs = np.diff(time_data)
    avg_time_diff_us = np.mean(time_diffs)
    sample_rate = 1 / (avg_time_diff_us / 1_000_000)
    return sample_rate


def sliding_window_split(signal, window_size=512, stride=256):
    """Window split"""
    windows = []
    for start in range(0, len(signal) - window_size + 1, stride):
        window = signal[start:start + window_size]
        windows.append(window)
    return windows


def analyze_csv_file_with_models(csv_path, window_size=512, stride=256, 
                                   ml_models=None, resonance_freqs=[40, 50, 60]):
    """
    Single CSV file analysis with ML model predictions
    - Pre-mapping: use peak frequency directly
    - Post-mapping: use ML models (SVM/RF/KNN) to predict resonance frequency
    """
    try:
        df = pd.read_csv(csv_path)
        if 'Accel_Z' not in df.columns:
            return None
        
        sample_rate = calculate_sample_rate(df)
        signal = df['Accel_Z'].dropna().values
        
        # window split
        windows = sliding_window_split(signal, window_size, stride)
        
        results = []
        
        for win_idx, window in enumerate(windows):
            features, peak_freq = extract_frequency_features_with_peak(
                window, sample_rate, resonance_freqs
            )
            
            # === Pre-mapping: use peak frequency directly ===
            powers_before = [
                calculate_veh_power(
                    frequency=peak_freq,
                    resonance_freq=res_freq
                )
                for res_freq in resonance_freqs
            ]
            label_before_idx = np.argmax(powers_before)
            label_before = resonance_freqs[label_before_idx]
            
            result_row = {
                'window_idx': win_idx,
                'peak_freq_original': peak_freq,
                'power_before_40Hz': powers_before[0],
                'power_before_50Hz': powers_before[1],
                'power_before_60Hz': powers_before[2],
                'label_before': label_before,
                'max_power_before': max(powers_before),
            }
            
            # === Post-mapping: ML model predictions ===
            if ml_models:
                for model_name, (model, scaler) in ml_models.items():
                    # Predict resonance frequency
                    mapped_freq = predict_resonance_frequency(
                        features, model, scaler, resonance_freqs
                    )
                    
                    # Calculate powers with mapped frequency
                    powers_after = [
                        calculate_veh_power(
                            frequency=mapped_freq,
                            resonance_freq=res_freq
                        )
                        for res_freq in resonance_freqs
                    ]
                    
                    result_row[f'{model_name}_mapped_freq'] = mapped_freq
                    result_row[f'{model_name}_power_40Hz'] = powers_after[0]
                    result_row[f'{model_name}_power_50Hz'] = powers_after[1]
                    result_row[f'{model_name}_power_60Hz'] = powers_after[2]
                    result_row[f'{model_name}_max_power'] = max(powers_after)
                    result_row[f'{model_name}_improvement'] = max(powers_after) - max(powers_before)
            
            results.append(result_row)
        
        return pd.DataFrame(results)
    
    except Exception as e:
        print(f"ERROR processing {csv_path}: {e}")
        return None


def analyze_folder_with_models(folder_path, window_size=512, stride=256, ml_models=None):
    """
    Analyze all CSV files in folder with ML model predictions
    """
    data_path = Path(folder_path)
    csv_files = sorted(list(data_path.glob("*.csv")))
    
    all_results = []
    resonance_freqs = [40, 50, 60]
    
    print(f"\n{'='*80}")
    print(f"Analysis: Generalized VEH Power with ML Models")
    print(f"{'='*80}")
    print(f"Models loaded: {list(ml_models.keys()) if ml_models else 'None'}")
    print(f"Resonance frequencies: {resonance_freqs}")
    print(f"FFT window size: {window_size}, stride: {stride}\n")
    
    for csv_file in csv_files:
        print(f"  Processing: {csv_file.name}...", end=" ")
        df_result = analyze_csv_file_with_models(
            csv_file, window_size, stride, ml_models, resonance_freqs
        )
        
        if df_result is not None:
            df_result['file_name'] = csv_file.name
            all_results.append(df_result)
            print(f"✓ ({len(df_result)} windows)")
        else:
            print("⊘ (skipped)")
    
    if not all_results:
        print("No analysis results")
        return None
    
    # Combined statistics
    combined_df = pd.concat(all_results, ignore_index=True)
    
    # Print statistics
    print(f"\n{'='*80}")
    print(f"[OVERALL STATISTICS]")
    print(f"{'='*80}")
    print(f"Total windows: {len(combined_df)}")
    print(f"Total files: {len(csv_files)}")
    
    print(f"\n[PRE-MAPPING POWER (Raw Frequency)]")
    avg_power_before = combined_df['max_power_before'].mean()
    print(f"  Average max power: {avg_power_before:12.2f} μW")
    print(f"  Min power: {combined_df['max_power_before'].min():12.2f} μW")
    print(f"  Max power: {combined_df['max_power_before'].max():12.2f} μW")
    print(f"  Std power: {combined_df['max_power_before'].std():12.2f} μW")
    
    if ml_models:
        for model_name in ml_models.keys():
            print(f"\n[POST-MAPPING POWER ({model_name})]")
            col_name = f'{model_name}_max_power'
            if col_name in combined_df.columns:
                avg_power_after = combined_df[col_name].mean()
                improvement = avg_power_after - avg_power_before
                improvement_ratio = (improvement / avg_power_before * 100) if avg_power_before > 0 else 0
                
                print(f"  Average max power: {avg_power_after:12.2f} μW")
                print(f"  Min power: {combined_df[col_name].min():12.2f} μW")
                print(f"  Max power: {combined_df[col_name].max():12.2f} μW")
                print(f"  Std power: {combined_df[col_name].std():12.2f} μW")
                print(f"  Power improvement: {improvement:12.2f} μW ({improvement_ratio:+.1f}%)")
    
    return combined_df


def generate_visualizations(results_df, ml_models, output_path, window_size=512):
    """
    Generate 6-subplot visualization matching the reference graph
    """
    if results_df is None or len(results_df) == 0:
        print("No results to visualize")
        return
    
    plt.style.use('seaborn-v0_8-whitegrid')
    resonance_freqs = [40, 50, 60]
    model_names = list(ml_models.keys()) if ml_models else []
    
    # Create figure with 2x3 subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Model Performance Comparison - Power Improvement Analysis (Window Size: {window_size})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Define colors
    pre_color = '#FF6B6B'
    model_colors = {'SVM': '#CD5C5C', 'RF': '#4ECDC4', 'KNN': '#F39C12', 'XGB': '#95E1D3'}
    
    # ========== (1) Mean Power - Pre-Mapping vs All Models ==========
    ax = axes[0, 0]
    
    categories = ['Pre-Mapping'] + model_names
    mean_powers = [results_df['max_power_before'].mean()] + \
                  [results_df[f'{m}_max_power'].mean() for m in model_names]
    
    bars = ax.bar(categories, mean_powers, 
                  color=[pre_color] + [model_colors.get(m, '#888888') for m in model_names],
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, val in zip(bars, mean_powers):
        height = bar.get_height()
        label_text = f'{val:.2e}' if val > 1e9 else f'{val:.2f}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
               label_text, ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_ylabel('Mean Power (μW)', fontsize=11, fontweight='bold')
    ax.set_title(f'(1) Mean Power - Window {window_size}', fontsize=12, fontweight='bold')
    ax.set_ylim(0, max(mean_powers) * 1.15)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # ========== (2) Power Improvement % ==========
    ax = axes[0, 1]
    
    baseline_power = results_df['max_power_before'].mean()
    improvements_pct = [(results_df[f'{m}_max_power'].mean() - baseline_power) / baseline_power * 100 
                       for m in model_names]
    
    bars = ax.bar(model_names, improvements_pct,
                  color=[model_colors.get(m, '#888888') for m in model_names],
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars, improvements_pct):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:+.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_ylabel('Power Improvement (%)', fontsize=11, fontweight='bold')
    ax.set_title(f'(2) Power Improvement % - Window {window_size}', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # ========== (3) Power Improvement Percentage (duplicate for symmetry) ==========
    ax = axes[0, 2]
    
    bars = ax.bar(model_names, improvements_pct,
                  color=[model_colors.get(m, '#888888') for m in model_names],
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars, improvements_pct):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:+.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_ylabel('Power Improvement Percentage', fontsize=11, fontweight='bold')
    ax.set_title(f'(3) Power Improvement Percentage - Window {window_size}', fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # ========== (4) Pre-Mapping vs Post-Mapping Power Output ==========
    ax = axes[1, 0]
    
    x_pos = np.arange(len(model_names))
    width = 0.35
    
    pre_vals = [results_df['max_power_before'].mean()] * len(model_names)
    post_vals = [results_df[f'{m}_max_power'].mean() for m in model_names]
    
    bars1 = ax.bar(x_pos - width/2, pre_vals, width, label='Pre-Mapping', 
                   color=pre_color, alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, post_vals, width, label='Post-Mapping',
                   color='#2ECC71', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Mean Power (μW)', fontsize=11, fontweight='bold')
    ax.set_title('(4) Pre-Mapping vs Post-Mapping Power Output', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # ========== (5) Power Variability (Standard Deviation) ==========
    ax = axes[1, 1]
    
    pre_std = results_df['max_power_before'].std()
    post_stds = [results_df[f'{m}_max_power'].std() for m in model_names]
    
    variability = [pre_std] + post_stds
    var_categories = ['Pre-Mapping'] + model_names
    
    bars = ax.bar(var_categories, variability,
                  color=[pre_color] + [model_colors.get(m, '#888888') for m in model_names],
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    for bar, val in zip(bars, variability):
        height = bar.get_height()
        label_text = f'{val:.2e}' if val > 1e9 else f'{val:.2f}'
        ax.text(bar.get_x() + bar.get_width()/2., height,
               label_text, ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax.set_ylabel('Standard Deviation (μW)', fontsize=11, fontweight='bold')
    ax.set_title(f'(5) Power Consistency (Lower = Better) - Window {window_size}', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # ========== (6) Performance Summary Table ==========
    ax = axes[1, 2]
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = []
    table_data.append(['Model', 'Mean Power', 'Improvement', 'Status'])
    
    for model_name in model_names:
        mean_pow = results_df[f'{model_name}_max_power'].mean()
        improve = (mean_pow - baseline_power) / baseline_power * 100
        status = 'Very Good' if improve > 120 else 'Good' if improve > 100 else 'Acceptable'
        
        table_data.append([
            model_name,
            f'{mean_pow:.2e}' if mean_pow > 1e9 else f'{mean_pow:.2f}',
            f'{improve:+.1f}%',
            status
        ])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.2, 0.3, 0.2, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style the table
    for i in range(len(table_data)):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#34495E')
                cell.set_text_props(weight='bold', color='white')
            else:
                if j == 0:
                    cell.set_facecolor('#ECF0F1')
                else:
                    cell.set_facecolor('#F8F9F9')
    
    ax.set_title(f'(6) Performance Summary - Window {window_size}', fontsize=12, fontweight='bold', pad=20)
    
    # Save figure
    plt.tight_layout()
    output_file = output_path / f'model_comparison_w{window_size}.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ 6-subplot graph saved: model_comparison_w{window_size}.png")
    plt.close()
    
    print(f"\n✓ All visualizations saved to: {output_path}")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    import sys
    
    workspace_root = Path(__file__).parent
    
    print("\n" + "="*80)
    print("GENERALIZED LINEAR ARRAY VEH POWER ANALYSIS")
    print("="*80)
    
    # Load ML models
    print("\n[Loading ML Models]")
    ml_models = load_ml_models()
    
    if not ml_models:
        print("WARNING: No ML models loaded. Proceeding with pre-mapping only.")
    
    # Default data folder
    data_folder = workspace_root / 'data_v3'
    
    if not data_folder.exists():
        print(f"ERROR: Data folder not found: {data_folder}")
        sys.exit(1)
    
    # Analyze for different window sizes
    print(f"\n[Analyzing Data]")
    
    for window_size in [512, 1024, 2048]:
        print(f"\n--- Processing Window Size: {window_size} ---")
        results_df = analyze_folder_with_models(
            str(data_folder),
            window_size=window_size,
            stride=window_size // 2,
            ml_models=ml_models
        )
        
        if results_df is not None:
            # Save results
            output_path = workspace_root / 'model_results'
            output_path.mkdir(exist_ok=True)
            
            csv_output = output_path / f'veh_power_analysis_w{window_size}.csv'
            results_df.to_csv(csv_output, index=False)
            print(f"\n✓ Results saved: {csv_output}")
            
            # Generate visualizations
            print(f"\n[Generating Visualizations for Window Size {window_size}]")
            generate_visualizations(results_df, ml_models, output_path, window_size=window_size)
        
        print(f"\n{'='*80}")
        print(f"ANALYSIS COMPLETE - Window Size {window_size}")
        print(f"{'='*80}")
