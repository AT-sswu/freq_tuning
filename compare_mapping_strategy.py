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
    Load trained ML models (RF, SVM, KNN) from workspace using pickle or joblib
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


def generate_visualizations(results_df, ml_models, output_path):
    """
    Generate 4 visualization graphs as requested
    """
    if results_df is None or len(results_df) == 0:
        print("No results to visualize")
        return
    
    plt.style.use('seaborn-v0_8-darkgrid')
    resonance_freqs = [40, 50, 60]
    model_names = list(ml_models.keys()) if ml_models else []
    
    # ========== Graph 1: Power Comparison (Pre vs Post) ==========
    fig1, ax = plt.subplots(figsize=(10, 6))
    
    categories = [f'{freq}Hz' for freq in resonance_freqs]
    x = np.arange(len(categories))
    width = 0.15
    
    # Pre-mapping (aggregated)
    pre_powers = [
        results_df['power_before_40Hz'].mean(),
        results_df['power_before_50Hz'].mean(),
        results_df['power_before_60Hz'].mean(),
    ]
    
    # Plot pre-mapping
    ax.bar(x - width/2 - width, pre_powers, width, label='Pre-Mapping (Raw)', alpha=0.85, color='#FF6B6B')
    
    # Plot post-mapping for each model
    colors = ['#4ECDC4', '#45B7D1', '#FFA07A']
    for idx, model_name in enumerate(model_names):
        post_powers = [
            results_df[f'{model_name}_power_40Hz'].mean(),
            results_df[f'{model_name}_power_50Hz'].mean(),
            results_df[f'{model_name}_power_60Hz'].mean(),
        ]
        ax.bar(x + (idx - len(model_names)/2) * width + width, post_powers, width, 
               label=f'Post-Mapping ({model_name})', alpha=0.85, color=colors[idx])
    
    ax.set_xlabel('Resonance Frequency', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Power (μW)', fontsize=12, fontweight='bold')
    ax.set_title('Power Comparison: Pre-Mapping vs Post-Mapping (ML Models)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'graph1_power_comparison.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Graph 1 saved: graph1_power_comparison.png")
    plt.close()
    
    # ========== Graph 2: Model Accuracy (Max Power by Model) ==========
    fig2, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data
    model_data = {
        'Pre-Mapping': results_df['max_power_before'].values,
    }
    for model_name in model_names:
        col_name = f'{model_name}_max_power'
        if col_name in results_df.columns:
            model_data[f'{model_name}'] = results_df[col_name].values
    
    # Box plot
    box_data = list(model_data.values())
    labels = list(model_data.keys())
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True, widths=0.6)
    
    # Color the boxes
    colors_box = ['#FF6B6B'] + colors[:len(model_names)]
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Maximum Power (μW)', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance: Maximum Power Distribution', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'graph2_model_performance.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Graph 2 saved: graph2_model_performance.png")
    plt.close()
    
    # ========== Graph 3: Power per Resonator (40Hz, 50Hz, 60Hz) ==========
    fig3, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax_idx, res_freq in enumerate(resonance_freqs):
        ax = axes[ax_idx]
        
        # Pre-mapping
        pre_col = f'power_before_{res_freq}Hz'
        ax.hist(results_df[pre_col], bins=30, alpha=0.6, label='Pre-Mapping', 
                color='#FF6B6B', edgecolor='black')
        
        # Post-mapping
        for model_idx, model_name in enumerate(model_names):
            post_col = f'{model_name}_power_{res_freq}Hz'
            if post_col in results_df.columns:
                ax.hist(results_df[post_col], bins=30, alpha=0.5, 
                       label=f'{model_name}', color=colors[model_idx], edgecolor='black')
        
        ax.set_xlabel('Power (μW)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax.set_title(f'{res_freq}Hz Resonator Power', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'graph3_power_per_resonator.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Graph 3 saved: graph3_power_per_resonator.png")
    plt.close()
    
    # ========== Graph 4: Power Improvement (Delta) ==========
    fig4, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate improvements
    improvements = []
    improvement_labels = []
    improvement_colors = []
    
    for model_name in model_names:
        col_name = f'{model_name}_improvement'
        if col_name in results_df.columns:
            improvements.append(results_df[col_name].values)
            improvement_labels.append(model_name)
            improvement_colors.append(colors[len(improvements)-1])
    
    # Violin plot
    if improvements:
        parts = ax.violinplot(improvements, positions=range(len(improvements)), 
                              showmeans=True, showmedians=True)
        
        # Color the violins
        for idx, pc in enumerate(parts['bodies']):
            pc.set_facecolor(improvement_colors[idx])
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(improvements)))
        ax.set_xticklabels(improvement_labels, fontsize=11)
        ax.set_ylabel('Power Improvement (μW)', fontsize=12, fontweight='bold')
        ax.set_title('Power Improvement Distribution by Model', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Zero Improvement')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / 'graph4_power_improvement.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Graph 4 saved: graph4_power_improvement.png")
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
    
    # Analyze
    print(f"\n[Analyzing Data]")
    results_df = analyze_folder_with_models(
        str(data_folder),
        window_size=512,
        stride=256,
        ml_models=ml_models
    )
    
    if results_df is not None:
        # Save results
        output_path = workspace_root / 'model_results'
        output_path.mkdir(exist_ok=True)
        
        csv_output = output_path / 'veh_power_analysis.csv'
        results_df.to_csv(csv_output, index=False)
        print(f"\n✓ Results saved: {csv_output}")
        
        # Generate visualizations
        print(f"\n[Generating Visualizations]")
        generate_visualizations(results_df, ml_models, output_path)
        
        print(f"\n{'='*80}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*80}")
