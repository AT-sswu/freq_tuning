"""
Energy Harvester Simulation: Comparison of Pre/Post Mapping Strategies

Power generation based on resonance frequency:
- Input frequency closer to resonance frequency -> higher power
- Simulate peak frequency input to 3 resonant harvesters (40, 50, 60Hz)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


def calculate_theoretical_power(acceleration_amplitude, frequency, 
                                resonance_freq,
                                mass=0.001,  # 1g (kg)
                                zeta_e=0.01,  # electrical damping ratio
                                zeta_m=0.01,  # mechanical damping ratio
                                scale_factor=1e6):
    """
    Theoretical power calculation based on paper Equation (5) with resonance frequency response
    
    P = (m·ζe·A²·ω²) / (4·ωn·ζT²·[(ωn² - ω²)² + (2·ζT·ωn·ω)²])
    
    At resonance (ω ≈ ωn): power is maximized
    
    Parameters:
    -----------
    acceleration_amplitude : float
        Acceleration amplitude A (m/s²)
    frequency : float
        Input frequency f (Hz) - measured actual frequency from sensor
    resonance_freq : float
        Harvester resonance frequency fn (Hz) - one of 40, 50, 60
    mass : float
        Harvester mass m (kg), default 1g
    zeta_e : float
        Electrical damping ratio ζe (dimensionless), paper recommends = ζm
    zeta_m : float
        Mechanical damping ratio ζm (dimensionless), paper measured value 0.01
    scale_factor : float
        Output unit conversion (1e6: μW, 1e3: mW, 1: W)
    
    Returns:
    --------
    power : float
        Expected output power (μW)
    """
    omega = 2 * np.pi * frequency  # input angular frequency (rad/s)
    omega_n = 2 * np.pi * resonance_freq  # resonance angular frequency (rad/s)
    zeta_T = zeta_e + zeta_m  # total damping ratio
    
    if omega_n == 0 or zeta_T == 0:
        return 0.0
    
    # frequency response function (maximum at resonance)
    freq_response_denominator = (omega_n**2 - omega**2)**2 + (2 * zeta_T * omega_n * omega)**2
    
    if freq_response_denominator == 0:
        return 0.0
    
    # power formula: power increases as input frequency approaches resonance frequency
    power = (mass * zeta_e * acceleration_amplitude**2 * omega**2) / \
            (4 * omega_n * zeta_T**2 * freq_response_denominator)
    
    return power * scale_factor  # convert to μW


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


def analyze_csv_file(csv_path, window_size=512, stride=256, resonance_freqs=[40, 50, 60]):
    """
    Single CSV file analysis:
    - Pre-mapping: use peak frequency directly
    - Post-mapping: map to nearest resonance frequency
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
                calculate_theoretical_power(
                    acceleration_amplitude=1.0,  # normalized acceleration
                    frequency=peak_freq,
                    resonance_freq=res_freq
                )
                for res_freq in resonance_freqs
            ]
            label_before = np.argmax(powers_before)
            
            # === Post-mapping: map to nearest resonance frequency ===
            res_array = np.array(resonance_freqs)
            mapped_freq = res_array[np.argmin(np.abs(res_array - peak_freq))]
            label_after = np.argmin(np.abs(res_array - peak_freq))
            
            powers_after = [
                calculate_theoretical_power(
                    acceleration_amplitude=1.0,
                    frequency=mapped_freq,  # mapped frequency
                    resonance_freq=res_freq
                )
                for res_freq in resonance_freqs
            ]
            
            results.append({
                'window_idx': win_idx,
                'peak_freq_original': peak_freq,
                'mapped_freq': mapped_freq,
                'freq_error': abs(peak_freq - mapped_freq),
                
                # Pre-mapping power
                'power_before_40Hz': powers_before[0],
                'power_before_50Hz': powers_before[1],
                'power_before_60Hz': powers_before[2],
                'label_before': resonance_freqs[label_before],
                'label_before_idx': label_before,
                'max_power_before': max(powers_before),
                
                # Post-mapping power
                'power_after_40Hz': powers_after[0],
                'power_after_50Hz': powers_after[1],
                'power_after_60Hz': powers_after[2],
                'label_after': resonance_freqs[label_after],
                'label_after_idx': label_after,
                'max_power_after': max(powers_after),
                
                # Comparison
                'label_match': label_before == label_after,
                'power_improvement': max(powers_after) - max(powers_before),
            })
        
        return pd.DataFrame(results)
    
    except Exception as e:
        print(f"ERROR processing {csv_path}: {e}")
        return None


def analyze_folder(folder_path, model_name="Model", window_size=512, stride=256):
    """
    Analyze all CSV files in folder and generate summary statistics
    """
    data_path = Path(folder_path)
    csv_files = sorted(list(data_path.glob("*.csv")))
    
    all_results = []
    
    print(f"\n{'='*80}")
    print(f"Analysis: {model_name}")
    print(f"{'='*80}")
    
    for csv_file in csv_files:
        print(f"  Processing: {csv_file.name}...", end=" ")
        df_result = analyze_csv_file(csv_file, window_size, stride)
        
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
    
    print(f"\n[Overall Statistics]")
    print(f"  Total windows: {len(combined_df)}")
    print(f"  Total files: {len(csv_files)}")
    
    # Label distribution pre/post mapping
    print(f"\n[Label Distribution - Pre-Mapping]")
    label_dist_before = combined_df['label_before'].value_counts().sort_index()
    for label, count in label_dist_before.items():
        ratio = count / len(combined_df) * 100
        print(f"  {int(label)}Hz: {count:5d} ({ratio:5.1f}%)")
    
    print(f"\n[Label Distribution - Post-Mapping]")
    label_dist_after = combined_df['label_after'].value_counts().sort_index()
    for label, count in label_dist_after.items():
        ratio = count / len(combined_df) * 100
        print(f"  {int(label)}Hz: {count:5d} ({ratio:5.1f}%)")
    
    # Mapping consistency
    match_count = combined_df['label_match'].sum()
    match_ratio = match_count / len(combined_df) * 100
    print(f"\n[Label Changes]")
    print(f"  Mapping unchanged: {match_count:5d} ({match_ratio:5.1f}%)")
    print(f"  Mapping changed: {len(combined_df) - match_count:5d} ({100-match_ratio:5.1f}%)")
    
    # Power comparison
    avg_power_before = combined_df['max_power_before'].mean()
    avg_power_after = combined_df['max_power_after'].mean()
    improvement = avg_power_after - avg_power_before
    improvement_ratio = (improvement / avg_power_before * 100) if avg_power_before > 0 else 0
    
    print(f"\n[Power Comparison]")
    print(f"  Average max power (pre-mapping): {avg_power_before:12.2f} μW")
    print(f"  Average max power (post-mapping): {avg_power_after:12.2f} μW")
    print(f"  Power improvement: {improvement:12.2f} μW ({improvement_ratio:+.1f}%)")
    
    # Frequency mapping error
    avg_freq_error = combined_df['freq_error'].mean()
    max_freq_error = combined_df['freq_error'].max()
    print(f"\n[Frequency Mapping Error]")
    print(f"  Average error: {avg_freq_error:.2f} Hz")
    print(f"  Maximum error: {max_freq_error:.2f} Hz")
    
    # Detailed comparison - first 5 windows
    print(f"\n[Detailed Comparison - First 5 Windows]")
    print(f"{'idx':<4} {'peak(Hz)':<10} {'mapped(Hz)':<10} {'error':<6} {'pre':<8} {'post':<8} {'change':<8}")
    print(f"{'-'*70}")
    
    for idx, row in combined_df.head(5).iterrows():
        change = "●" if row['label_match'] else "○"
        print(f"{idx:<4} {row['peak_freq_original']:<10.2f} {row['mapped_freq']:<10.0f} "
              f"{row['freq_error']:<6.2f} {row['label_before']:<8.0f} {row['label_after']:<8.0f} {change:<8}")
    
    return combined_df


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python compare_mapping_strategy.py <folder_path> [model_name] [window_size] [stride]")
        print("\nExamples:")
        print("  python compare_mapping_strategy.py /Users/seohyeon/AT_freq_tuning/data_v3 'Raw Data'")
        print("  python compare_mapping_strategy.py /Users/seohyeon/AT_freq_tuning/freq_tuning_svm/svm_512 'SVM 512' 512 256")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "Model"
    window_size = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    stride = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    
    results_df = analyze_folder(folder_path, model_name, window_size, stride)
    
    if results_df is not None:
        # Save results
        output_path = Path(folder_path).parent / f"{model_name.replace(' ', '_')}_comparison.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\n✓ Results saved: {output_path}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Energy Harvester Pre/Post Mapping Comparison: {model_name}', fontsize=16)
        
        # 1. Label distribution comparison
        ax = axes[0, 0]
        labels = ['40Hz', '50Hz', '60Hz']
        before = [
            len(results_df[results_df['label_before'] == 40]),
            len(results_df[results_df['label_before'] == 50]),
            len(results_df[results_df['label_before'] == 60]),
        ]
        after = [
            len(results_df[results_df['label_after'] == 40]),
            len(results_df[results_df['label_after'] == 50]),
            len(results_df[results_df['label_after'] == 60]),
        ]
        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, before, width, label='Pre-Mapping', alpha=0.8)
        ax.bar(x + width/2, after, width, label='Post-Mapping', alpha=0.8)
        ax.set_ylabel('Sample Count')
        ax.set_title('Label Distribution Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 2. Power comparison
        ax = axes[0, 1]
        ax.scatter(results_df['max_power_before'], results_df['max_power_after'], alpha=0.5, s=10)
        min_val = min(results_df['max_power_before'].min(), results_df['max_power_after'].min())
        max_val = max(results_df['max_power_before'].max(), results_df['max_power_after'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Equal')
        ax.set_xlabel('Max Power Pre-Mapping (μW)')
        ax.set_ylabel('Max Power Post-Mapping (μW)')
        ax.set_title('Power Improvement')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 3. Frequency mapping error
        ax = axes[1, 0]
        ax.hist(results_df['freq_error'], bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(results_df['freq_error'].mean(), color='r', linestyle='--', 
                   label=f"Mean: {results_df['freq_error'].mean():.2f} Hz")
        ax.set_xlabel('Frequency Mapping Error (Hz)')
        ax.set_ylabel('Frequency')
        ax.set_title('Frequency Mapping Error Distribution')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Peak frequency distribution
        ax = axes[1, 1]
        ax.hist(results_df['peak_freq_original'], bins=50, alpha=0.7, label='Original Peak', edgecolor='black')
        ax.hist(results_df['mapped_freq'], bins=50, alpha=0.7, label='Post-Mapping', edgecolor='black')
        ax.axvline(40, color='red', linestyle='--', alpha=0.5, label='Resonance Freq')
        ax.axvline(50, color='green', linestyle='--', alpha=0.5)
        ax.axvline(60, color='blue', linestyle='--', alpha=0.5)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Frequency')
        ax.set_title('Peak Frequency Distribution')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plot_path = Path(folder_path).parent / f"{model_name.replace(' ', '_')}_comparison.png"
        plt.savefig(plot_path, dpi=150)
        print(f"✓ Graph saved: {plot_path}")
        plt.close()
