"""
Model Improvement Metrics Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

def plot_model_improvements():
    """Generate comprehensive model improvement visualization"""
    
    # Sample data from analysis
    models = ['RF', 'SVM', 'kNN']
    pre_mapping_mean = 22656552794.6860  # Pre-mapping baseline
    
    # Post-mapping data (from your analysis output)
    post_mapping_data = {
        'RF': {
            'mean': 116125106019.6821,
            'median': 98174770424.6810,
            'std': 23280298034.7395,
            'min': 0,
            'max': 0,
        },
        'SVM': {
            'mean': 114610842208.5045,
            'median': 98174770424.6810,
            'std': 21173212775.3324,
            'min': 0,
            'max': 0,
        },
        'kNN': {
            'mean': 136122501952.0159,
            'median': 147262155637.0215,
            'std': 20560267200.2598,
            'min': 0,
            'max': 0,
        }
    }
    
    # Calculate improvements
    metrics = {}
    for model in models:
        post = post_mapping_data[model]
        metrics[model] = {
            'mean': post['mean'],
            'median': post['median'],
            'std': post['std'],
            'mean_improvement': post['mean'] - pre_mapping_mean,
            'mean_improvement_pct': ((post['mean'] - pre_mapping_mean) / pre_mapping_mean) * 100,
            'median_improvement': post['median'] - pre_mapping_mean,
            'median_improvement_pct': ((post['median'] - pre_mapping_mean) / pre_mapping_mean) * 100,
        }
    
    # Create comprehensive figure
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    colors = {'RF': '#3498db', 'SVM': '#e74c3c', 'kNN': '#f39c12'}
    sorted_models = sorted(models)
    
    # (1) Mean Power Comparison - WITH PRE-MAPPING BASELINE
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(sorted_models) + 1)  # Include pre-mapping
    width = 0.18
    
    # Pre-mapping bar
    pre_bar = ax1.bar(0, pre_mapping_mean, width*2, label='Pre-Mapping', alpha=0.8, 
                      color='#e74c3c', edgecolor='black', linewidth=1.5)
    
    # Post-mapping bars
    means = [metrics[m]['mean'] for m in sorted_models]
    for i, (model, mean_val) in enumerate(zip(sorted_models, means)):
        ax1.bar(i + 1, mean_val, width*2, label=f'Post-{model}', 
               alpha=0.8, color=colors[model], edgecolor='black', linewidth=1.5)
    
    ax1.set_ylabel('Mean Power (μW)', fontsize=12, fontweight='bold')
    ax1.set_title('(1) Mean Power: Pre-Mapping vs All Models', fontsize=12, fontweight='bold')
    ax1.set_xticks([0, 1, 2, 3])
    ax1.set_xticklabels(['Pre-Map', 'RF', 'SVM', 'kNN'], fontsize=11)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Add value labels
    ax1.text(0, pre_mapping_mean + max(means)*0.02, f'{pre_mapping_mean:.2e}', 
            ha='center', fontsize=9, fontweight='bold')
    for i, (model, v) in enumerate(zip(sorted_models, means)):
        improvement_pct = metrics[model]['mean_improvement_pct']
        ax1.text(i + 1, v + max(means)*0.02, f'{v:.2e}\n+{improvement_pct:.0f}%', 
                ha='center', fontsize=8, fontweight='bold')
    
    # (2) Mean Power Improvement (%)
    ax2 = fig.add_subplot(gs[0, 1])
    improvements_pct = [metrics[m]['mean_improvement_pct'] for m in sorted_models]
    bars = ax2.bar(sorted_models, improvements_pct, color=[colors[m] for m in sorted_models], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_ylabel('Improvement (%)', fontsize=12, fontweight='bold')
    ax2.set_title('(2) Mean Power Improvement %', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    max_pct = max(improvements_pct)
    for i, (model, v) in enumerate(zip(sorted_models, improvements_pct)):
        ax2.text(i, v + max_pct*0.02, f'{v:+.1f}%', ha='center', fontsize=11, fontweight='bold', color=colors[model])
    
    # (3) Power Improvement (%) - Same as (2) for clarity
    ax3 = fig.add_subplot(gs[0, 2])
    improvements_pct_v2 = [metrics[m]['mean_improvement_pct'] for m in sorted_models]
    bars = ax3.bar(sorted_models, improvements_pct_v2, color=[colors[m] for m in sorted_models], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.axhline(0, color='black', linestyle='-', linewidth=1)
    ax3.set_ylabel('Power Improvement (%)', fontsize=12, fontweight='bold')
    ax3.set_title('(3) Power Improvement Percentage', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    max_pct_v2 = max(improvements_pct_v2)
    for i, (model, v) in enumerate(zip(sorted_models, improvements_pct_v2)):
        ax3.text(i, v + max_pct_v2*0.02, f'{v:+.1f}%', ha='center', fontsize=11, fontweight='bold', color=colors[model])
    
    # (4) Pre vs Post-Mapping Power Comparison (Left-Right Split)
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Create synthetic pre-mapping distribution (repeating baseline)
    pre_dist = np.full(len(sorted_models), pre_mapping_mean)
    post_dists = [metrics[m]['mean'] for m in sorted_models]
    
    x_pos = np.arange(len(sorted_models))
    width = 0.35
    
    # Left: Pre-Mapping
    bars1 = ax4.bar(x_pos - width/2, pre_dist, width, label='Pre-Mapping', 
                   alpha=0.8, color='#e74c3c', edgecolor='black', linewidth=1.5)
    
    # Right: Post-Mapping
    bars2 = ax4.bar(x_pos + width/2, post_dists, width, label='Post-Mapping', 
                   alpha=0.8, color='#2ecc71', edgecolor='black', linewidth=1.5)
    
    ax4.set_ylabel('Mean Power (μW)', fontsize=12, fontweight='bold')
    ax4.set_title('(4) Pre-Mapping vs Post-Mapping Power Output', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(sorted_models, fontsize=11)
    ax4.legend(fontsize=11, loc='upper left')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    # Add value labels
    for i, (model, post_val) in enumerate(zip(sorted_models, post_dists)):
        # Pre value
        ax4.text(i - width/2, pre_dist[i] + max(post_dists)*0.02, f'{pre_dist[i]:.1e}', 
                ha='center', fontsize=8, fontweight='bold', color='#c0392b')
        # Post value
        ax4.text(i + width/2, post_val + max(post_dists)*0.02, f'{post_val:.1e}', 
                ha='center', fontsize=8, fontweight='bold', color='#27ae60')
    
    # (5) Standard Deviation - Power Consistency
    ax5 = fig.add_subplot(gs[1, 1])
    stds = [metrics[m]['std'] for m in sorted_models]
    bars = ax5.bar(sorted_models, stds, color=[colors[m] for m in sorted_models], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax5.set_ylabel('Standard Deviation (μW)', fontsize=12, fontweight='bold')
    ax5.set_title('(5) Power Variability (Lower = More Consistent)', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    ax5.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    for i, (model, v) in enumerate(zip(sorted_models, stds)):
        ax5.text(i, v + max(stds)*0.02, f'{v:.2e}', ha='center', fontsize=9, fontweight='bold')
    
    # (6) Summary Metrics Table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    # Create table data
    table_data = []
    table_data.append(['Model', 'Mean Power', 'Improvement', 'Status'])
    
    for model in sorted_models:
        mean_val = metrics[model]['mean']
        improvement_pct = metrics[model]['mean_improvement_pct']
        # Determine status
        if improvement_pct > 450:
            status = '🏆 Excellent'
        elif improvement_pct > 400:
            status = '✓ Very Good'
        else:
            status = '○ Good'
        
        table_data.append([
            model,
            f'{mean_val/1e9:.1f}B',
            f'{improvement_pct:+.1f}%',
            status
        ])
    
    table = ax6.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.15, 0.3, 0.25, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Header styling
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#34495e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Row styling
    for i in range(1, len(table_data)):
        model = table_data[i][0]
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ecf0f1')
            else:
                table[(i, j)].set_facecolor('#ffffff')
            table[(i, j)].set_text_props(weight='bold')
    
    ax6.set_title('(6) Performance Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('Model Performance Comparison - Power Improvement Analysis', 
                fontsize=15, fontweight='bold', y=0.995)
    
    # Save
    output_path = Path("/Users/seohyeon/AT_freq_tuning") / "veh_model_improvements_w512.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n{'='*100}")
    print(f"✓ Model improvement comparison graph saved: {output_path}")
    print(f"{'='*100}\n")
    
    plt.close()
    
    # Print detailed metrics
    print(f"\n{'='*100}")
    print("DETAILED MODEL PERFORMANCE METRICS")
    print(f"{'='*100}\n")
    
    print(f"{'Pre-Mapping Baseline:':<40} Mean Power = {pre_mapping_mean:.4e} μW\n")
    
    for model in sorted_models:
        print(f"{model} Model:")
        print(f"  ├─ Mean Power:                {metrics[model]['mean']:.4e} μW")
        print(f"  ├─ Mean Improvement (Abs):    {metrics[model]['mean_improvement']:+.4e} μW")
        print(f"  ├─ Mean Improvement (%):      {metrics[model]['mean_improvement_pct']:+.2f}%")
        print(f"  ├─ Median Power:              {metrics[model]['median']:.4e} μW")
        print(f"  ├─ Median Improvement (%):    {metrics[model]['median_improvement_pct']:+.2f}%")
        print(f"  └─ Standard Deviation:        {metrics[model]['std']:.4e} μW")
        print()
    
    # Ranking
    print(f"{'='*100}")
    print("MODEL RANKING")
    print(f"{'='*100}\n")
    
    rank_by_mean = sorted(models, key=lambda m: metrics[m]['mean'], reverse=True)
    rank_by_improvement = sorted(models, key=lambda m: metrics[m]['mean_improvement_pct'], reverse=True)
    
    print("By Mean Power Output:")
    for i, model in enumerate(rank_by_mean, 1):
        print(f"  {i}. {model:6s} - {metrics[model]['mean']:.4e} μW ({metrics[model]['mean_improvement_pct']:+.2f}%)")
    
    print("\nBy Power Improvement %:")
    for i, model in enumerate(rank_by_improvement, 1):
        print(f"  {i}. {model:6s} - {metrics[model]['mean_improvement_pct']:+.2f}% improvement")
    
    print(f"\n{'='*100}\n")


if __name__ == "__main__":
    plot_model_improvements()
