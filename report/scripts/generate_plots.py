import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.dirname(SCRIPT_DIR)
CSV_FILE = os.path.join(REPORT_DIR, "data", "report_01_final_comparison.csv")
OUTPUT_DIR = os.path.join(REPORT_DIR, "plots", "training_plots")

def generate_comparison_plot(csv_path, output_dir):
    if not os.path.exists(csv_path):
        print(f"Błąd: Nie znaleziono pliku {csv_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    csv_filename = os.path.basename(csv_path)
    output_filename = os.path.join(output_dir, csv_filename.replace(".csv", "_plot.png"))

    df = pd.read_csv(csv_path)
    
    mean_row = df[df['Image'] == 'Mean'].iloc[0]
    
    metrics = ['Sens', 'Spec', 'F1', 'Gm']
    metric_labels = ['Sensitivity', 'Specificity', 'F-score', 'G-mean']
    
    baseline_vals = [mean_row[f'BaselineNew_{m}'] for m in metrics]
    ml_vals = [mean_row[f'MLv3_{m}'] for m in metrics]
    dl_vals = [mean_row[f'DLNew_{m}'] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width, baseline_vals, width, label='Baseline', color='#4c72b0')
    rects2 = ax.bar(x, ml_vals, width, label='Machine Learning (Random Forest)', color='#dd8452')
    rects3 = ax.bar(x + width, dl_vals, width, label='Deep Learning (U-Net)', color='#55a868')
    
    ax.set_ylabel('Metric Value', fontsize=12)
    ax.set_title('Average Metrics Comparison', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    fig.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")
    
    plt.close(fig)

if __name__ == "__main__":
    generate_comparison_plot(CSV_FILE, OUTPUT_DIR)