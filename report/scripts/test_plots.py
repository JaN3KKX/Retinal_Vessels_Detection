import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.dirname(SCRIPT_DIR)
CSV_FILE = os.path.join(REPORT_DIR, "data", "report_01_test_comparison.csv")
OUTPUT_DIR = os.path.join(REPORT_DIR, "plots", "test_plots")

def generate_individual_metric_plots(csv_path, output_dir):
    if not os.path.exists(csv_path):
        print(f"Error: File not found {csv_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    
    df_images = df[df['Image'] != 'Mean']
    
    image_names = df_images['Image'].tolist()
    
    metrics = {
        'Acc': 'Accuracy',
        'Sens': 'Sensitivity',
        'Spec': 'Specificity',
        'F1': 'F-score',
        'Gm': 'G-mean'
    }
    
    colors = ['#4c72b0', '#dd8452', '#55a868']

    for short_name, full_name in metrics.items():
        baseline_vals = df_images[f'BaselineNew_{short_name}'].tolist()
        ml_vals = df_images[f'MLv3_{short_name}'].tolist()
        dl_vals = df_images[f'DLNew_{short_name}'].tolist()

        x = np.arange(len(image_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(12, 6))

        rects1 = ax.bar(x - width, baseline_vals, width, label='Baseline (Improved)', color=colors[0])
        rects2 = ax.bar(x, ml_vals, width, label='ML (Random Forest)', color=colors[1])
        rects3 = ax.bar(x + width, dl_vals, width, label='DL (U-Net)', color=colors[2])

        ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'Comparison of Models - {full_name}', fontsize=15, pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(image_names, fontsize=11)
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
                            ha='center', va='bottom', fontsize=9, rotation=0)

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)

        fig.tight_layout()
        
        output_filename = f"plot_{short_name}.png"
        full_output_path = os.path.join(output_dir, output_filename)
        
        plt.savefig(full_output_path, dpi=300, bbox_inches='tight')
        
        plt.close(fig)

if __name__ == "__main__":
    generate_individual_metric_plots(CSV_FILE, OUTPUT_DIR)
    print("\nPlots saved to report/plots/test_plots/!")