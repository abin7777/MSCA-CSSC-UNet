import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def bland_altman_plot(test_trues, test_preds):
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    differences = test_preds - test_trues
    means = (test_preds + test_trues) / 2
    mean_diff = np.mean(differences)
    sd_diff = np.std(differences)

    plt.figure(figsize=(10, 6))
    plt.scatter(means, differences, alpha=0.7)
    plt.axhline(mean_diff, color='red', linestyle='-', label=f'ME: {mean_diff:.2f}')
    plt.axhline(mean_diff + 1.96*sd_diff, color='gray', linestyle='--', 
                label=f'+1.96SD: {mean_diff + 1.96*sd_diff:.2f}')
    plt.axhline(mean_diff - 1.96*sd_diff, color='gray', linestyle='--', 
                label=f'-1.96SD: {mean_diff - 1.96*sd_diff:.2f}')

    plt.xlabel('Average of True and Estimated Value(mmHg)')
    plt.ylabel('Error in Prediction(mmHg)')
    plt.title(f'Bland-Altman Plot')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

def linear_regression_plot(test_trues, test_preds):

    slope, intercept = np.polyfit(test_trues, test_preds, 1)  
    regression_line = slope * test_trues + intercept

    plt.figure(figsize=(10, 8))
    plt.scatter(test_trues, test_preds, s=30)
    plt.plot(test_trues, regression_line, color="red")
    plt.xlabel("True Value(mmHg)", fontsize=20)
    plt.ylabel("Estimated Value(mmHg)", fontsize=20)
    plt.title("Linear Regression Plot for ABP Prediction", fontsize=20)

    plt.text(0.05, 0.95, 'PCC: 0.896\np-value: 0.000', 
            transform=plt.gca().transAxes,  
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
            verticalalignment='top')

    plt.tight_layout()
    plt.show()

def plot_absolute_error_distribution(test_trues, test_preds):
    MAE_array = np.abs(test_preds - test_trues)
    plt.figure(figsize=(8, 5))
    plt.hist(MAE_array, bins=25, color='skyblue', edgecolor='black', alpha=0.7)
    plt.xlabel("Absolute Error (mmHg)", fontsize=15)
    plt.ylabel("Number of Samples", fontsize=15)
    plt.title("Absolute Errors in ABP Prediction", fontsize=15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('/gjh/Distribution-of-Absolute-Errors_t.pdf')

def plot_within_vs_outside_comparison(test_trues, test_preds):
    mean_val = (test_trues + test_preds) / 2  
    diff_val = test_preds - test_trues        

    mean_diff = np.mean(diff_val)
    std_diff = np.std(diff_val)

    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff

    within_mask = (diff_val <= upper_limit) & (diff_val >= lower_limit)
    outlier_mask = (diff_val > upper_limit) | (diff_val < lower_limit)

    within_mean_vals = mean_val[within_mask]
    outlier_mean_vals = mean_val[outlier_mask]

    bins = np.linspace(mean_val.min(), mean_val.max(), 20)  
    bin_centers = (bins[:-1] + bins[1:]) / 2

    outlier_counts, _ = np.histogram(outlier_mean_vals, bins=bins)
    within_counts, _ = np.histogram(within_mean_vals, bins=bins)

    plt.figure(figsize=(16, 6))

    bar_width = 2.0 
    bars1 = plt.bar(bin_centers - bar_width/2, within_counts, bar_width, 
                    label='Inliers.Total number:314649', color='lightgreen', alpha=0.8, 
                    edgecolor='darkgreen', linewidth=1)

    bars2 = plt.bar(bin_centers + bar_width/2, outlier_counts, bar_width, 
                    label='Outliers.Total number:17997', color='red', alpha=0.8, 
                    edgecolor='darkred', linewidth=1)

    plt.xlabel('Average of True and Estimated Value (mmHg)', fontsize=15)
    plt.ylabel('Number of Points', fontsize=15)
    plt.title('Comparison of Inliers and Outliers', fontsize=15)
    plt.legend(fontsize=15)

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_ecg_with_weights(ecg_signal, weights, figsize=(15, 6)):
    fig, ax1 = plt.subplots(figsize=figsize)
    x = np.arange(len(ecg_signal))
    
    points = np.array([x, ecg_signal]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    norm = plt.Normalize(weights.min(), weights.max())
    lc = LineCollection(segments, cmap='rainbow', norm=norm)
    lc.set_array(weights[1:]) 
    lc.set_linewidth(3)
    
    line = ax1.add_collection(lc)
    ax1.set_xlim(x.min(), x.max())
    ax1.set_ylim(ecg_signal.min(), ecg_signal.max())
    ax1.set_xlabel('Time step', fontsize=15)
    ax1.set_ylabel('Amplitude', fontsize=15)
    cbar = fig.colorbar(line, ax=ax1)
    
    plt.tight_layout()
    plt.savefig('/gjh/Attention_wave_0_1.pdf', 
            dpi=300,
            bbox_inches='tight',    # 去除多余白边
            pad_inches=0.05,        # 保留少量边距
            transparent=False,
            facecolor='white')


