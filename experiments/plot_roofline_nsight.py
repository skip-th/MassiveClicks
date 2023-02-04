# Plot the roofline model using the performance on the x-axis and the
# arithmetic intensity on the y-axis for the Titan X GPU.

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.ticker as mticker
from matplotlib.ticker import StrMethodFormatter, NullFormatter
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Define the roofline model.
def roofline_model(x, bandwidth):
    return x * bandwidth

# Create the figure.
fig, ax = plt.subplots(1, 2, figsize=(13, 5.75))

# Plot the roofline model
peak_flops_sp = [9031, 30256] # GFLOPS
peak_flops_dp = [141, 469] # GFLOPS
bandwidth  = [415.6, 727.6] # GB/s
x_min = 0.001
x_max = 100
y_min = 0.1
y_max = 100000
for i in [0, 1]:
    ax[i].plot([x_min, peak_flops_sp[i]/bandwidth[i]], [peak_flops_sp[i], peak_flops_sp[i]], color='black', linestyle='--', linewidth=2)
    ax[i].plot([peak_flops_sp[i]/bandwidth[i], x_max], [peak_flops_sp[i], peak_flops_sp[i]], color='black', linestyle='-', linewidth=3)
    # ax[i].plot([x_min, peak_flops_dp[i]/bandwidth[i]], [peak_flops_dp[i], peak_flops_dp[i]], color='black', linestyle='--', linewidth=2)
    # ax[i].plot([peak_flops_dp[i]/bandwidth[i], x_max], [peak_flops_dp[i], peak_flops_dp[i]], color='black', linestyle='-', linewidth=3)
    ax[i].plot([x_min, peak_flops_sp[i]/bandwidth[i]], [x_min*bandwidth[i], peak_flops_sp[i]/bandwidth[i]*bandwidth[i]] , color='black', linestyle='-', linewidth=3)
    ax[i].plot([peak_flops_sp[i]/bandwidth[i], x_max], [peak_flops_sp[i]/bandwidth[i]*bandwidth[i], x_max*bandwidth[i]], color='black', linestyle='--', linewidth=2)
    ax[i].axvline(x=peak_flops_sp[i]/bandwidth[i], color='lightgray', linestyle='--', linewidth=1.5)
    ax[i].text(x_min + x_min/10, peak_flops_sp[i] + (2000 if i==0 else 3500), f'Peak SP FLOPS = {peak_flops_sp[i]} GFLOPS', fontsize=12, color='black', rotation=0)
    # ax[i].text(x_min + x_min/10, peak_flops_dp[i] + (20 if i==0 else 35), f'Peak DP FLOPS = {peak_flops_dp[i]} GFLOPS', fontsize=12, color='black', rotation=0)
    ax[i].text(x_min + x_min/10, roofline_model(x_min, bandwidth[i] + (250 if i==0 else 450)), f'Bandwidth = {bandwidth[i]} GB/s', fontsize=12, color='black', rotation=29)
    ax[i].text(peak_flops_sp[i]/bandwidth[i] + 2, y_min * 1.35, 'Compute-bound', fontsize=11, color='lightgray', rotation=90)
    ax[i].text(peak_flops_sp[i]/bandwidth[i] - (6 if i == 0 else 12), y_min * 1.35, 'Memory-bound', fontsize=11, color='lightgray', rotation=90)

# Plot the measurements.
# A4000
ax[0].plot(0.92, 249.022, color='red',   marker='s', label='PBM training')
ax[0].plot(0.04,   3.381, color='red',   marker='^', label='PBM update')
ax[0].plot(2.83, 418.555, color='blue',  marker='s', label='CCM training')
ax[0].plot(0.01,   1.013, color='blue',  marker='^', label='CCM update')
ax[0].plot(2.28, 379.957, color='green', marker='s', label='DBN training')
ax[0].plot(0.003,  0.229, color='green', marker='^', label='DBN update')
ax[0].plot(0.74, 187.162, color='gold',  marker='s', label='UBM training')
ax[0].plot(0.18,  15.991, color='gold',  marker='^', label='UBM update')
# A6000
ax[1].plot(0.15,  101.995, color='red',   marker='s', label='PBM training')
ax[1].plot(0.04,    8.032, color='red',   marker='^', label='PBM update')
ax[1].plot(2.69, 1128.415, color='blue',  marker='s', label='CCM training')
ax[1].plot(0.01,    2.323, color='blue',  marker='^', label='CCM update')
ax[1].plot(2.14,  977.909, color='green', marker='s', label='DBN training')
ax[1].plot(0.008,   0.486, color='green', marker='^', label='DBN update')
ax[1].plot(0.12,   74.282, color='gold',  marker='s', label='UBM training')
ax[1].plot(0.16,   39.005, color='gold',  marker='^', label='UBM update')

# Set the x-axis and y-axis limits and labels.
for i in [0, 1]:
    ax[i].set_xlabel('Kernel Arithmetic Intensity (FLOPS/Byte)')
    ax[i].set_ylabel('Performance (GFLOPS)')
    ax[i].set_xscale('log')
    ax[i].set_yscale('log')
    ax[i].set_xlim(x_min, x_max)
    ax[i].set_ylim(y_min, y_max)
    ax[i].grid(True)
    # ax[i].yaxis.set_major_formatter(StrMethodFormatter('{x:.0f}'))

# Show the plot.
ax[0].set_title('A4000')
ax[1].set_title('A6000')
fig.suptitle(f'Roofline Model (Nsight Compute)', fontsize=16)
plt.tight_layout()
fig.subplots_adjust(bottom=0.225)
plt.legend(loc='lower center', bbox_to_anchor=(-0.075, -0.325), fancybox=True, ncol=4)
plt.savefig(f'figures/A4000_A6000_roofline_nsight_sp.png', format='png', dpi=1000)
plt.show()