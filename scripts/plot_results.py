import matplotlib.pyplot as plt
import numpy as np

gpu_data = {'Q4_0': 88.37, 'Q4_K_M': 91.54, 'Q5_K_M': 77.61, 'Q8_0': 76.39}
rpi_data = {'Q4_0': 2.28, 'Q4_K_M': 2.23, 'Q5_K_M': 2.05, 'Q8_0': 1.69}
sizes    = {'Q4_0': 1.61, 'Q4_K_M': 1.64, 'Q5_K_M': 1.75, 'Q8_0': 2.09}

quants = list(gpu_data.keys())
x = np.arange(len(quants))
width = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Llama 3.2-1B Mental Health Model\nRTX 3050 Laptop GPU vs Raspberry Pi 4 CPU', fontsize=13, fontweight='bold')

ax1 = axes[0]
b1 = ax1.bar(x - width/2, list(gpu_data.values()), width, label='RTX 3050 GPU', color='#6C5CE7', alpha=0.85)
b2 = ax1.bar(x + width/2, list(rpi_data.values()), width, label='Raspberry Pi 4 CPU', color='#E17055', alpha=0.85)
ax1.set_xlabel('Quantization Level')
ax1.set_ylabel('Generation Speed (tokens/sec)')
ax1.set_title('Inference Speed Comparison')
ax1.set_xticks(x)
ax1.set_xticklabels(quants)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
for bar in b1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
for bar in b2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=9)

speedups = [gpu_data[q]/rpi_data[q] for q in quants]
b3 = axes[1].bar(quants, speedups, color='#00B894', alpha=0.85)
axes[1].set_xlabel('Quantization Level')
axes[1].set_ylabel('Speedup Factor (x times faster)')
axes[1].set_title('GPU Speedup over Raspberry Pi 4')
axes[1].grid(axis='y', alpha=0.3)
for bar, s in zip(b3, speedups):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{s:.1f}x', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('results/benchmark_comparison.png', dpi=150, bbox_inches='tight')
print('Saved: results/benchmark_comparison.png')

print('\nSUMMARY:')
print(f'{"Quant":<10} {"GPU":>10} {"RPi4":>10} {"Speedup":>10} {"Size GB":>10}')
print('-' * 52)
for q in quants:
    print(f'{q:<10} {gpu_data[q]:>9.2f} {rpi_data[q]:>9.2f} {gpu_data[q]/rpi_data[q]:>9.1f}x {sizes[q]:>9}')
