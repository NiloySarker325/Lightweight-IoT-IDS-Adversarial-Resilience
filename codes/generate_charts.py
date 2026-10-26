import matplotlib.pyplot as plt
import numpy as np

# Set clean style for academic publication
plt.style.use('seaborn-v0_8-whitegrid')

# Data extracted from terminal results
mlp_labels = ['Clean\nBaseline', 'FGSM\nAttack', 'Adversarial\nTraining']
mlp_acc = [97.02, 86.60, 98.18]

rf_labels = ['Clean\nBaseline', 'HopSkipJump\nAttack']
rf_acc = [100.00, 52.00]

# Setup a side-by-side dual chart figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# Plot 1: PyTorch MLP (Baseline, Attack, Defense)
bars1 = ax1.bar(mlp_labels, mlp_acc, color=["#0d0e0d", "#7e6666", "#a3d1f2"], edgecolor='black')
ax1.set_title('PyTorch MLP: Vulnerability & Recovery', fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontweight='bold')
ax1.set_ylim(0, 115)

# Add percentage values above MLP bars
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval}%', ha='center', va='bottom', fontweight='bold')

# Plot 2: Random Forest (Baseline, Attack)
bars2 = ax2.bar(rf_labels, rf_acc, color=['#0d0e0d', '#7e6666'], edgecolor='black', width=0.6)
ax2.set_title('Random Forest: Black-Box Degradation', fontweight='bold')
ax2.set_ylim(0, 115)

# Add percentage values above RF bars
for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval + 2, f'{yval}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()

# Save as a high-resolution 300 DPI image for IEEE insertion
plt.savefig('iot_research_results.png', dpi=300, bbox_inches='tight')
print("Chart successfully saved as 'iot_research_results.png'")
plt.show()