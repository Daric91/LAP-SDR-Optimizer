import matplotlib.pyplot as plt
import numpy as np

# 把你终端里的数据手动提出来（步数，Baseline，LAP-SDR）
steps = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950]
baseline = [7.3517, 6.9495, 7.6083, 6.3392, 7.2594, 6.4959, 6.9750, 6.7336, 6.3531, 6.8079, 6.7520, 7.0745, 6.8661, 6.7959, 6.5411, 6.6531, 6.9898, 6.8179, 7.1559]
lapsdr = [7.3464, 6.9456, 7.6006, 6.3325, 7.2541, 6.4955, 6.9721, 6.7308, 6.3528, 6.8026, 6.7482, 7.0723, 6.8608, 6.7957, 6.5396, 6.6529, 6.9898, 6.8329, 7.1536]

# 计算优势 (Baseline - LAP-SDR)
advantages = [b - l for b, l in zip(baseline, lapsdr)]

plt.figure(figsize=(12, 5))
colors = ['green' if a >= 0 else 'red' for a in advantages]
plt.bar(steps, advantages, width=20, color=colors)

plt.axhline(0, color='black', linewidth=1.5)
plt.title("LAP-SDR Advantage over Baseline (Green = LAP-SDR Wins)")
plt.xlabel("Training Steps")
plt.ylabel("Loss Advantage (Baseline - LAP-SDR)")
plt.grid(axis='y', alpha=0.3)
plt.savefig("advantage_chart.png")
plt.show()
