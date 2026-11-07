# LAP-SDR-Optimizer

**A stateless dynamic routing optimizer that mitigates gradient shocks and beats AdamW on heterogeneous LLM training (89.4% win rate on a single GPU).**

## Overview
Training Large Language Models (LLMs) on heterogeneous datasets (e.g., mixing natural text, code, and math) often suffers from severe **gradient shocks** when the data distribution shifts. Standard optimizers like AdamW treat every micro-batch equally, wasting compute on trivial data and failing to adapt quickly to complex data.

**LAP-SDR (Look-Ahead Probe with Softmax Dynamic Routing)** solves this by dynamically reallocating compute to the hardest samples without breaking the global learning rate.

## Core Mechanisms
1. **Look-Ahead Probe**: Uses a no-gradient forward pass to estimate the difficulty (loss) of each micro-batch in real-time.
2. **Softmax Routing**: Dynamically assigns higher gradient weights to hard data and lower weights to easy data (e.g., $w_{\max}=1.5, w_{\min}=0.6$).
3. **Zero-Sum Renormalization**: Strictly preserves the global learning rate schedule by keeping the sum of weights equal to the accumulation steps.

## Empirical Results
* **Macro Level**: Achieves an **89.4% win rate** over the AdamW baseline in a 50M parameter model A/B test.
* **Micro Level (Extreme Imbalance)**: Under an 80/20 (Easy/Hard) data distribution, LAP-SDR demonstrates significantly **faster adaptation speed** on complex data, achieving a lower hard-loss than AdamW.

