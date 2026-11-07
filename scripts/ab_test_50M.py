import torch
import torch.nn.functional as F
import numpy as np
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from Freedream_1B import GPT, CONFIG, DataPrefetcher


# ======================================================
# 1. 基础训练步 (Baseline)
# ======================================================
def train_step_baseline(model, optimizer, x_step, y_step, base_lr, accumulation):
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr
    optimizer.zero_grad(set_to_none=True)

    loss_step_accum = 0.0

    for i in range(accumulation):
        x = x_step[i:i + 1]
        y = y_step[i:i + 1]
        loss = model(x, labels=y)

        loss_step_accum += loss.item() / accumulation
        scaled_loss = loss / accumulation
        scaled_loss.backward()
        del x, y, loss, scaled_loss

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    return loss_step_accum, grad_norm


# ======================================================
# 2. LAP-SDR (你的平滑防过拟合算法)
# ======================================================
def train_step_lap_sdr(model, optimizer, x_step, y_step, base_lr, accumulation, tau=1.0):
    for param_group in optimizer.param_groups:
        param_group['lr'] = base_lr
    optimizer.zero_grad(set_to_none=True)

    micro_batches = []
    probe_losses = []

    with torch.no_grad():
        for i in range(accumulation):
            x = x_step[i:i + 1]
            y = y_step[i:i + 1]
            micro_batches.append((x, y))
            loss = model(x, labels=y)
            probe_losses.append(loss.item())

    loss_tensor = torch.tensor(probe_losses, device=x_step.device, dtype=torch.float32)

    # 🚨 极其温和的 Softmax 分配 (tau = 2.0)
    weights = F.softmax(loss_tensor / tau, dim=0)
    scaled_weights = weights * accumulation

    # 🚨 极其温和的截断 (0.8 到 1.2)
    clamped_weights = torch.clamp(scaled_weights, min=0.6, max=1.5)

    # 重新归一化，保证总和等于 accumulation
    final_weights = clamped_weights * (accumulation / clamped_weights.sum())

    loss_step_accum = 0.0
    for i in range(accumulation):
        x, y = micro_batches[i]
        loss = model(x, labels=y)
        # 带着极其温和的权重进行反向传播
        weighted_loss = loss * final_weights[i] / accumulation
        weighted_loss.backward()
        loss_step_accum += loss.item() / accumulation
        del x, y, loss, weighted_loss

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    return loss_step_accum, grad_norm, final_weights


# ======================================================
# 3. 核心 A/B 跑批逻辑
# ======================================================
def run_100b_ab_test():
    device = torch.device("cuda")
    tokens_file = "/home/darren/Freedream_1B/tokens_pretrain_v2.bin"

    tokens = np.memmap(tokens_file, dtype=np.uint16, mode="r")
    print(f"Loaded tokens: {len(tokens):,}")

    CONFIG["layers"] = 12
    CONFIG["dim"] = 1024
    CONFIG["heads"] = 8
    CONFIG["ffn_dim"] = 2816

    model_A = GPT(32001).to(device).to(torch.bfloat16)
    model_B = GPT(32001).to(device).to(torch.bfloat16)
    model_B.load_state_dict(model_A.state_dict())

    opt_A = torch.optim.AdamW(model_A.parameters(), lr=1e-4)
    opt_B = torch.optim.AdamW(model_B.parameters(), lr=1e-4)

    context = 4096
    random_starts = np.random.randint(0, len(tokens) - context - 1, size=1000 * 16)
    prefetcher = DataPrefetcher(tokens, context, 1, 16, random_starts, device)

    def get_lr(step):
        max_lr, min_lr = 1.0e-5, 2.0e-6
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * step / 1000))

    baseline_losses = []
    lapsdr_losses = []
    total_steps = 1000

    model_A = torch.compile(model_A)
    model_B = torch.compile(model_B)

    print("⚔️ 启动 A/B 平滑度对比测试...")
    start_time = time.time()

    model_A.train()
    model_B.train()

    for step in tqdm(range(total_steps)):
        current_lr = get_lr(step)
        x_step, y_step = prefetcher.next_step()

        loss_A, _ = train_step_baseline(model_A, opt_A, x_step, y_step, current_lr, 16)
        baseline_losses.append(loss_A)

        loss_B, _, weights_B = train_step_lap_sdr(model_B, opt_B, x_step, y_step, current_lr, 16)
        lapsdr_losses.append(loss_B)

        if step % 50 == 0:
            w_str = ", ".join([f"{w:.2f}" for w in weights_B.tolist()])
            print(f"\nStep {step}")
            print(f"  Overall -> Baseline: {loss_A:.4f} | LAP-SDR: {loss_B:.4f}")
            print(f"  Weights: [{w_str}]")

    print(f"\n🏁 完成！耗时 {time.time() - start_time:.1f} 秒")

    # ==========================================
    # 🚨 统计与画图：证明平滑度与防过拟合
    # ==========================================
    def smooth(y, box_pts):
        box = np.ones(box_pts) / box_pts
        return np.convolve(y, box, mode='valid')

    # 窗口调成 10
    def rolling_std(y, window=10):
        stds = []
        for i in range(len(y) - window + 1):
            stds.append(np.std(y[i:i+window]))
        return np.array(stds)

    plt.figure(figsize=(12, 5))

    # 画图 1：整体 Loss 平滑曲线
    plt.subplot(1, 2, 1)
    plt.plot(smooth(baseline_losses, 10), label="Baseline (Overall)", color="blue", alpha=0.7)
    plt.plot(smooth(lapsdr_losses, 10), label="LAP-SDR (Overall)", color="red", alpha=0.7)
    plt.title("Smoothed Training Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 画图 2：Loss 震荡方差（平滑度）
    plt.subplot(1, 2, 2)
    plt.plot(rolling_std(baseline_losses), label="Baseline Variance", color="blue")
    plt.plot(rolling_std(lapsdr_losses), label="LAP-SDR Variance", color="red")
    plt.title("Loss Fluctuation Variance (Lower is Smoother)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("ab_test_result.png")
    print("📊 图表已保存为 ab_test_result.png！立刻打开看！")


if __name__ == "__main__":
    run_100b_ab_test()
