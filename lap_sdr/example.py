import torch
import torch.nn as nn
import torch.nn.functional as F
from lap_sdr import LAPSDRRouter

# ==========================================
# 1. Create a Dummy Model (Just for demo)
# ==========================================
class DummyGPT(nn.Module):
    def __init__(self, vocab_size=10000, dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.lm_head = nn.Linear(dim, vocab_size)
        
    def forward(self, x, labels=None):
        x = self.embed(x)
        logits = self.lm_head(x)
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            return loss
        return logits

# ==========================================
# 2. Setup
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DummyGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Initialize LAP-SDR Router
router = LAPSDRRouter(
    accumulation=16,
    tau=1.0,
    w_min=0.6,
    w_max=1.4,
    probe_len=256 
)

# ==========================================
# 3. Mock Data Prefetcher (Simulating your dataloader)
# ==========================================
# Shape: [accumulation, batch_size, seq_len]
x_step = torch.randint(0, 10000, (16, 1, 512)).to(device)
y_step = torch.randint(0, 10000, (16, 1, 512)).to(device)

# ==========================================
# 4. Training Step
# ==========================================
model.train()
optimizer.zero_grad()

print("Running one LAP-SDR training step...")

# One-step call
avg_loss, final_weights = router.train_step(model, x_step, y_step)

# Gradient clipping & step
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()

print(f"Step completed successfully!")
print(f"Average Loss: {avg_loss:.4f}")
print(f"Dynamic Weights (first 4): {[round(w, 2) for w in final_weights[:4].tolist()]}")
