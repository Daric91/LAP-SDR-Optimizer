# ==========================================================
# FREEDREAM ~977M GPT V2 — Core Architecture
# ==========================================================

import os
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

# ==========================================================
# DEVICE & CONFIG
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "dim": 1792,
    "layers": 24,
    "heads": 16,
    "context": 4096,
    "ffn_dim": 4736,
    "vocab_size": 32001,
    "dropout": 0.0,
    "rope_theta": 10000.0,
}

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_FILE = os.path.join(PROJECT_ROOT, "tokenizer_v2.json")

# ==========================================================
# TOKENIZER
# ==========================================================
class BPETokenizer:
    def __init__(self):
        self.tokenizer = None

    def load(self, path=TOKENIZER_FILE):
        self.tokenizer = Tokenizer.from_file(str(path))
        self.tokenizer.decoder = ByteLevelDecoder()

    def save(self, path=TOKENIZER_FILE):
        self.tokenizer.save(str(path))

    def encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)

    def size(self):
        return self.tokenizer.get_vocab_size()

    def token_id(self, token):
        return self.tokenizer.token_to_id(token)

tokenizer = BPETokenizer()

# ==========================================================
# ROPE
# ==========================================================
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_position_embeddings)

    def _build_cache(self, seq_len):
        positions = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(positions, self.inv_freq)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, q, k):
        seq_len = q.shape[-2]
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        cos = self.cos_cached[:seq_len][None, None, :, :]
        sin = self.sin_cached[:seq_len][None, None, :, :]
        return apply_rotary(q, cos, sin), apply_rotary(k, cos, sin)

def apply_rotary(x, cos, sin):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    r1 = x1 * cos - x2 * sin
    r2 = x1 * sin + x2 * cos
    return torch.stack([r1, r2], dim=-1).flatten(-2)

# ==========================================================
# ATTENTION & MLP
# ==========================================================
class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        d = CONFIG["dim"]
        heads = CONFIG["heads"]
        self.heads = heads
        self.head_dim = d // heads
        self.qkv = nn.Linear(d, d * 3, bias=True)
        self.proj = nn.Linear(d, d, bias=True)
        self.rope = RotaryEmbedding(self.head_dim, CONFIG["context"], CONFIG["rope_theta"])
        self.dropout = CONFIG["dropout"]

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        d = CONFIG["dim"]
        hidden = CONFIG["ffn_dim"]
        self.gate_proj = nn.Linear(d, hidden, bias=True)
        self.up_proj = nn.Linear(d, hidden, bias=True)
        self.down_proj = nn.Linear(hidden, d, bias=True)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        d = CONFIG["dim"]
        self.ln1 = nn.LayerNorm(d)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(d)
        self.mlp = SwiGLU()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# ==========================================================
# GPT
# ==========================================================
class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        d = CONFIG["dim"]
        self.context = CONFIG["context"]
        self.token_embedding = nn.Embedding(vocab_size, d)
        self.blocks = nn.ModuleList([Block() for _ in range(CONFIG["layers"])])
        self.norm = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        self._apply_residual_scaling()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _apply_residual_scaling(self):
        scale = 1.0 / math.sqrt(2.0 * CONFIG["layers"])
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=0.02 * scale)
            nn.init.normal_(block.mlp.down_proj.weight, mean=0.0, std=0.02 * scale)

    def forward(self, x, labels=None):
        B, T = x.shape
        x = self.token_embedding(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        if labels is not None:
            x_flat = x.view(-1, x.size(-1))
            labels_flat = labels.view(-1)
            total_loss = 0.0
            valid_tokens = 0
            chunk_size = 512

            for i in range(0, x_flat.size(0), chunk_size):
                c_x = x_flat[i:i + chunk_size]
                c_labels = labels_flat[i:i + chunk_size]
                c_logits = self.lm_head(c_x)
                c_loss = F.cross_entropy(c_logits.float(), c_labels, ignore_index=-100, reduction='sum')
                total_loss += c_loss
                valid_tokens += (c_labels != -100).sum()

            loss = total_loss / max(1, valid_tokens)
            return loss
        else:
            return self.lm_head(x)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}
