'''
Note:
The models pt file is now available on huggingface: https://huggingface.co/Digilidaahz/Freedream-1B-Instruct-V2
Please replace the CKPT_PATH as the actual pt file's name
'''

import os
import torch
import torch.nn.functional as F
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

# 🚨 导入你 Freedream_sft.py 里面的核心组件
from Freedream import GPT, tokenizer, CONFIG

# 🚨 修改为你的最新权重路径
CKPT_PATH = "model.pt" # Please replace as the actual name


@register_model("freedream")
class FreedreamLM(LM):
    def __init__(self, device="cuda", **kwargs):
        super().__init__()
        # 🚨 核心修复：避开父类的只读属性名
        self.my_device = device
        print(f"🚀 正在加载 Freedream 模型 (Device: {device})...")

        tokenizer.load("tokenizer_v2.json")
        vocab = tokenizer.size()
        self.model = GPT(vocab).to(self.my_device).to(torch.bfloat16)

        if not os.path.exists(CKPT_PATH):
            raise FileNotFoundError(f"❌ 找不到权重文件: {CKPT_PATH}")

        ck = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
        self.model.load_state_dict(clean_state_dict, strict=True)
        del ck
        self.model.eval()

    def loglikelihood(self, requests):
        results = []
        for req in requests:
            context = req.args[0]
            continuation = req.args[1]

            ctx_ids = tokenizer.encode(context)
            cont_ids = tokenizer.encode(continuation)

            if len(cont_ids) == 0:
                results.append((0.0, False))
                continue

            if len(ctx_ids) == 0:
                ctx_ids = [tokenizer.token_id("<|end|>")]

            # 防止长度超出 context
            if len(ctx_ids) + len(cont_ids) > CONFIG["context"]:
                ctx_ids = ctx_ids[-(CONFIG["context"] - len(cont_ids)):]

            input_ids = torch.tensor([ctx_ids + cont_ids], device=self.my_device)

            with torch.no_grad():
                logits = self.model(input_ids)

            shift_logits = logits[0, len(ctx_ids) - 1:-1, :].float()
            shift_labels = torch.tensor(cont_ids, device=self.my_device)

            # 用 cross_entropy 算 sum，避免显式生成巨大的 log_probs 张量
            loss = F.cross_entropy(shift_logits, shift_labels, reduction='sum')
            total_logprob = -loss.item()

            # 判断是否 greedy
            is_greedy = (shift_logits.argmax(dim=-1) == shift_labels).all().item()

            results.append((total_logprob, is_greedy))
        return results

    def loglikelihood_rolling(self, requests):
        results = []
        for req in requests:
            context = req.args[0]
            ctx_ids = tokenizer.encode(context)

            if len(ctx_ids) <= 1:
                results.append(0.0)
                continue

            total_log_prob = 0.0
            window = CONFIG["context"]

            # 核心防爆机制：分段滑动窗口处理 (非重叠，防止显存炸裂)
            for i in range(0, len(ctx_ids) - 1, window):
                chunk = ctx_ids[i:i + window]
                if len(chunk) < 2:
                    break

                input_ids = torch.tensor([chunk], device=self.my_device)
                with torch.no_grad():
                    logits = self.model(input_ids)

                shift_logits = logits[0, :-1, :].float()
                shift_labels = torch.tensor(chunk[1:], device=self.my_device)

                # 🚨 核心修改：用 cross_entropy 替代 log_softmax + gather
                loss = F.cross_entropy(shift_logits, shift_labels, reduction='sum')
                total_log_prob += -loss.item()

            # 🚨 核心修改：返回纯 float，不返回 tuple
            results.append(total_log_prob)
        return results


    def generate_until(self, requests):
        res = []
        for req in requests:
            prompt = req.args[0]
            max_toks = req.args[1].get("max_gen_toks", 50)

            ids = tokenizer.encode(prompt)
            if len(ids) > CONFIG["context"]:
                ids = ids[-CONFIG["context"]:]
            x = torch.tensor(ids, dtype=torch.long, device=self.my_device).unsqueeze(0)

            generated = []
            end_id = tokenizer.token_id("<|end|>")

            with torch.no_grad():
                for _ in range(max_toks):
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = self.model(x)[:, -1, :].float()

                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    tid = next_token.item()

                    if tid == end_id:
                        break
                    generated.append(tid)
                    x = torch.cat([x, next_token], dim=1)

            res.append(tokenizer.decode(generated).strip())
        return res