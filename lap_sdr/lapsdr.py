import torch
import torch.nn.functional as F

class LAPSDRRouter:
    """
    LAP-SDR: Look-Ahead Probe with Softmax Dynamic Routing.
    """
    def __init__(self, accumulation=16, tau=1.0, w_min=0.6, w_max=1.4, probe_len=256):
        self.accumulation = accumulation
        self.tau = tau
        self.w_min = w_min
        self.w_max = w_max
        self.probe_len = probe_len

    def compute_weights(self, model, x_step, y_step):
        """
        动态探路并计算 Zero-Sum 权重
        """
        model.eval()
        probe_losses = []
        
        with torch.no_grad():
            for micro_step in range(self.accumulation):
                # 只截取前 probe_len 个 token 进行极速探路
                x_probe = x_step[micro_step:micro_step+1, :self.probe_len]
                y_probe = y_step[micro_step:micro_step+1, :self.probe_len]
                
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    probe_loss = model(x_probe, labels=y_probe)
                probe_losses.append(probe_loss.detach())

        loss_tensor = torch.stack(probe_losses)
        
        # Softmax Routing
        weights = F.softmax(loss_tensor / self.tau, dim=0) * self.accumulation
        
        # Clamp 截断
        weights = torch.clamp(weights, min=self.w_min, max=self.w_max)
        
        # Zero-Sum Renormalization
        final_weights = weights * (self.accumulation / weights.sum())
        
        model.train()
        return final_weights

    def train_step(self, model, x_step, y_step):
        """
        执行带权重的反向传播，返回真实的平均 Loss
        """
        final_weights = self.compute_weights(model, x_step, y_step)
        
        loss_step_accum = 0.0
        for micro_step in range(self.accumulation):
            x = x_step[micro_step:micro_step+1]
            y = y_step[micro_step:micro_step+1]
            
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, labels=y)
                # 核心魔法：乘上动态权重
                weighted_loss = loss * final_weights[micro_step] / self.accumulation
                
            weighted_loss.backward()
            loss_step_accum += loss.item() / self.accumulation
            
            del x, y, loss, weighted_loss
            
        return loss_step_accum, final_weights
