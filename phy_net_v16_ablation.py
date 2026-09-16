import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import spectral_norm
import numpy as np
import os
import warnings
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ==========================================
# 0. 全局配置与消融配置定义
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Tri-Scale Dynamic Pyramid & Ablation Training running on: {DEVICE}")

# 固定全局随机种子
torch.manual_seed(42)
np.random.seed(42)

# 创建存放权重的文件夹
os.makedirs("weights", exist_ok=True)

class AblationConfig:
    """控制每个模块是否开启的配置类"""
    def __init__(self, use_ps, use_sn, use_dp, use_so_vib, use_msjl, save_name):
        self.use_ps = use_ps             # 是否使用 Phy-Stream (物理双流)
        self.use_sn = use_sn             # 是否使用 Spectral Normalization (谱归一化)
        self.use_dp = use_dp             # 是否使用 Dynamic Pyramid (动态金字塔)
        self.use_so_vib = use_so_vib     # 是否使用 Newton-Schulz Pool + VIB
        self.use_msjl = use_msjl         # 是否使用 MSJL 联合训练策略 (Mixup + Noise)
        self.save_name = save_name       # 权重保存名称

# ==========================================
# 1. 基础组件
# ==========================================

class SpectralDerivativeLayer(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x):
        if x.shape[1] > 1:
            diff = x[:, 1:, :, :] - x[:, :-1, :, :]
            diff = F.pad(diff, (0, 0, 0, 0, 0, 1), "replicate")
        else:
            diff = x 
        return diff

class LipschitzConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, k, p=0, d=1, use_sn=True):
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, k, padding=p, dilation=d, bias=False)
        # 消融 SN: 如果 use_sn 为 True，使用谱归一化；否则使用普通卷积
        self.conv = spectral_norm(conv) if use_sn else conv
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class LipschitzLinear(nn.Module):
    def __init__(self, in_features, out_features, use_sn=True):
        super().__init__()
        linear = nn.Linear(in_features, out_features)
        self.linear = spectral_norm(linear) if use_sn else linear
    def forward(self, x): return self.linear(x)

class NewtonSchulzPool(nn.Module):
    def __init__(self, num_iterations=3):
        super().__init__()
        self.num_iterations = num_iterations
    def forward(self, x):
        B, N, C = x.shape
        mean = x.mean(dim=1, keepdim=True)
        x = x - mean
        sigma = torch.bmm(x.transpose(1, 2), x).div(N - 1)
        trace = torch.diagonal(sigma, dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
        sigma = sigma / (trace + 1e-6)
        Y = sigma
        I = torch.eye(C, device=x.device).unsqueeze(0).expand(B, C, C)
        Z = I.clone()
        for _ in range(self.num_iterations):
            T = 0.5 * (3.0 * I - torch.bmm(Z, Y))
            Y = torch.bmm(Y, T)
            Z = torch.bmm(T, Z)
        return Y.view(B, -1)

class SpectralAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# --- Tri-Scale Dynamic Block---
class RobustDynamicScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=4, use_sn=True):
        super().__init__()
        self.conv_spectral = LipschitzConv2d(in_channels, out_channels, k=1, p=0, d=1, use_sn=use_sn)
        self.conv_local = LipschitzConv2d(in_channels, out_channels, k=3, p=1, d=1, use_sn=use_sn)
        self.conv_context = LipschitzConv2d(in_channels, out_channels, k=3, p=3, d=3, use_sn=use_sn)
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // reduction, 3 * out_channels, bias=True),
        )
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):
        feat_spectral = self.conv_spectral(x)
        feat_local = self.conv_local(x)
        feat_context = self.conv_context(x)
        
        feat_sum = feat_spectral + feat_local + feat_context
        b, c, _, _ = feat_sum.size()
        
        w = self.gap(feat_sum).view(b, c)
        w = self.fc(w) 
        w = w.view(b, 3, c).permute(0, 2, 1) 
        w = self.softmax(w) 
        
        out = w[:, :, 0].view(b, c, 1, 1) * feat_spectral + \
              w[:, :, 1].view(b, c, 1, 1) * feat_local + \
              w[:, :, 2].view(b, c, 1, 1) * feat_context
        return out

class BasicConvBlock(nn.Module):
    """消融实验使用：普通的 3x3 卷积提取器，用于替换 Dynamic Pyramid"""
    def __init__(self, in_channels, out_channels, use_sn=True):
        super().__init__()
        self.conv = LipschitzConv2d(in_channels, out_channels, k=3, p=1, use_sn=use_sn)
    def forward(self, x):
        return self.conv(x)

# ==========================================
# 2. 核心网络
# ==========================================

class RobustPhyNet(nn.Module):
    def __init__(self, config: AblationConfig, in_channels=30, num_classes=16, embed_dim=64):
        super().__init__()
        self.config = config
        
        def build_stem(in_c, out_c):
            if config.use_dp:
                return nn.Sequential(
                    RobustDynamicScaleBlock(in_c, out_c, use_sn=config.use_sn),
                    RobustDynamicScaleBlock(out_c, out_c, use_sn=config.use_sn)
                )
            else:
                return nn.Sequential(
                    BasicConvBlock(in_c, out_c, use_sn=config.use_sn),
                    BasicConvBlock(out_c, out_c, use_sn=config.use_sn)
                )

        if config.use_ps:
            self.phy_op = SpectralDerivativeLayer()
            self.phy_attn = SpectralAttention(in_channels)
            self.phy_stem = build_stem(in_channels, embed_dim)
            
        self.int_stem = build_stem(in_channels, embed_dim)
        
        if config.use_ps:
            self.fusion_gate = nn.Sequential(
                LipschitzLinear(embed_dim * 2, embed_dim, use_sn=config.use_sn),
                nn.Sigmoid()
            )
        
        if config.use_so_vib:
            self.pool = NewtonSchulzPool(num_iterations=3)
            self.feat_dim = embed_dim * embed_dim 
            self.vib_enc = nn.Sequential(
                LipschitzLinear(self.feat_dim, 512, use_sn=config.use_sn),
                nn.LayerNorm(512),
                nn.GELU(),
                LipschitzLinear(512, 256, use_sn=config.use_sn) 
            )
            self.classifier = LipschitzLinear(128, num_classes, use_sn=config.use_sn)
        else:
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.classifier = LipschitzLinear(embed_dim, num_classes, use_sn=config.use_sn)

    def forward(self, x, mixup_lambda=None, target_a=None, target_b=None, inject_noise=False):
        B, C, H, W = x.shape
        
        # --- A. 特征提取阶段 ---
        feat_phy_flat = None
        if self.config.use_ps:
            x_phy_raw = self.phy_op(x)
            x_phy_weighted = self.phy_attn(x_phy_raw)
            feat_phy_flat = self.phy_stem(x_phy_weighted).flatten(2).transpose(1, 2)
            
        feat_int_flat = self.int_stem(x).flatten(2).transpose(1, 2)
        
        if self.config.use_ps:
            # 自适应噪声注入
            if inject_noise and self.training and self.config.use_msjl:
                scale_phy = feat_phy_flat.std() * 0.1
                scale_int = feat_int_flat.std() * 0.1
                feat_phy_flat = feat_phy_flat + torch.randn_like(feat_phy_flat) * scale_phy
                feat_int_flat = feat_int_flat + torch.randn_like(feat_int_flat) * scale_int

            # 特征层 Manifold Mixup
            if mixup_lambda is not None and self.training and self.config.use_msjl:
                feat_phy_flat = mixup_lambda * feat_phy_flat + (1 - mixup_lambda) * feat_phy_flat.flip(0)
                feat_int_flat = mixup_lambda * feat_int_flat + (1 - mixup_lambda) * feat_int_flat.flip(0)

            # 门控融合
            combined = torch.cat([feat_phy_flat, feat_int_flat], dim=-1)
            gate = self.fusion_gate(combined)
            feat_fused = gate * feat_phy_flat + (1 - gate) * feat_int_flat
        else:
            feat_fused = feat_int_flat
            
            # 单流情况下的自适应噪声与 Mixup
            if inject_noise and self.training and self.config.use_msjl:
                scale_int = feat_fused.std() * 0.1
                feat_fused = feat_fused + torch.randn_like(feat_fused) * scale_int
            if mixup_lambda is not None and self.training and self.config.use_msjl:
                feat_fused = mixup_lambda * feat_fused + (1 - mixup_lambda) * feat_fused.flip(0)

        # --- B. 池化与编码阶段 ---
        mu, logvar = None, None
        if self.config.use_so_vib:
            pooled = self.pool(feat_fused)
            params = self.vib_enc(pooled)
            mu = params[:, :128]
            logvar = params[:, 128:]
            
            if self.training:
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
            else:
                z = mu
            logits = self.classifier(z)
        else:
            feat_fused_spatial = feat_fused.transpose(1, 2).view(B, -1, H, W)
            pooled = self.gap(feat_fused_spatial).squeeze(-1).squeeze(-1)
            logits = self.classifier(pooled)
            
        return logits, feat_phy_flat, mu, logvar

# ==========================================
# 3. 训练策略
# ==========================================

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_ablation_model(model, train_loader, config, epochs=60): 
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"\n[{config.save_name}] 开始训练 | MSJL: {config.use_msjl} | Epochs: {epochs}")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        clean_acc = 0
        total_samples = 0
        
        for data, target in train_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            
            # --- 路线 A: MSJL 联合训练 ---
            if config.use_msjl:
                if np.random.rand() < 0.5:
                    alpha = 1.0
                    lam = np.random.beta(alpha, alpha)
                    index = torch.randperm(data.size(0)).to(DEVICE)
                    mixed_data = lam * data + (1 - lam) * data[index]
                    target_a, target_b = target, target[index]
                    
                    logits, _, mu, logvar = model(mixed_data, inject_noise=True)
                    loss_mix = mixup_criterion(F.cross_entropy, logits, target_a, target_b, lam)
                    loss_vib = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)) if mu is not None else 0
                    loss = loss_mix + 0.01 * loss_vib
                else:
                    logits_clean, _, mu_c, logvar_c = model(data, inject_noise=False)
                    loss_cls = F.cross_entropy(logits_clean, target)
                    noise = torch.randn_like(data) * 0.05
                    logits_noisy, _, _, _ = model(data + noise, inject_noise=False)
                    loss_stab = F.kl_div(F.log_softmax(logits_noisy, dim=1),
                                         F.softmax(logits_clean, dim=1),
                                         reduction='batchmean')
                    loss_vib = -0.5 * torch.mean(torch.sum(1 + logvar_c - mu_c.pow(2) - logvar_c.exp(), dim=1)) if mu_c is not None else 0
                    loss = loss_cls + 5.0 * loss_stab + 0.01 * loss_vib
            
            # --- 路线 B: 基础训练 ---
            else:
                logits, _, _, _ = model(data, inject_noise=False)
                loss = F.cross_entropy(logits, target)
                pred = logits.argmax(dim=1)
                clean_acc += (pred == target).sum().item()
                total_samples += target.size(0)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if config.use_msjl:
                with torch.no_grad():
                    if np.random.rand() < 0.5:
                        pred = model(data, inject_noise=False)[0].argmax(dim=1)
                        clean_acc += (pred == target).sum().item()
                        total_samples += target.size(0)
        
        scheduler.step()
        acc_display = clean_acc / total_samples if total_samples > 0 else 0.0
        
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"  Ep {epoch+1:2d}/{epochs} | Avg Loss: {total_loss/len(train_loader):.4f} | Est Clean Acc: {acc_display:.4f}")
            
    # 保存权重
    save_path = os.path.join("weights", config.save_name)
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存至: {save_path}")
    return model

# ==========================================
# 4. 攻击验证模块 (AttackRunner)
# ==========================================

class AttackRunner:
    @staticmethod
    def fgsm(model, X, y, epsilon=0.03):
        delta = torch.zeros_like(X, requires_grad=True)
        model.eval()
        logits, _, _, _ = model(X + delta, inject_noise=False)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        grad = delta.grad.detach()
        delta.data = epsilon * torch.sign(grad)
        return (X + delta).detach()

    @staticmethod
    def pgd(model, X, y, epsilon=0.03, alpha=0.007, num_iter=20):
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)
        delta.requires_grad = True
        for _ in range(num_iter):
            model.eval()
            logits, _, _, _ = model(X + delta, inject_noise=False)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta + alpha * torch.sign(grad)
            d = torch.clamp(d, -epsilon, epsilon)
            delta.data.copy_(d)
            delta.grad.zero_()
        return (X + delta).detach()
        
    @staticmethod
    def bim(model, X, y, epsilon=0.03, alpha=0.007, num_iter=20):
        """BIM (Basic Iterative Method) - 无随机初始化的迭代攻击"""
        delta = torch.zeros_like(X, requires_grad=True)
        for _ in range(num_iter):
            model.eval()
            logits, _, _, _ = model(X + delta, inject_noise=False)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta + alpha * torch.sign(grad)
            d = torch.clamp(d, -epsilon, epsilon)
            delta.data.copy_(d)
            delta.grad.zero_()
        return (X + delta).detach()
    
    @staticmethod
    def mim(model, X, y, epsilon=0.03, alpha=0.007, num_iter=20, decay=1.0):
        """MIM (Momentum Iterative Method) - 动量迭代跳出局部极值"""
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)
        delta.requires_grad = True
        momentum = torch.zeros_like(X)
        for _ in range(num_iter):
            model.eval()
            logits, _, _, _ = model(X + delta, inject_noise=False)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            grad = delta.grad.detach()
            grad_norm = torch.norm(grad, p=1) + 1e-10
            grad = grad / grad_norm
            momentum = decay * momentum + grad
            d = delta + alpha * torch.sign(momentum)
            d = torch.clamp(d, -epsilon, epsilon)
            delta.data.copy_(d)
            delta.grad.zero_()
        return (X + delta).detach()

    @staticmethod
    def cw_linf(model, X, y, epsilon=0.03, alpha=0.007, num_iter=20):
        """C&W Linf 变体 - 基于 Margin Loss 优化的攻击"""
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)
        delta.requires_grad = True
        for _ in range(num_iter):
            model.eval()
            logits, _, _, _ = model(X + delta, inject_noise=False)
            num_classes = logits.shape[1]
            real = logits.gather(1, y.unsqueeze(1)).squeeze(1)
            other = (logits - F.one_hot(y, num_classes=num_classes) * 1e4).max(1)[0]
            # Maximize the difference between other best class and true class
            loss = (other - real).mean() 
            loss.backward()
            grad = delta.grad.detach()
            d = delta + alpha * torch.sign(grad)
            d = torch.clamp(d, -epsilon, epsilon)
            delta.data.copy_(d)
            delta.grad.zero_()
        return (X + delta).detach()

    @staticmethod
    def _dlr_loss(logits, y, targeted=False, y_target=None):

        B = logits.size(0)
        z_y = logits.gather(1, y.unsqueeze(1)).squeeze(1)
        z_sorted, _ = torch.sort(logits, dim=1, descending=True)
        z_p1 = z_sorted[:, 0]
        z_p3 = z_sorted[:, 2] if logits.size(1)>=3 else z_sorted[:, -1]
        denom = z_p1 - z_p3 + 1e-8

        if not targeted:
            logit_mask = logits.scatter(1, y.unsqueeze(1), -float("inf"))
            z_max_other, _ = logit_mask.max(dim=1)
            loss = -(z_y - z_max_other) / denom
        else:
            z_t = logits.gather(1, y_target.unsqueeze(1)).squeeze(1)
            loss = -(z_t - z_y) / denom
        return loss

    @staticmethod
    def apgd_ce(model, X, y, epsilon, num_iter=20, restarts=1):

        B, C, H, W = X.shape
        best_adv = X.clone()
        best_loss = torch.full((B,), -float("inf"), device=X.device)

        for _ in range(restarts):
            delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)
            delta.requires_grad = True
            # APGD 自适应步长初始化
            step_size = 2 * epsilon
            loss_prev = None
            no_improve_count = 0
            patience = max(1, int(num_iter * 0.2))

            for t in range(num_iter):
                model.eval()
                x_curr = torch.clamp(X + delta, X-epsilon, X+epsilon)
                logits, _, _, _ = model(x_curr, inject_noise=False)
                loss_sample = F.cross_entropy(logits, y, reduction="none")
                loss_mean = loss_sample.mean()
                loss_mean.backward()
                grad = delta.grad.detach()

                d = delta + step_size * torch.sign(grad)
                d = torch.clamp(d, -epsilon, epsilon)
                delta.data.copy_(d)
                delta.grad.zero_()

                with torch.no_grad():
                    curr_loss_val = loss_mean.item()
                if loss_prev is not None and curr_loss_val <= loss_prev + 1e-8:
                    no_improve_count += 1
                else:
                    no_improve_count = 0
                if no_improve_count >= patience:
                    step_size *= 0.5
                    no_improve_count = 0
                loss_prev = curr_loss_val

            # 更新本重启的全局最优对抗样本
            with torch.no_grad():
                x_fin = torch.clamp(X+delta, X-epsilon, X+epsilon)
                logits_fin,_,_,_ = model(x_fin, inject_noise=False)
                curr_loss = F.cross_entropy(logits_fin, y, reduction="none")
                mask = curr_loss > best_loss
                best_adv[mask] = x_fin[mask]
                best_loss[mask] = curr_loss[mask]
        return best_adv.detach()

    @staticmethod
    def apgd_t(model, X, y, epsilon, num_iter=20, restarts=1):

        B, C, H, W = X.shape
        best_adv = X.clone()
        best_loss = torch.full((B,), -float("inf"), device=X.device)
        # 预选出定向目标标签
        with torch.no_grad():
            logits_clean, _, _, _ = model(X, inject_noise=False)
            logit_mask = logits_clean.scatter(1, y.unsqueeze(1), -float("inf"))
            y_target = logit_mask.argmax(dim=1)

        for _ in range(restarts):
            delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)
            delta.requires_grad = True
            step_size = 2 * epsilon
            loss_prev = None
            no_improve_count = 0
            patience = max(1, int(num_iter * 0.2))

            for t in range(num_iter):
                model.eval()
                x_curr = torch.clamp(X + delta, X-epsilon, X+epsilon)
                logits, _, _, _ = model(x_curr, inject_noise=False)
                loss_sample = AttackRunner._dlr_loss(logits, y, targeted=True, y_target=y_target)
                loss_mean = loss_sample.mean()
                loss_mean.backward()
                grad = delta.grad.detach()

                d = delta + step_size * torch.sign(grad)
                d = torch.clamp(d, -epsilon, epsilon)
                delta.data.copy_(d)
                delta.grad.zero_()

                with torch.no_grad():
                    curr_loss_val = loss_mean.item()
                if loss_prev is not None and curr_loss_val <= loss_prev + 1e-8:
                    no_improve_count +=1
                else:
                    no_improve_count = 0
                if no_improve_count >= patience:
                    step_size *= 0.5
                    no_improve_count = 0
                loss_prev = curr_loss_val

            with torch.no_grad():
                x_fin = torch.clamp(X+delta, X-epsilon, X+epsilon)
                logits_fin,_,_,_ = model(x_fin, inject_noise=False)
                curr_loss = AttackRunner._dlr_loss(logits_fin, y, targeted=True, y_target=y_target)
                mask = curr_loss > best_loss
                best_adv[mask] = x_fin[mask]
                best_loss[mask] = curr_loss[mask]
        return best_adv.detach()

    @staticmethod
    def fab_t_linf(model, X, y, epsilon, num_iter=20, restarts=1):

        B, C, H, W = X.shape
        best_adv = X.clone().detach()
        best_dist = torch.full((B,), float("inf"), device=X.device)

        with torch.no_grad():
            logits_clean, _, _, _ = model(X, inject_noise=False)
            logit_mask = logits_clean.scatter(1, y.unsqueeze(1), -float("inf"))
            y_target = logit_mask.argmax(dim=1)

        for _ in range(restarts):
            delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)
            alpha = 1.0

            for it in range(num_iter):
                x_curr = X + delta
                x_curr = torch.clamp(x_curr, X - epsilon, X + epsilon)
                x_curr.requires_grad_(True)

                logits, _, _, _ = model(x_curr, inject_noise=False)
                loss_sample = AttackRunner._dlr_loss(logits, y, targeted=True, y_target=y_target)
                loss = loss_sample.mean()
                loss.backward()
                grad = x_curr.grad.detach()

                g = grad
                g_sign = torch.sign(g)

                diff_z = (logits.gather(1, y_target.unsqueeze(1)).squeeze(1)
                          - logits.gather(1, y.unsqueeze(1)).squeeze(1)).detach()
                g_abs_sum = torch.sum(torch.abs(g).reshape(B, -1), dim=-1) + 1e-8
                t = (-diff_z) / g_abs_sum

                t_exp = t.view(B, 1, 1, 1)
                x_proj = x_curr + t_exp * g_sign

                alpha = max(0.05, alpha * 0.95)
                x_new = (1.0 - alpha) * X + alpha * x_proj

                delta_new = x_new - X
                delta_new = torch.clamp(delta_new, -epsilon, epsilon)
                delta.data.copy_(delta_new.data)

                x_curr.grad.zero_()

            with torch.no_grad():
                x_fin = X + delta
                # x_fin = torch.clamp(x_fin, 0.0,1.0)
                x_fin = torch.clamp(x_fin, X - epsilon, X + epsilon)
                logits_fin, _, _, _ = model(x_fin, inject_noise=False)
                pred_fin = logits_fin.argmax(dim=1)
                mask_success = (pred_fin == y_target)
                dist_curr = torch.max(torch.abs(delta).reshape(B, -1), dim=1)[0]

                mask_update = mask_success & (dist_curr < best_dist)
                best_adv[mask_update] = x_fin[mask_update]
                best_dist[mask_update] = dist_curr[mask_update]

        return best_adv.detach()

    @staticmethod
    def square_linf(model, X, y, epsilon, num_iter=500):

        B, C, H, W = X.shape
        device = X.device

        # 初始化：扰动直接落在边界 ±epsilon
        delta = epsilon * torch.randint(0, 2, size=X.shape, device=device) * 2 - 1
        x_adv = torch.clamp(X + delta, X - epsilon, X + epsilon).detach()

        with torch.no_grad():
            logits_init, _, _, _ = model(x_adv, inject_noise=False)
            loss_best = AttackRunner._dlr_loss(logits_init, y, targeted=False)

        for step in range(num_iter):
            # 方块边长调度：逐步缩小
            h = max(2, int(H * (0.5 ** (step / (num_iter * 0.3)))))

            # 每个样本独立随机方块左上角坐标
            x0 = torch.randint(0, W - h + 1, (B,), device=device)
            y0 = torch.randint(0, H - h + 1, (B,), device=device)

            x_candidate = x_adv.clone()
            for b in range(B):
                # 在h×h方块内翻转扰动符号
                patch = x_candidate[b, :, y0[b]:y0[b]+h, x0[b]:x0[b]+h]
                patch = X[b, :, y0[b]:y0[b]+h, x0[b]:x0[b]+h] - (patch - X[b, :, y0[b]:y0[b]+h, x0[b]:x0[b]+h])
                x_candidate[b, :, y0[b]:y0[b]+h, x0[b]:x0[b]+h] = patch

            x_candidate = torch.clamp(x_candidate, X - epsilon, X + epsilon)

            with torch.no_grad():
                logits_cand, _, _, _ = model(x_candidate, inject_noise=False)
                loss_cand = AttackRunner._dlr_loss(logits_cand, y, targeted=False)

            # 贪心：loss提升则接受扰动
            mask_update = loss_cand > loss_best
            x_adv[mask_update] = x_candidate[mask_update]
            loss_best[mask_update] = loss_cand[mask_update]

        return x_adv.detach()

    @staticmethod
    def autoattack(model, X, y, epsilon=0.03):

        x_adv = X.clone()
        B = X.size(0)
        attacked_mask = torch.zeros(B, dtype=torch.bool, device=X.device)

        # 1. APGD‑CE
        x_apgd_ce = AttackRunner.apgd_ce(model, X, y, epsilon=epsilon, num_iter=20)
        pred_ce = model(x_apgd_ce, inject_noise=False)[0].argmax(dim=1)
        mask_ce = (pred_ce != y) & (~attacked_mask)
        x_adv[mask_ce] = x_apgd_ce[mask_ce]
        attacked_mask[mask_ce] = True

        # 2. APGD‑T Targeted
        if (~attacked_mask).any():
            x_apgd_t = AttackRunner.apgd_t(model, X, y, epsilon=epsilon, num_iter=20)
            pred_t = model(x_apgd_t, inject_noise=False)[0].argmax(dim=1)
            mask_t = (pred_t != y) & (~attacked_mask)
            x_adv[mask_t] = x_apgd_t[mask_t]
            attacked_mask[mask_t] = True

        # 3. FAB‑T Targeted
        if (~attacked_mask).any():
            x_fab_t = AttackRunner.fab_t_linf(model, X, y, epsilon=epsilon, num_iter=20)
            pred_fabt = model(x_fab_t, inject_noise=False)[0].argmax(dim=1)
            mask_fabt = (pred_fabt != y) & (~attacked_mask)
            x_adv[mask_fabt] = x_fab_t[mask_fabt]
            attacked_mask[mask_fabt] = True

        # 4. Square Attack 黑盒兜底
        if (~attacked_mask).any():
            x_sq = AttackRunner.square_linf(model, X, y, epsilon=epsilon, num_iter=50)
            pred_sq = model(x_sq, inject_noise=False)[0].argmax(dim=1)
            mask_sq = (pred_sq != y) & (~attacked_mask)
            x_adv[mask_sq] = x_sq[mask_sq]
            attacked_mask[mask_sq] = True

        return x_adv.detach()

    @staticmethod
    def transfer_attack(model, X, y, epsilon=0.05):
        """Transfer-based Attack (Black-box) - 模拟外部跨模型迁移攻击"""
        model.eval()
        with torch.no_grad():
            num_classes = model(X, inject_noise=False)[0].shape[1]
            
        # 实时构建一个替代白盒网络进行梯度欺骗
        class SurrogateNet(nn.Module):
            def __init__(self, in_c, num_c):
                super().__init__()
                self.conv = nn.Conv2d(in_c, 64, 3, padding=1)
                self.gap = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(64, num_c)
            def forward(self, x):
                x = F.relu(self.conv(x))
                x = self.gap(x).view(x.size(0), -1)
                return self.fc(x), None, None, None
                
        surrogate = SurrogateNet(X.shape[1], num_classes).to(X.device)
        surrogate.eval()
        
        # 在替代模型上生成 PGD 对抗样本，再去攻击目标网络
        delta = torch.zeros_like(X).uniform_(-epsilon, epsilon)
        delta.requires_grad = True
        for _ in range(15):
            logits, _, _, _ = surrogate(X + delta)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            d = delta + 0.01 * torch.sign(delta.grad.detach())
            d = torch.clamp(d, -epsilon, epsilon)
            delta.data.copy_(d)
            delta.grad.zero_()
        return (X + delta).detach()

    @staticmethod
    def uap(model, X, y, epsilon=0.05, num_iter=10):
        """UAP (Universal Adversarial Perturbation) - 全局通用对抗扰动"""
        # 生成一个通用于整个 batch 的扰动量
        delta = torch.zeros((1, X.shape[1], X.shape[2], X.shape[3]), device=X.device, requires_grad=True)
        alpha = 0.01
        for _ in range(num_iter):
            model.eval()
            logits, _, _, _ = model(X + delta, inject_noise=False)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            d = delta + alpha * torch.sign(delta.grad.detach())
            d = torch.clamp(d, -epsilon, epsilon)
            delta.data.copy_(d)
            delta.grad.zero_()
        return (X + delta).detach()

    @staticmethod
    def adversarial_patch(model, X, y, patch_size=3):
        """Adversarial Patch - 模拟局部传感器损坏或云雾遮挡的物理补丁"""
        x_adv = X.clone()
        delta = torch.zeros((1, X.shape[1], patch_size, patch_size), device=X.device, requires_grad=True)
        alpha = 0.05
        for _ in range(20):
            model.eval()
            x_temp = x_adv.clone()
            x_temp[:, :, 0:patch_size, 0:patch_size] += delta
            logits, _, _, _ = model(x_temp, inject_noise=False)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            d = delta + alpha * torch.sign(delta.grad.detach())
            delta.data.copy_(d)
            delta.grad.zero_()
        
        x_adv[:, :, 0:patch_size, 0:patch_size] += delta.detach()
        return x_adv.detach()


def evaluate_comprehensive(model, test_loader):
    model.eval()
    
    epsilons = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    
    attacks = {
        "Clean": lambda m, x, y: x,
        "Noise (0.05)": lambda m, x, y: x + torch.randn_like(x) * 0.05,
    }
    
    # 1. 阶梯式的扰动预算 (PGD & FGSM)
    for eps in epsilons:
        attacks[f"FGSM (e={eps})"] = lambda m, x, y, e=eps: AttackRunner.fgsm(m, x, y, epsilon=e)
        attacks[f"PGD-20 (e={eps})"] = lambda m, x, y, e=eps: AttackRunner.pgd(m, x, y, epsilon=e, num_iter=20)
    
    # 2. 进阶白盒攻击 (White-box Attacks)
    attacks["BIM-20 (e=0.05)"] = lambda m, x, y: AttackRunner.bim(m, x, y, epsilon=0.05, num_iter=20)
    attacks["MIM-20 (e=0.05)"] = lambda m, x, y: AttackRunner.mim(m, x, y, epsilon=0.05, num_iter=20)
    attacks["C&W-Linf (e=0.05)"] = lambda m, x, y: AttackRunner.cw_linf(m, x, y, epsilon=0.05, num_iter=20)
    attacks["AutoAttack (e=0.05)"] = lambda m, x, y: AttackRunner.autoattack(m, x, y, epsilon=0.05)
    
    # 3. 黑盒攻击 (Black-box Attacks)
    attacks["Transfer (e=0.05)"] = lambda m, x, y: AttackRunner.transfer_attack(m, x, y, epsilon=0.05)

    # 4. 通用与物理攻击 (Physical / Universal)
    attacks["UAP (e=0.05)"] = lambda m, x, y: AttackRunner.uap(m, x, y, epsilon=0.05, num_iter=10)
    attacks["Adv Patch (3x3)"] = lambda m, x, y: AttackRunner.adversarial_patch(m, x, y, patch_size=3)
    
    print("\n" + "="*55)
    print("正在执行对抗鲁棒性评估...")
    print(f"{'Attack Method':<25} | {'Accuracy':<10}")
    print("-" * 55)
    
    all_data = []
    all_targets = []
    for data, target in test_loader:
        all_data.append(data)
        all_targets.append(target)
    
    if not all_data:
        return
        
    X_test = torch.cat(all_data).to(DEVICE)
    y_test = torch.cat(all_targets).to(DEVICE)
    total = X_test.size(0)
    
    for name, attack_fn in attacks.items():
        batch_size = 64
        correct = 0
        num_batches = (total + batch_size - 1) // batch_size
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, total)
            x_batch = X_test[start:end]
            y_batch = y_test[start:end]
            x_adv = attack_fn(model, x_batch, y_batch)
            with torch.no_grad():
                out = model(x_adv, inject_noise=False)[0] 
                pred = out.argmax(dim=1)
                correct += (pred == y_batch).sum().item()
        acc = correct / total
        print(f"{name:<25} | {acc:.4f}")
    print("=" * 55)

# ==========================================
# 5. 数据处理 & 主流程
# ==========================================

def applyPCA(X, numComponents=30):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

def createPatches(X, y, windowSize=13, removeZeroLabels=True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = np.pad(X, ((margin, margin), (margin, margin), (0, 0)), mode='constant')
    patchesData = []
    patchesLabels = []
    h, w, c = X.shape
    for r in range(margin, h + margin):
        for c in range(margin, w + margin):
            gt_val = y[r-margin, c-margin]
            if removeZeroLabels and gt_val == 0: continue
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData.append(patch)
            patchesLabels.append(gt_val - 1)
    return np.array(patchesData), np.array(patchesLabels)

def load_local_data():
    paths = ["data/Indian_pines_corrected.mat", "./data/Indian_pines_corrected.mat", "Indian_pines_corrected.mat"]
    gt_paths = ["data/Indian_pines_gt.mat", "./data/Indian_pines_gt.mat", "Indian_pines_gt.mat"]
    data_path, gt_path = None, None
    for p in paths:
        if os.path.exists(p): data_path = p; break
    for p in gt_paths:
        if os.path.exists(p): gt_path = p; break
            
    if data_path and gt_path:
        try:
            print(f"Loading data from: {data_path}")
            data = sio.loadmat(data_path)['indian_pines_corrected']
            gt = sio.loadmat(gt_path)['indian_pines_gt']
            print(f"Loaded Indian Pines successfully. Shape: {data.shape}")
            return data, gt
        except Exception as e:
            print(f"Error reading .mat files: {e}")
    return None, None

def run_all_ablations(train_loader, test_loader, epochs=60):
    ablation_configs = [
        # Full Model!
        AblationConfig(use_ps=True, use_sn=True, use_dp=True, use_so_vib=True, use_msjl=True, save_name="best_RobustPhyNet.pth"),
        
        # Ablations
        AblationConfig(False, False, False, False, False, "ablation_base.pth"),
        AblationConfig(True, False, False, False, False, "ablation_ps.pth"),
        AblationConfig(True, True, False, False, False, "ablation_sn.pth"),
        AblationConfig(True, True, True, False, False, "ablation_dp.pth"),
        AblationConfig(True, True, True, True, False, "ablation_vib.pth")
    ]
    
    print("\n" + "="*50)
    print("开始执行自动化消融实验训练")
    print("="*50)
    
    trained_models = {}
    for cfg in ablation_configs:
        print(f"\n正在配置并初始化模型变体: {cfg.save_name}")
        model = RobustPhyNet(config=cfg, in_channels=30, num_classes=16).to(DEVICE)
        model = train_ablation_model(model, train_loader, config=cfg, epochs=epochs)
        trained_models[cfg.save_name] = model
        
        evaluate_comprehensive(model, test_loader)
        
    print("\n所有消融模型均已训练、评估并保存完毕！")
    print("现在所有的权重已备齐。")
    return trained_models

if __name__ == "__main__":
    raw_data, raw_gt = load_local_data()
    if raw_data is not None:
        print("Preprocessing data (PCA=30, PatchSize=13)...")
        data_pca = applyPCA(raw_data, numComponents=30)
        X, y = createPatches(data_pca, raw_gt, windowSize=13)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.05, stratify=y, random_state=42
        )
        print(f"Train samples: {len(X_train)} (5%), Test samples: {len(X_test)}")
        
        train_ds = TensorDataset(torch.Tensor(X_train).permute(0,3,1,2), torch.LongTensor(y_train))
        test_ds = TensorDataset(torch.Tensor(X_test).permute(0,3,1,2), torch.LongTensor(y_test))
        
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        
        run_all_ablations(train_loader, test_loader, epochs=60)
    else:
        print("未检测到数据，正在使用模拟数据进行代码走通测试...")
        dummy_x = torch.rand(200, 30, 13, 13)
        dummy_y = torch.randint(0, 16, (200,))
        train_ds = TensorDataset(dummy_x, dummy_y)
        test_ds = TensorDataset(dummy_x, dummy_y)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
        
        run_all_ablations(train_loader, test_loader, epochs=2)