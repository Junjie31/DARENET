"""
DareNet —— DARE-Net 的单专家骨干（论文 Sec.3）。

说明：本模块实现的是「单分支专门化」的骨干。DARE 的两个专家不是在本模块内
同时激活的共享骨干双分支，而是通过 ``ffn_branch`` 二选一，各自训练出**一套独立的
backbone / Mamba 权重**（见 ``pretrained/PreTrainedModels/`` 中分别以
``_spectral_`` / ``_gated_tf_`` 命名的 checkpoint）：

    ffn_branch = 'spectral'  ->  Complex（Precision）专家：复数频域 FFN
    ffn_branch = 'gated_tf'  ->  Gated（Robust）专家：门控时频融合 FFN

论文 Sec.3 的 Eq. routing ``Y = X + beta*Y_prec + (1-beta)*Y_rob`` 中的 ``beta``
在此处退化为训练时的二值专家选择（beta=1 训 Complex，beta=0 训 Gated），对应论文
"training-time expert-specialisation formulation" 的表述。推理时的双专家路由 /
融合 / 拒绝，见 ``DualExpertDareNet.py``（显式 MoE 视图）。
"""
import math
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from einops import rearrange
from timm.models.layers import trunc_normal_, lecun_normal_, DropPath
from mamba_ssm.modules.mamba_simple import Mamba

# === 关键：复用 v6 中的 Fusion_Stem / Attention_mask，确保 stem 部分 100% 一致 ===
# from neural_methods.model.DareNet_v6 import Fusion_Stem, Attention_mask  # ← 按你的工程改路径


# # -----------------------
# # Safe helpers
# # -----------------------
def nan_safe(x: torch.Tensor) -> torch.Tensor:
    """Replace NaN/Inf with finite numbers (0)."""
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

def design_bandpass_fir(fps: int, low_bpm: float = 40.0, high_bpm: float = 180.0, taps: int = 7) -> torch.Tensor:
    """Tiny windowed-sinc band-pass FIR (odd taps)."""
    assert taps % 2 == 1, "FIR taps must be odd."
    nyq = fps / 2.0
    f1 = (low_bpm / 60.0) / nyq
    f2 = (high_bpm / 60.0) / nyq
    n = torch.arange(taps, dtype=torch.float32)
    m = (taps - 1) / 2.0

    def sinc(x):  # numeric-safe sinc
        return torch.where(x == 0, torch.ones_like(x), torch.sin(torch.pi * x) / (torch.pi * x))

    h_lp2 = 2 * f2 * sinc(2 * f2 * (n - m))
    h_lp1 = 2 * f1 * sinc(2 * f1 * (n - m))
    h_bp = h_lp2 - h_lp1
    w = 0.54 - 0.46 * torch.cos(2 * torch.pi * n / (taps - 1))  # Hamming
    h = h_bp * w
    h = h / (h.sum() + 1e-6)
    return h

def apply_temporal_fir(diff_nd: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    Depthwise temporal 1D FIR on D: diff_nd [N,D,C,H,W], h [T].
    """
    N, D, C, H, W = diff_nd.shape
    if D < 3 or h.numel() < 3:
        return diff_nd
    T = h.numel()
    pad = (T - 1) // 2
    x = diff_nd.permute(0, 3, 4, 2, 1).contiguous().reshape(N * H * W, C, D)
    h = h.to(x.device, x.dtype).reshape(1, 1, T)
    x = F.pad(x, (pad, pad), mode='replicate')
    x = F.conv1d(x, h.repeat(C, 1, 1), stride=1, padding=0, groups=C)
    x = x.reshape(N, H, W, C, D).permute(0, 4, 3, 1, 2).contiguous()
    return nan_safe(x)


# -----------------------
# Optional small extras (default OFF to mimic original)
# -----------------------
class SoftSkinMask(nn.Module):
    """
    Soft skin attention (safe, conservative):
    mask in [0.95, 1.05], initialized to 1.0 (no change).
    """
    def __init__(self):
        super().__init__()
        self.score = nn.Conv2d(3, 1, kernel_size=1, bias=True)
        with torch.no_grad():
            nn.init.zeros_(self.score.weight)
            nn.init.zeros_(self.score.bias)

    def forward(self, raw_flat: torch.Tensor) -> torch.Tensor:
        s = self.score(raw_flat)
        m = 1.0 + 0.1 * (torch.sigmoid(s) - 0.5)  # [0.95, 1.05]
        return nan_safe(m)



class SafuGate(nn.Module):
    def __init__(self, c_mid: int, hidden: int = 32, d_code: int = 0):
        super().__init__()
        self.d_code = d_code
        self.norm = nn.LayerNorm(2*c_mid + d_code)
        self.fc1 = nn.Linear(2*c_mid + d_code, hidden)
        self.fc2 = nn.Linear(hidden, 2)
        nn.init.kaiming_uniform_(self.fc1.weight, a=0.1)
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, a=0.1)
        nn.init.zeros_(self.fc2.bias)
    def forward(self, xr, xd, domain_code=None):
        r = F.adaptive_avg_pool2d(xr, 1).flatten(1)
        d = F.adaptive_avg_pool2d(xd, 1).flatten(1)
        h = torch.cat([r, d], dim=1)
        if self.d_code > 0:
            if domain_code is None:
                raise ValueError("domain_code is required")
            dc = domain_code.expand(h.shape[0], -1)
            h = torch.cat([h, dc], dim=1)
        h = self.norm(h)
        h = F.relu(self.fc1(h), inplace=False)
        ab = torch.sigmoid(self.fc2(h))
        ab = nan_safe(ab)
        alpha = ab[:, 0].reshape(-1, 1, 1, 1)
        beta  = ab[:, 1].reshape(-1, 1, 1, 1)
        return alpha, beta



class Fusion_Stem(nn.Module):
    """
    保留原骨干拓扑（7×7 + BN + ReLU + MaxPool）与总下采样 /8，
    仅在时间差分与数值稳定上做增强 + 极轻量门控。

    输出：严格 [N*D, 24, H/8, W/8]，不产生 NaN/Inf。
    """
    def __init__(self,
                 apha: float = 0.6,
                 belta: float = 0.4,
                 dim: int = 24,
                 use_diff_rms_norm: bool = True,
                 rms_min: float = 1e-4,
                 use_bandpass: bool = False,
                 fps: int = 30,
                 low_bpm: float = 40.0,
                 high_bpm: float = 180.0,
                 taps: int = 7,
                 use_skin_mask: bool = False,
                 use_safu_gate: bool = True,
                 gate_scale_stage1: float = 0,   # 新增：SafuGate 的影响强度
                 res_inject_scale: float = 0.0,
                 bn_eps: float = 1e-3,
                 temporal_pad_mode: str = 'replicate'):
        super().__init__()

        # ---- keep original backbone topology (space encoder) ----
        self.stem11 = nn.Sequential(
            nn.Conv2d(3, dim // 2, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(dim // 2, eps=bn_eps, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False),   # /4
        )
        self.stem12 = nn.Sequential(
            nn.Conv2d(12, dim // 2, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(dim // 2, eps=bn_eps, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False),   # /4
        )
        self.stem21 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(dim, eps=bn_eps, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False),   # /8
        )
        self.stem22 = nn.Sequential(
            nn.Conv2d(dim // 2, dim, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(dim, eps=bn_eps, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=False),   # /8
        )

        # ---- options ----
        self.apha = apha
        self.belta = belta
        self.use_diff_rms_norm = use_diff_rms_norm
        self.rms_min = rms_min
        self.use_bandpass = use_bandpass and (fps is not None)
        self.fps = fps
        self.low_bpm = low_bpm
        self.high_bpm = high_bpm
        self.taps = taps
        if self.use_bandpass:
            h = design_bandpass_fir(fps=self.fps, low_bpm=self.low_bpm, high_bpm=self.high_bpm, taps=self.taps)
            self.register_buffer('bp_kernel', h)
        else:
            self.register_buffer('bp_kernel', torch.tensor([], dtype=torch.float32))

        self.use_skin_mask = use_skin_mask
        self.skin_mask = SoftSkinMask() if use_skin_mask else nn.Identity()

        self.use_safu_gate = use_safu_gate
        self.gate_scale_stage1 = gate_scale_stage1
        self.gate = SafuGate(c_mid=dim // 2,
                             hidden=max(32, dim)
                            #  gate_scale=gate_scale_stage1) if use_safu_gate else None
                             ) 
        # residual energy injection (safe, default 0.0)
        self.res_proj = nn.Conv2d(12, dim, kernel_size=1, bias=False)
        nn.init.kaiming_uniform_(self.res_proj.weight, a=0.1)
        self.res_scale = res_inject_scale

        # output projection (identity if dim==24)
        self.out_proj = nn.Identity() if dim == 24 else nn.Conv2d(dim, 24, kernel_size=1, bias=False)

        # temporal pad mode
        self.temporal_pad_mode = temporal_pad_mode

    # ---- temporal slicing (replicate-safe; mimic original for D>=3) ----
    def _build_temporal_slices(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        N, D, C, H, W = x.shape
        if self.temporal_pad_mode == 'legacy' and D >= 3:
            x1 = torch.cat([x[:, :1], x[:, :1], x[:, :D - 2]], dim=1)
            x2 = torch.cat([x[:, :1], x[:, :D - 1]], dim=1)
            x3 = x
            x4 = torch.cat([x[:, 1:], x[:, D - 1:]], dim=1)
            x5 = torch.cat([x[:, 2:], x[:, D - 1:], x[:, D - 1:]], dim=1)
        else:
            # Robust replicate padding（默认），避免 D 很小时构造切片崩溃
            left  = x[:, :2] if D >= 2 else x[:, :1].repeat(1, 2, 1, 1, 1)
            right = x[:, -2:] if D >= 2 else x[:, -1:].repeat(1, 2, 1, 1, 1)
            xp = torch.cat([left, x, right], dim=1)  # [N, D+4, C, H, W]
            x1 = xp[:, 0:D].contiguous()
            x2 = xp[:, 1:D + 1].contiguous()
            x3 = xp[:, 2:D + 2].contiguous()
            x4 = xp[:, 3:D + 3].contiguous()
            x5 = xp[:, 4:D + 4].contiguous()
        return x1, x2, x3, x4, x5

    # ---- forward ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input : x [N, D, 3, H, W]
        Output: fused [N*D, 24, H/8, W/8]
        """
        assert x is not None and x.dim() == 5, f"Expected [N,D,3,H,W], got {None if x is None else x.shape}"
        N, D, C, H, W = x.shape
        assert C == 3, f"Expected 3 channels, got C={C}"

        # 1) 时间切片 + 帧间差分（12 通道）
        x1, x2, x3, x4, x5 = self._build_temporal_slices(x)
        diff_nd = torch.cat([x2 - x1, x3 - x2, x4 - x3, x5 - x4], dim=2).contiguous()  # [N,D,12,H,W]
        diff_nd = nan_safe(diff_nd)

        # 2) 可选：时间带通（fps 匹配时才打开）
        if self.use_bandpass and self.bp_kernel.numel() > 0:
            diff_nd = apply_temporal_fir(diff_nd, self.bp_kernel)  # [N,D,12,H,W]

        # 3) 可选：逐时间步 RMS 标准化
        if self.use_diff_rms_norm:
            rms = torch.sqrt((diff_nd ** 2).mean(dim=(2, 3, 4), keepdim=True) + 1e-6)
            rms = torch.clamp(rms, min=self.rms_min)
            diff_nd = nan_safe(diff_nd / rms)

        # 4) 可选：保守皮肤掩码
        if isinstance(self.skin_mask, SoftSkinMask):
            raw_flat = x3.contiguous().reshape(N * D, 3, H, W)
            m = self.skin_mask(raw_flat).reshape(N, D, 1, H, W).contiguous()
            diff_nd = nan_safe(diff_nd * m)

        # 5) 展开时间 -> stage-1 编码（/4）
        x_raw  = x3.contiguous().reshape(N * D, 3,  H, W)
        x_diff = diff_nd.contiguous().reshape(N * D, 12, H, W)

        xr = self.stem11(x_raw)      # [N*D, dim//2, H/4, W/4]
        xd = self.stem12(x_diff)     # [N*D, dim//2, H/4, W/4]

        # 6) 轻量门控：只做小幅系数调整
        if self.use_safu_gate and (self.gate is not None):
            domain_code = torch.zeros((N, 1), device=x.device) 
            alpha_hat, beta_hat = self.gate(xr, xd, domain_code=domain_code)       # [N*D,1,1,1], in (0,1) and ~0.5
            # domain_code = torch.zeros((B, 1)) 
            # alpha_hat, beta_hat = self.gate(xr, xd)  
            # 基准为全局 apha/belta，乘上一个 [1-e, 1+e] 的小扰动
            # 这样不会完全推翻全局配比，更鲁棒
            eps = 0.2  # 可以再挪到 __init__ 里做成超参
            A = self.apha * (1.0 + eps * (alpha_hat - 0.5) * 2.0)
            B = self.belta * (1.0 + eps * (beta_hat  - 0.5) * 2.0)
        else:
            A = self.apha
            B = self.belta

        # 7) 融合 -> stage-2（/8）
        x_path1 = A * xr + B * xd
        x_path1 = self.stem21(x_path1)                  # [N*D, dim, H/8, W/8]
        x_path2 = self.stem22(xd)                       # [N*D, dim, H/8, W/8]

        fused = self.apha * x_path1 + self.belta * x_path2
        fused = nan_safe(fused)

        # 8) 极轻能量残差注入（默认 0.0，不改变原分布）
        if self.res_scale > 0.0:
            h8, w8 = fused.shape[-2], fused.shape[-1]
            diff_8 = F.adaptive_avg_pool2d(x_diff, (h8, w8))   # [N*D,12,H/8,W/8]
            rei = self.res_proj(diff_8)                        # [N*D,dim,H/8,W/8]
            fused = nan_safe(fused + self.res_scale * rei)

        # 9) 输出通道为 24，且数值有限
        fused = self.out_proj(fused)               # [N*D,24,H/8,W/8]
        fused = nan_safe(fused)
        return fused




class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=3, keepdim=True)
        xsum = torch.sum(xsum, dim=4, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[3] * xshape[4] * 0.5

    def get_config(self):
        """May be generated manually. """
        config = super(Attention_mask, self).get_config()
        return config


# =============================================================================
# Frequencydomain_FFN  (v8: v6 实现 + ffn_branch 开关)
# =============================================================================
class Frequencydomain_FFN(nn.Module):
    """
    v6 行为完全保留；新增 ffn_branch 开关。
    submodule 命名与 v6 完全一致：
      spectral 分支 -> r, i, rb, ib, fc1, fc2
      gated_tf 分支 -> time_branch, freq_branch, fusion_gate, out_proj
    """
    VALID_BRANCH = {"auto", "spectral", "gated_tf"}

    def __init__(self, dim, mlp_ratio, dataset_name, ffn_branch="auto"):
        super().__init__()
        assert ffn_branch in self.VALID_BRANCH, f"bad ffn_branch={ffn_branch}"
        self.dataset_name = dataset_name
        self.ffn_branch = ffn_branch
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.active_branch = self._resolve_branch(ffn_branch, dataset_name)

        if self.active_branch == "spectral":
            self._init_ubfc_path(dim, mlp_ratio)
            self.forward = self._forward_ubfc
        else:
            self._init_pure_path(dim, mlp_ratio)
            self.forward = self._forward_pure

    @staticmethod
    def _resolve_branch(ffn_branch, dataset_name):
        if ffn_branch == "spectral":
            return "spectral"
        if ffn_branch == "gated_tf":
            return "gated_tf"
        # auto: 与 v6 一致
        if "UBFC" in dataset_name:
            return "gated_tf"
        return "gated_tf"

    # -------- v6 _init_ubfc_path 原样复制 --------
    def _init_ubfc_path(self, dim, mlp_ratio):
        self.scale = 0.02
        self.hidden_dim = int(dim * mlp_ratio)
        self.r = nn.Parameter(self.scale * torch.randn(self.hidden_dim, self.hidden_dim))
        self.i = nn.Parameter(self.scale * torch.randn(self.hidden_dim, self.hidden_dim))
        self.rb = nn.Parameter(self.scale * torch.randn(self.hidden_dim))
        self.ib = nn.Parameter(self.scale * torch.randn(self.hidden_dim))
        self.fc1 = nn.Sequential(
            nn.Conv1d(dim, self.hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(self.hidden_dim, dim, 1, 1, 0, bias=False),
            nn.BatchNorm1d(dim),
        )

    # -------- v6 _init_pure_path 原样复制 --------
    def _init_pure_path(self, dim, mlp_ratio):
        self.hidden_dim = int(dim * mlp_ratio)
        self.time_branch = nn.Sequential(
            nn.Conv1d(dim, self.hidden_dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv1d(self.hidden_dim, self.hidden_dim, 3, padding=1),
            nn.GELU(),
        )
        self.freq_branch = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(self.hidden_dim, dim)

    # -------- v6 _compute_fingerprint 原样复制 --------
    def _compute_fingerprint(self, x_in):
        sig = x_in[:, :, 0]
        diff = sig[:, 1:] - sig[:, :-1]
        energy = torch.mean(diff ** 2, dim=1) * 100.0
        fft_val = torch.abs(torch.fft.rfft(sig, dim=1))
        fft_power = fft_val ** 2
        psd = fft_power / (torch.sum(fft_power, dim=1, keepdim=True) + 1e-8)
        entropy = -torch.sum(psd * torch.log(psd + 1e-8), dim=1)
        return torch.stack([energy, entropy], dim=1)

    # -------- v6 _forward_ubfc 原样复制 --------
    def _forward_ubfc(self, x, return_fp=False):
        B, N, C = x.shape
        if return_fp:
            fp = self._compute_fingerprint(x)

        x = self.fc1(x.transpose(1, 2)).transpose(1, 2)
        x_fre = torch.fft.fft(x, dim=1, norm='ortho')
        x_real = F.relu(
            torch.einsum('bnc,cc->bnc', x_fre.real, self.r)
            - torch.einsum('bnc,cc->bnc', x_fre.imag, self.i)
            + self.rb
        )
        x_imag = F.relu(
            torch.einsum('bnc,cc->bnc', x_fre.imag, self.r)
            + torch.einsum('bnc,cc->bnc', x_fre.real, self.i)
            + self.ib
        )
        x_fre = torch.stack([x_real, x_imag], dim=-1).float()
        x_fre = torch.view_as_complex(x_fre)
        x = torch.fft.ifft(x_fre, dim=1, norm="ortho")
        x = x.to(torch.float32)
        x = self.fc2(x.transpose(1, 2)).transpose(1, 2)

        if return_fp:
            return x, fp
        return x

    # -------- v6 _forward_pure 原样复制 --------
    def _forward_pure(self, x, return_fp=False):
        B, N, C = x.shape
        if return_fp:
            fp = self._compute_fingerprint(x)

        time_feat = self.time_branch(x.transpose(1, 2)).transpose(1, 2)

        x_freq = torch.fft.fft(x, dim=1, norm='ortho')
        freq_magnitude = torch.abs(x_freq)
        freq_feat = self.freq_branch(freq_magnitude)

        combined = torch.cat([time_feat, freq_feat], dim=-1)
        gate = self.fusion_gate(combined)
        fused = gate * time_feat + (1 - gate) * freq_feat

        output = self.out_proj(fused)

        if return_fp:
            return x + output, fp
        return x + output


# =============================================================================
# MambaLayer  (v6 原样)
# =============================================================================
class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=48, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        B, N, C = x.shape
        x_norm = self.norm(x)
        x_mamba = self.mamba(x_norm)
        return x_mamba


# =============================================================================
# Block_mamba  (v6 原样 + ffn_branch 透传)
# =============================================================================
class Block_mamba(nn.Module):
    def __init__(
        self,
        dataset_name,
        dim,
        mlp_ratio,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        ffn_branch="auto",
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.attn = MambaLayer(dim)
        self.mlp = Frequencydomain_FFN(dim, mlp_ratio, dataset_name, ffn_branch=ffn_branch)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, return_fp=False):
        B, D, C = x.size()
        path = 3
        segment = 2 ** (path - 1)
        tt = D // segment
        x_r = x.repeat(segment, 1, 1)
        x_o = x_r.clone()
        for i in range(1, segment):
            x_o[i * B:(i + 1) * B, :D - i * tt, :] = x_r[i * B:(i + 1) * B, i * tt:, :]
        x_o = self.attn(x_o)
        for i in range(1, segment):
            for j in range(i):
                x_o[0:B, tt * i: tt * (i + 1), :] = (
                    x_o[0:B, tt * i: tt * (i + 1), :]
                    + x_o[B * (j + 1):B * (j + 2), tt * (i - j - 1): tt * (i - j), :]
                )
            x_o[0:B, tt * i: tt * (i + 1), :] = x_o[0:B, tt * i: tt * (i + 1), :] / (i + 1)
        x = x + self.drop_path(self.norm1(x_o[0:B]))

        if return_fp:
            mlp_out, fp = self.mlp(self.norm2(x), return_fp=True)
            x = x + self.drop_path(mlp_out)
            return x, fp
        else:
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


# =============================================================================
# init helpers (v6 原样)
# =============================================================================
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


# =============================================================================
# DareNet (v6 原样 + ffn_branch 透传)
# =============================================================================
class DareNet(nn.Module):
    def __init__(
        self,
        depth=24,
        embed_dim=96,
        mlp_ratio=2,
        drop_rate=0.,
        drop_path_rate=0.1,
        initializer_cfg=None,
        device=None,
        dtype=None,
        dataset_name='UBFC',
        ffn_branch='auto',           # ★ v8 唯一新增
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        kwargs.update(factory_kwargs)
        super().__init__()
        self.embed_dim = embed_dim
        self.dataset_name = dataset_name
        self.ffn_branch = ffn_branch

        self.Fusion_Stem = Fusion_Stem(dim=embed_dim // 4)   # ← 复用 v6 的 Fusion_Stem
        self.attn_mask = Attention_mask()                    # ← 复用 v6 的 Attention_mask

        self.stem3 = nn.Sequential(
            nn.Conv3d(embed_dim // 4, embed_dim,
                      kernel_size=(2, 5, 5), stride=(2, 1, 1), padding=(0, 2, 2)),
            nn.BatchNorm3d(embed_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        inter_dpr = [0.0] + dpr
        self.blocks = nn.ModuleList([
            Block_mamba(
                dataset_name=dataset_name,
                dim=embed_dim,
                mlp_ratio=mlp_ratio,
                drop_path=inter_dpr[i],
                norm_layer=nn.LayerNorm,
                ffn_branch=ffn_branch,
            )
            for i in range(depth)
        ])

        self.upsample = nn.Upsample(scale_factor=2)
        self.ConvBlockLast = nn.Conv1d(embed_dim, 1, kernel_size=1, stride=1, padding=0)

        self.apply(segm_init_weights)
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    # ---- helper ----
    def active_ffn_branches(self):
        return [blk.mlp.active_branch for blk in self.blocks]

    # ---- forward (v6 原样) ----
    def forward(self, x, return_fp=False):
        B, D, C, H, W = x.shape

        x = self.Fusion_Stem(x)
        x = x.view(B, D, self.embed_dim // 4, H // 8, W // 8).permute(0, 2, 1, 3, 4)
        x = self.stem3(x)

        mask = torch.sigmoid(x)
        mask = self.attn_mask(mask)
        x = x * mask

        x = torch.mean(x, 4)
        x = torch.mean(x, 3)
        x = rearrange(x, 'b c t -> b t c')

        if return_fp:
            fp_list = []
            for blk in self.blocks:
                x, fp = blk(x, return_fp=True)
                fp_list.append(fp)
            fp_stacked = torch.stack(fp_list, dim=0)
            fp_mean = fp_stacked.mean(dim=0)
        else:
            for blk in self.blocks:
                x = blk(x)

        rPPG = x.permute(0, 2, 1)
        rPPG = self.upsample(rPPG)
        rPPG = self.ConvBlockLast(rPPG)
        rPPG = rPPG.squeeze(1)

        if return_fp:
            return rPPG, fp_mean
        return rPPG