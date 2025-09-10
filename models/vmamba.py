import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

# import traceback
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"
# print(traceback.extract_stack())


try:  ## successfully try
    "sscore acts the same as mamba_ssm"
    SSMODE = "sscore"
    # import selective_scan_cuda_core
    import adaptive_selective_scan_cuda_core
except Exception as e:
    print(e, flush=True)
    "you should install mamba_ssm to use this"
    SSMODE = "mamba_ssm"
    import selective_scan_cuda


# fvcore flops =======================================


# cross selective scan ===============================


class Adaptive_SelectiveScan(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(
        ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, snr=10
    ):
        assert nrows in [1, 2, 3, 4], f"{nrows}"  # 8+ is too slow to compile
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True
        # print(SSMODE)
        if SSMODE == "mamba_ssm":
            out, x, *rest = selective_scan_cuda.fwd(
                u, delta, A, B, C, D, None, delta_bias, delta_softplus
            )
        else:

            out, x, *rest = adaptive_selective_scan_cuda_core.fwd(
                u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, snr
            )
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)

        ##---- if you want check the CSI-forgetten, you can add the following code, our results get from the checkpoint train on DIV2K dataset with rayleigh fading channel------#

        # delta=F.softplus(delta.float()+delta_bias[..., None].float())
        # # snr_vec=A.new_zeros(u.shape[0],A.shape[0],A.shape[1])
        # deltaA=torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        # A_cum=deltaA.new_ones(deltaA.shape)
        # A_cum[:,:,0,:]=deltaA[:,:,0,:]
        # for i in range(1,deltaA.shape[2]):
        #       A_cum[:,:,i,:]=A_cum[:,:,i-1,:]*deltaA[:,:,i,:]
        # breakpoint()

        ####------------------end---------------------#
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        flag = 0
        # breakpoint()
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        if SSMODE == "mamba_ssm":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u,
                delta,
                A,
                B,
                C,
                D,
                None,
                delta_bias,
                dout,
                x,
                None,
                None,
                ctx.delta_softplus,
                False,  # option to recompute out_z, not used here
            )
        else:
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = adaptive_selective_scan_cuda_core.bwd(
                u,
                delta,
                A,
                B,
                C,
                D,
                delta_bias,
                dout,
                x,
                ctx.delta_softplus,
                1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        # breakpoint()

        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None)


class SelectiveScan(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}"  # 8+ is too slow to compile
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        # all in float
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True
        # print(SSMODE)
        if SSMODE == "mamba_ssm":
            out, x, *rest = selective_scan_cuda.fwd(
                u, delta, A, B, C, D, None, delta_bias, delta_softplus
            )
        else:
            out, x, *rest = selective_scan_cuda_core.fwd(
                u, delta, A, B, C, D, delta_bias, delta_softplus, nrows
            )
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)

        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        # breakpoint()
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        if SSMODE == "mamba_ssm":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u,
                delta,
                A,
                B,
                C,
                D,
                None,
                delta_bias,
                dout,
                x,
                None,
                None,
                ctx.delta_softplus,
                False,  # option to recompute out_z, not used here
            )
        else:

            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u,
                delta,
                A,
                B,
                C,
                D,
                delta_bias,
                dout,
                x,
                ctx.delta_softplus,
                1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )
        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        # breakpoint()

        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


class CrossScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)

        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)  # 前向扫描
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)  # 翻转，等于竖向扫描
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])  # 翻转，另外两向
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):

        B, C, H, W = ctx.shape
        L = H * W
        #
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(
            B, -1, L
        )
        end = time.time()

        return y.view(B, -1, H, W)


class CrossMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)

        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(
            B, D, -1
        )  # 四个方向整理顺序相加

        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):

        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs, None, None


class CrossScan2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)

        xs = x.new_empty((B, 2, C, H * W))
        xs[:, 0] = x.flatten(2, 3)  # 前向扫描

        xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])  # 翻转，另外两向
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # start=time.time()
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0].view(B, 1, -1, L) + ys[:, 1].flip(dims=[-1]).view(B, 1, -1, L)

        end = time.time()

        return ys.view(B, -1, H, W)


class CrossMerge2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        # print(ys.shape)
        # print(H,W)
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        # print(ys[].shape)
        ys = ys[:, 0].view(B, 1, D, -1) + ys[:, 1].flip(dims=[-1]).view(B, 1, D, -1)
        # print(ys.shape)
        # y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1) #四个方向整理顺序相加

        y = ys.view(B, D, -1)
        # print(y.shape)
        return y

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 2, C, L))
        xs[:, 0] = x

        # xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])

        xs = xs.view(B, 2, C, H, W)
        # print(xs.shape)
        return xs, None, None


def cross_selective_scan(
    x: torch.Tensor = None,
    x_proj_weight: torch.Tensor = None,
    x_proj_bias: torch.Tensor = None,
    dt_projs_weight: torch.Tensor = None,
    dt_projs_bias: torch.Tensor = None,
    A_logs: torch.Tensor = None,
    Ds: torch.Tensor = None,
    out_norm: torch.nn.Module = None,
    nrows=-1,
    delta_softplus=True,
    to_dtype=True,
    scan="cross",
    conv_scan=None,
    conv_merge=None,
    scan_number=4,
    adaptive="no",
    snr=10,
    InvertScan=None,
    InvertMerge=None,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, H, W = x.shape
    # print(D)
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W
    # print(R,N,N)
    if nrows < 1:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1
    # if scan=="cross":
    if scan_number == 2:
        xs = CrossScan2.apply(x)
    elif scan_number == 4:
        xs = CrossScan.apply(x)
    else:
        raise ValueError("scan number error")

    # print(xs.shape)
    # print(xs.get_device(),x_proj_weight.get_device())
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)  # 两个矩阵相乘的自定义方法
    # breakpoint()
    # print(xs.shape,x_dbl.shape,x_proj_weight.shape)
    if x_proj_bias is not None:
        # print(x_dbl.shape.x_proj_weight.shape)
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)

    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

    xs = xs.view(B, -1, L).to(torch.float)
    # print(xs.shape)
    dts = dts.contiguous().view(B, -1, L).to(torch.float)
    As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
    Bs = Bs.contiguous().to(torch.float)
    Cs = Cs.contiguous().to(torch.float)
    # print('CS',Cs.shape)
    Ds = Ds.to(torch.float)  # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)
    # breakpoint()

    def selective_scan(
        u,
        delta,
        A,
        B,
        C,
        D=None,
        delta_bias=None,
        delta_softplus=True,
        nrows=1,
        adaptive="no",
        snr=10,
    ):
        # print("u.shape[2]", u.shape[2])
        if adaptive == "ssm":
            return Adaptive_SelectiveScan.apply(
                u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, snr
            )
        else:
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows, adaptive, snr
    ).view(B, K, -1, H, W)
    # if scan=="cross":
    if scan_number == 2:
        y: torch.Tensor = CrossMerge2.apply(ys)
    elif scan_number == 4:
        y: torch.Tensor = CrossMerge.apply(ys)
    else:
        raise ValueError("scan number error")

    y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
    if out_norm is not None:
        y = out_norm(y)
    y = y.view(B, H, W, -1)
    # print("y",y.shape)
    return y.to(x.dtype) if to_dtype else y


# =====================================================


class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        x = x[0]
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        # print('downsampling')
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchReverseMerging2D(nn.Module):
    r"""Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    """

    def __init__(self, dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.increment = nn.Linear(dim, out_dim * 4, bias=False)
        self.norm = norm_layer(dim)
        # self.proj = nn.ConvTranspose2d(dim // 4, 3, 3, stride=1, padding=1)
        # self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        x = x[0]
        H, W, _ = x.shape[-3:]
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = self.norm(x)
        x = self.increment(x)
        x = x.permute(0, 3, 1, 2)
        x = nn.PixelShuffle(2)(x)

        x = x.permute(0, 2, 3, 1)
        return x


class SS2D(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3,  # < 2 means no conv
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        simple_init=False,
        # ======================
        forward_type="v2",
        scan="cross",
        PE="no",
        resolution=128,
        scan_number=4,
        adaptive="no",
        # ======================
        **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        self.adaptive = adaptive

        self.scan = scan
        self.PE = PE
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.scan_number = scan_number
        # disable z act ======================================
        self.disable_z_act = forward_type[-len("nozact") :] == "nozact"
        if self.disable_z_act:
            forward_type = forward_type[: -len("nozact")]

        # softmax | sigmoid | norm ===========================
        if forward_type[-len("softmax") :] == "softmax":
            forward_type = forward_type[: -len("softmax")]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type[-len("sigmoid") :] == "sigmoid":
            forward_type = forward_type[: -len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)  # layernorm

        # forward_type =======================================
        self.forward_core = dict(
            v0=self.forward_corev0,
            v0_seq=self.forward_corev0_seq,
            v1=self.forward_corev2,
            v2=self.forward_corev2,
            share_ssm=self.forward_corev0_share_ssm,
            share_a=self.forward_corev0_share_a,
        ).get(forward_type, self.forward_corev2)
        # print("forward_type_2",forward_type)
        self.K = scan_number if forward_type not in ["share_ssm"] else 1  # 扫描大小这里也要改
        self.K2 = self.K if forward_type not in ["share_a"] else 1
        # print(self.K)
        # in proj =======================================
        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
        if self.scan == "learning":
            # self.conv_scan=conv_scan(d_expand,1,3,1,1)
            # self.conv_merge=conv_merge(d_expand,1,3,1,1)
            self.conv_scan = conv_scan(d_expand, 32, 4, 4, 0)
            self.conv_merge = conv_merge(d_expand, 2, 3, 1, 1)
        else:
            self.scan_direction = None
            self.scan_direction_reverse = None

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        # print(len(self.x_proj))
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )  # (K, N, inner)
        del self.x_proj
        # print(self.x_proj_weight.shape)
        # dt proj ============================
        self.dt_projs = [
            self.dt_init(
                self.dt_rank,
                d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            )
            for _ in range(self.K)
        ]  # 8,256,1
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )  # (K, inner)
        del self.dt_projs

        # A, D =======================================
        self.A_logs = self.A_log_init(
            self.d_state, d_inner, copies=self.K2, merge=True
        )  # (K * D, N)
        self.Ds = self.D_init(d_inner, copies=self.K2, merge=True)  # (K * D)

        # out proj =======================================
        self.out_proj = nn.Linear(d_expand, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if simple_init:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((self.K2 * d_inner)))
            self.A_logs = nn.Parameter(
                torch.randn((self.K2 * d_inner, self.d_state))
            )  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32

        # print(A_log.shape)
        if copies > 0:

            A_log = repeat(A_log, "d n -> r d n", r=copies)
            # print(A_log.shape)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        # print(A_log.shape)
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    # only used to run previous version
    def forward_corev0(self, x: torch.Tensor, to_dtype=False, channel_first=False):
        def selective_scan(
            u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1
        ):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack(
            [x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
            dim=1,
        ).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float()  # (b, k, d_state, l)
        Cs = Cs.float()  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float())  # (k * d, d_state)
        Ds = self.Ds.float()  # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        # assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = (
            torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )
        invwh_y = (
            torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        return y.to(x.dtype) if to_dtype else y

    # only has speed difference with v0
    def forward_corev0_seq(self, x: torch.Tensor, to_dtype=False, channel_first=False):
        def selective_scan(
            u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1
        ):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack(
            [x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
            dim=1,
        ).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float()  # (b, k, d, l)
        dts = dts.contiguous().float()  # (b, k, d, l)
        Bs = Bs.float()  # (b, k, d_state, l)
        Cs = Cs.float()  # (b, k, d_state, l)

        As = -torch.exp(self.A_logs.float()).view(K, -1, self.d_state)  # (k, d, d_state)
        Ds = self.Ds.float().view(K, -1)  # (k, d)
        dt_projs_bias = self.dt_projs_bias.float().view(K, -1)  # (k, d)

        # assert len(xs.shape) == 4 and len(dts.shape) == 4 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 3 and len(Ds.shape) == 2 and len(dt_projs_bias.shape) == 2

        out_y = []
        for i in range(4):
            yi = selective_scan(
                xs[:, i],
                dts[:, i],
                As[i],
                Bs[:, i],
                Cs[:, i],
                Ds[i],
                delta_bias=dt_projs_bias[i],
                delta_softplus=True,
            ).view(B, -1, L)
            out_y.append(yi)
        out_y = torch.stack(out_y, dim=1)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = (
            torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )
        invwh_y = (
            torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3)
            .contiguous()
            .view(B, -1, L)
        )
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous()  # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        return y.to(x.dtype) if to_dtype else y

    def forward_corev0_share_ssm(self, x: torch.Tensor, channel_first=False):
        """
        we may conduct this ablation later, but not with v0.
        """
        ...

    def forward_corev0_share_a(self, x: torch.Tensor, channel_first=False):
        """
        we may conduct this ablation later, but not with v0.
        """
        ...

    def forward_corev2(self, x: torch.Tensor, nrows=-1, SNR=10, channel_first=False):

        nrows = 1
        if not channel_first:  # channel_first : True
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:  # False
            x = self.in_rank(x)
        # print("x.shape[2]",x.shaper[2])

        x = cross_selective_scan(
            x,
            self.x_proj_weight,
            None,
            self.dt_projs_weight,
            self.dt_projs_bias,
            self.A_logs,
            self.Ds,
            getattr(self, "out_norm", None),
            nrows=nrows,
            delta_softplus=True,
            scan=self.scan,
            scan_number=self.scan_number,
            adaptive=self.adaptive,
            snr=SNR,
        )

        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor, SNR, **kwargs):
        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
            if not self.disable_z_act:

                z = self.act((z))
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act((self.conv2d(x)))  # (b, d, h, w)
        else:
            if self.disable_z_act:
                x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
                x = self.act(x)
            else:
                xz = self.act(xz)
                x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        y = self.forward_core(x, channel_first=(self.d_conv > 1), SNR=SNR)

        y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class conv_scan(nn.Module):
    def __init__(self, C, num_expand, kernel=4, stride=4, padding=0):
        super().__init__()
        self.scan = nn.Conv2d(
            C, C * num_expand, kernel_size=kernel, stride=stride, padding=padding, groups=C
        )

    def forward(self, x):
        assert len(x.shape) == 4
        B, C, H, W = x.shape
        xs = x.new_empty(B, 4, C, H * W)
        xs[:, 0:2] = self.scan(x).view(B, 2, C, W * H)
        xs[:, 2:4] = xs[:, 0:2].flip(dims=[-1]).flip(dims=[-2])
        # print(xs.shape)
        #
        return xs


class conv_merge(nn.Module):
    def __init__(
        self,
        C,
        num_expand,
        kernel=3,
        stride=1,
        padding=1,
    ):
        super().__init__()
        self.merge = nn.Conv2d(
            C * num_expand, C, kernel_size=kernel, stride=stride, padding=padding, groups=C
        )

    def forward(self, x):
        assert len(x.shape) == 5
        B, D, C, H, W = x.shape
        xs = self.merge(x[:, 0:2].view(B, (D // 2) * C, H, W)).view(B, C, H * W)
        xs = self.merge(x[:, 2:4].flip(dims=[-1]).view(B, (D // 2) * C, H, W)).view(B, C, H * W)
        return xs


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        channels_first=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        # =============================
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_simple_init=False,
        forward_type="v2",
        # =============================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        # =============================
        use_checkpoint: bool = False,
        scan="cross",
        PE="no",
        resolution=128,
        scan_number=4,
        extent="no",
        channel_adaptive="no",
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.channel_adaptive = channel_adaptive

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                simple_init=ssm_simple_init,
                # ==========================
                forward_type=forward_type,
                scan=scan,
                PE=PE,
                resolution=resolution,
                scan_number=scan_number,
                adaptive=channel_adaptive,
            )
        self.extent = extent
        if extent == "no":
            pass
        elif extent == "MLP":
            self.norm_extent = norm_layer(hidden_dim)
            self.extent = Mlp(
                hidden_dim,
                hidden_dim,
                hidden_dim,
                act_layer=mlp_act_layer,
                drop=mlp_drop_rate,
                channels_first=False,
            )
        else:
            raise ValueError("extent method error")
        self.drop_path = DropPath(drop_path)

    def forward(self, y):
        x = y[0]
        SNR = y[1]

        x = x + self.drop_path(self.op(self.norm(x), SNR))
        if self.extent != "no":

            x = x + self.drop_path(self.extent(self.norm_extent(x)))

        return (x, SNR)
