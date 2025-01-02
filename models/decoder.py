import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
from models.vmamba import *

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SNR_embedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.SNRembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.zeros_(module.bias)

    def forward(self, SNR):
        
        emb1 = self.SNRembedding(SNR)
        return emb1
    
class AdaptiveModulator(nn.Module):
    def __init__(self, M):
        super(AdaptiveModulator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.Sigmoid()
        )

    def forward(self, snr):
        return self.fc(snr)

class Mamba_decoder(nn.Module):
    def __init__(
        self, 
        patch_size=2, 
        in_chans=332, 
        out_chans=36,
        depths=[2, 2, 9, 2], 
        dims=[768, 384, 192, 96], 
        # =========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_simple_init=False,
        forward_type="v2", 
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        # =========================
        drop_path_rate=0.1, 
        patch_norm=True, 
        norm_layer="LN",
        sample_version: str = "v2", # "v1", "v2", "v3"
        patchembed_version: str = "v1", # "v1", "v2"
        use_checkpoint=False,  
        channel_adaptive='CA',
        scan="cross",
        PE='no',
        img_resolution=256,
        scan_number=4,
        extent='no',
        channel_input='conv',
        **kwargs,
    ):
        super().__init__()
        
        depths=list(reversed(depths))
        self.num_layers = len(depths)
        # if isinstance(dims, int):
        #     dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        dims=list(reversed(dims))
        print("decoder dims:",dims)
        self.num_features = dims[0]
        self.dims = dims
        self.channel_adaptive=channel_adaptive
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        if norm_layer.lower() in ["ln"]:
            norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]

        if ssm_act_layer.lower() in ["silu", "gelu", "relu"]:
            ssm_act_layer: nn.Module = _ACTLAYERS[ssm_act_layer.lower()]

        if mlp_act_layer.lower() in ["silu", "gelu", "relu"]:
            mlp_act_layer: nn.Module = _ACTLAYERS[mlp_act_layer.lower()]

        # _make_patch_embed = dict(
        #     v1=self._make_patch_embed, 
        #     v2=self._make_patch_embed_v2,
        # ).get(patchembed_version, None)
        # self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer)

        _make_upsample = dict(
            v1=PatchReverseMerging2D
        ).get(sample_version, None)

        
        if self.channel_adaptive=='CA':
            self.SNR_embedding = SNR_embedding(25,dims[-1],dims[-1])
            self.proj_list=nn.ModuleList()
        elif self.channel_adaptive=='attn':
            self.hidden_dim = int(self.dims[0] * 1.5)
            self.layer_num = layer_num = 7
            self.bm_list = nn.ModuleList()
            self.sm_list = nn.ModuleList()
            self.sm_list.append(nn.Linear(self.dims[0], self.hidden_dim))
            for i in range(layer_num):
                if i == layer_num - 1:
                    outdim = self.dims[0]
                else:
                    outdim = self.hidden_dim
                self.bm_list.append(AdaptiveModulator(self.hidden_dim))
                self.sm_list.append(nn.Linear(self.hidden_dim, outdim))

            self.sigmoid = nn.Sigmoid()
        elif self.channel_adaptive=='ssm':
            pass
        elif self.channel_adaptive=='no':
            pass
        else:
            raise ValueError("channel adaptive error")
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            upsample = _make_upsample(
                self.dims[i_layer], 
                self.dims[i_layer + 1], 
                norm_layer=norm_layer,
            ) if (i_layer < self.num_layers - 1) else _make_upsample(
                self.dims[-1], 
                3, 
                norm_layer=norm_layer,
            )

            self.layers.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                upsample=upsample,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                SNR_dim=dims[-1],
                scan=scan,
                PE=PE,
                resolution=img_resolution//(2**(self.num_layers-i_layer)),
                scan_number=scan_number,
                extent=extent
            ))
        
        self.channel_input=channel_input
        if channel_input=='conv':
            self.head=nn.Conv2d(out_chans, dims[0],kernel_size=1, padding=0, stride=1)
        elif channel_input=='fc':
            self.head=nn.Linear(out_chans,dims[0])
        elif channel_input in ['ssm', 'ssm_revise']:
            self.out_norm=nn.LayerNorm(out_chans)
            self.head=nn.Conv2d(out_chans,dims[0], kernel_size=1, padding=0, stride=1)
            self.K=self.K2=1
            d_inner=out_chans
            self.dt_scale=1.0
            self.d_state=int((img_resolution//2**(4))**2//4)
            self.dt_rank=int(self.d_state//2)
            
            self.x_proj = [
                nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False,)
                for _ in range(self.K)
                ]
            self.out_norm = nn.LayerNorm(d_inner)
            #print(len(self.x_proj))
            self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
            del self.x_proj
            #print(self.x_proj_weight.shape)
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(self.dt_rank, d_inner, self.dt_scale,'random', 0.001, 0.1, 1e-4)
                for _ in range(self.K)
            ] # 8,256,1
            self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
            del self.dt_projs
            
            # A, D =======================================
            self.A_logs = self.A_log_init(self.d_state, d_inner, copies=self.K2, merge=True) # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=self.K2, merge=True) # (K * D)
#swin transformer 3,1,1
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # used in building optimizer
    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {}

    # used in building optimizer
    # @torch.jit.ignore
    # def no_weight_decay_keywords(self):
    #     return {}

    # @staticmethod
    # def _make_patch_embed(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
    #     return nn.Sequential(
    #         nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True),
    #         Permute(0, 2, 3, 1),
    #         (norm_layer(embed_dim) if patch_norm else nn.Identity()), 
    #     )

    #@staticmethod
    # def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
    #     assert patch_size == 4
    #     return nn.Sequential(
    #         nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
    #         (Permute(0, 2, 3, 1) if patch_norm else nn.Identity()),
    #         (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
    #         (Permute(0, 3, 1, 2) if patch_norm else nn.Identity()),
    #         nn.GELU(),
    #         nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
    #         Permute(0, 2, 3, 1),
    #         (norm_layer(embed_dim) if patch_norm else nn.Identity()),
    #     )
    
    # @staticmethod
    # def _make_downsample(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
    #     return nn.Sequential(
    #         Permute(0, 3, 1, 2),
    #         nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
    #         Permute(0, 2, 3, 1),
    #         norm_layer(out_dim),
    #     )

    # @staticmethod
    # def _make_downsample_v3(dim=96, out_dim=192, norm_layer=nn.LayerNorm):
    #     return nn.Sequential(
    #         Permute(0, 3, 1, 2),
    #         nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1),
    #         Permute(0, 2, 3, 1),
    #         norm_layer(out_dim),
    #     )

    
    def _make_layer(self,
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        upsample=nn.Identity(),
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_rank_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_simple_init=False,
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        SNR_dim=96,
        scan="cross",
        PE='no',
        resolution=128,
        scan_number=4,
        extent='no',
        **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        
        if self.channel_adaptive=='CA':
            self.proj_list.append(nn.Linear(SNR_dim, dim))

        for d in range(depth):
            blocks.append(VSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_simple_init=ssm_simple_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
                scan=scan,
                PE=PE,
                resolution=resolution,
                scan_number=scan_number,
                extent=extent,
                channel_adaptive=self.channel_adaptive
            ))
        
        return nn.Sequential(OrderedDict(
            blocks=nn.Sequential(*blocks,),
            upsample=upsample,
        ))
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        #print(dt_rank,dt_scale)
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

        #print(A_log.shape)
        if copies > 0:

            A_log = repeat(A_log, "d n -> r d n", r=copies)
            #print(A_log.shape)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        #print(A_log.shape)
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
    
    def forward(self,x: torch.Tensor, SNR):
        # if self.channel_input in ['fc', 'conv']:
        x = self.head(x)
        # elif self.channel_input in ['ssm', 'ssm_revise']: 
            
        #     pwr=cross_selective_scan(
        #         x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
        #         self.A_logs, self.Ds, getattr(self, "out_norm", None),
        #         nrows=-1, delta_softplus=True,scan='cross',scan_number=1,
        #     )
        #     if self.channel_input=='ssm_revise':
        #         pwr=pwr.permute(0,3,1,2)+x
        #     elif self.channel_input=='ssm':
        #         pwr=pwr.permute(0,3,1,2)
        #     #print(pwr.shape)
        #     #print(x.shape)
        #     #x=nn.SiLU()(pwr)*x
        #     x=self.head(pwr)
        # #x = self.head(x)
        # else:
        #     raise ValueError("channel input error")
        x=x.permute(0,2,3,1)
        if self.channel_adaptive == "CA":
            SNR_embed = x.new_ones([x.shape[0], ], dtype=torch.long)*SNR
            SNR_embedding=self.SNR_embedding(SNR_embed)
            for layer, proj in zip(self.layers, self.proj_list):
                emb=proj(SNR_embedding)[:,None,None,:]
                x=x+emb
            #print(proj(SNR_embedding))
            #print(layer)
                x = layer((x,SNR))
        elif self.channel_adaptive == "attn":
            B, H, W, C=x.shape
            x=x.flatten(1,2)
            snr_cuda = torch.tensor(SNR, dtype=torch.float).cuda()
            snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
            for i in range(self.layer_num):
                if i == 0:
                    temp = self.sm_list[i](x.detach())
                else:
                    temp = self.sm_list[i](temp)

                bm = self.bm_list[i](snr_batch).unsqueeze(1).expand(-1, H*W, -1)
                #print(temp.shape)
                #print(bm.shape)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val
            b,l,c=x.shape
            x=x.view(B, int(math.sqrt(l)), int(math.sqrt(l)), -1)
            for layer in self.layers:
                x = layer((x,SNR))
        else:
            for layer in self.layers:
            #print(x.shape)
                x = layer((x,SNR))
        #print(x.shape)
        #x=x[0]
        x=x.permute(0,3,1,2)

        return x

    def flops(self, shape=(3, 224, 224)):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScan": selective_scan_flop_jit,
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        return sum(Gflops.values()) * 1e9
        return f"params {params} GFLOPs {sum(Gflops.values())}"

    # used to load ckpt from previous training code
    # def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):

    #     def check_name(src, state_dict: dict = state_dict, strict=False):
    #         if strict:
    #             if prefix + src in list(state_dict.keys()):
    #                 return True
    #         else:
    #             key = prefix + src
    #             for k in list(state_dict.keys()):
    #                 if k.startswith(key):
    #                     return True
    #         return False

    #     def change_name(src, dst, state_dict: dict = state_dict, strict=False):
    #         if strict:
    #             if prefix + src in list(state_dict.keys()):
    #                 state_dict[prefix + dst] = state_dict[prefix + src]
    #                 state_dict.pop(prefix + src)
    #         else:
    #             key = prefix + src
    #             for k in list(state_dict.keys()):
    #                 if k.startswith(key):
    #                     new_k = prefix + dst + k[len(key):]
    #                     state_dict[new_k] = state_dict[k]
    #                     state_dict.pop(k)

    #     change_name("patch_embed.proj", "patch_embed.0")
    #     change_name("patch_embed.norm", "patch_embed.2")
    #     for i in range(100):
    #         for j in range(100):
    #             change_name(f"layers.{i}.blocks.{j}.ln_1", f"layers.{i}.blocks.{j}.norm")
    #             change_name(f"layers.{i}.blocks.{j}.self_attention", f"layers.{i}.blocks.{j}.op")
    #     change_name("norm", "classifier.norm")
    #     change_name("head", "classifier.head")

    #     return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

def create_decoder(config):
    decoder_kwargs=dict(        
        patch_size=config.MODEL.VSSM.PATCH_SIZE, 
        in_chans=config.MODEL.VSSM.IN_CHANS, 
        out_chans=config.MODEL.VSSM.OUT_CHANS,
        depths=config.MODEL.VSSM.DEPTHS, 
        dims=config.MODEL.VSSM.EMBED_DIM, 
        # ===================
        ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
        ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
        ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
        ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
        ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
        ssm_conv=config.MODEL.VSSM.SSM_CONV,
        ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
        ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
        ssm_simple_init=config.MODEL.VSSM.SSM_SIMPLE_INIT,
        forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
        # ===================
        mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
        mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
        mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
        # ===================
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        patch_norm=config.MODEL.VSSM.PATCH_NORM,
        norm_layer=config.MODEL.VSSM.NORM_LAYER,
        sample_version=config.MODEL.VSSM.DOWNSAMPLE,
        patchembed_version=config.MODEL.VSSM.PATCHEMBED,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        channel_adaptive=config.CHANNEL.ADAPTIVE,
        scan=config.MODEL.VSSM.SCAN,
        PE=config.MODEL.VSSM.PE,
        img_resolution=config.DATA.IMG_SIZE,
        scan_number=config.MODEL.VSSM.SCAN_NUMBER,
        extent=config.MODEL.VSSM.Extent,
        channel_input=config.MODEL.VSSM.channel_input)
    
    model = Mamba_decoder(**decoder_kwargs)
    return model

