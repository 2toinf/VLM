from turtle import forward
from cv2 import EVENT_MOUSEMOVE
from sqlalchemy import true
import torch
import torch.nn as nn
import numpy as np
import torch
import clip
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from timm.models.registry import register_model
from timm.models.vision_transformer import Mlp, DropPath, PatchEmbed, trunc_normal_

class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, mlp_ratio=4., context_dim=768,
                 attn_drop=0.0, proj_drop=0.0, drop_path=0.0, use_context=True):
        super().__init__()
        
        self.use_context = use_context
        if self.use_context:
            self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_drop, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=attn_drop, batch_first=True)

        dim_feedforward = int(mlp_ratio * d_model)
        self.mlp = Mlp(in_features=d_model, hidden_features=dim_feedforward, act_layer=nn.GELU, drop=proj_drop)

        if self.use_context:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm1_kv = nn.LayerNorm(context_dim)
            self.to_k = nn.Linear(context_dim, d_model)
            self.to_v = nn.Linear(context_dim, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, tgt, memory):
        
        if self.use_context:
            memory = self.norm1_kv(memory)
            k = self.to_k(memory)
            v = self.to_v(memory)
            tgt2 = self.cross_attn(
                self.norm1(tgt),
                k,
                v
            )[0]
            # tgt = tgt + tgt2
            tgt = tgt + tgt2

        tgt_norm = self.norm2(tgt)
        q = k = v = tgt_norm
        tgt2 = self.self_attn(q+1e-8, k, v)[0]
        tgt = tgt + tgt2

        # ffn
        tgt = tgt + self.drop_path(self.mlp(self.norm3(tgt)))
        return tgt


class ConditionalTranslator(nn.Module):
    def __init__(self,
                in_chans,
                input_length,
                depth = 12,
                embed_dim = 768,
                context_dim = 768,
                mlp_ratio=4.,
                num_heads=12,
                drop_path_rate=0,
                use_cls_token = False
        ):
        super().__init__()
        self.in_chans = in_chans
        self.input_length = input_length
        self.embed_dim = embed_dim
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_cls_token else None
        embed_len = input_length + (1 if use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            DecoderBlock(
                d_model=embed_dim, 
                nhead=num_heads, 
                mlp_ratio=mlp_ratio, 
                context_dim=context_dim,
                drop_path=dpr[i],
                use_context=True
            )
            for i in range(depth)])
        

        self.input_proj = torch.nn.Conv2d(in_chans, embed_dim,1)
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = torch.nn.Conv2d(embed_dim, in_chans,1)
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


    def forward(self, x, context):
        '''
            x: Tensor(B, C, H, W)


        '''
        B, C, H, W = x.shape
        
        x = self.input_proj(x).reshape(B, self.embed_dim, H * W).permute(0, 2, 1)


        if self.cls_token is not None:
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim = 1)
        
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x, context)
        
        x = self.norm(x)

        x = x.permute(0, 2, 1).reshape(B, self.embed_dim, H, W)
        x = self.output_proj(x)

        return x




@register_model
def translator_base(in_chans = 4,
                    input_length = 256,
                    **kwargs):
    model = ConditionalTranslator(
        in_chans=in_chans,
        input_length=input_length,
        embed_dim=768,
        depth=9,
        num_heads=12,
        mlp_ratio=4.,
    )
    return model

@register_model
def translator_f4(in_chans = 3,
                    input_length = 1024,
                    **kwargs):
    model = ConditionalTranslator(
        in_chans=in_chans,
        input_length=input_length,
        embed_dim=768,
        depth=6,
        num_heads=12,
        mlp_ratio=4.,
    )
    return model



class ConditionalTranslatorKL(nn.Module):
    def __init__(self,
                input_length,
                depth = 12,
                embed_dim = 768,
                context_dim = 768,
                mlp_ratio=4.,
                num_heads=12,
                drop_path_rate=0,
                use_cls_token = False,
                codebook_size = 256
        ):
        super().__init__()
        self.input_length = input_length
        self.embed_dim = embed_dim
        self.codebook_size = codebook_size
        self.code_emb = nn.Embedding(codebook_size, embed_dim)
        embed_len = input_length + (1 if use_cls_token else 0)
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            DecoderBlock(
                d_model=embed_dim, 
                nhead=num_heads, 
                mlp_ratio=mlp_ratio, 
                context_dim=context_dim,
                drop_path=dpr[i],
                use_context=True
            )
            for i in range(depth)])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, codebook_size)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


    def forward(self, indices, context):
        '''
            indices: Tensor(B*L)


        '''
        x = self.code_emb(indices).reshape(-1, self.input_length, self.embed_dim)
        B = x.shape[0]
        
        x = x + self.pos_embed

        for blk in self.blocks:
            x = blk(x, context)
        
        x = self.norm(x)
        x = self.head(x).reshape(B * self.input_length, self.codebook_size)
        return x

@register_model
def translator_kl_base(
                    input_length = 256,
                    **kwargs):
    model = ConditionalTranslatorKL(
        input_length=input_length,
        embed_dim=768,
        depth=9,
        num_heads=12,
        mlp_ratio=4.,
    )
    return model

@register_model
def translator_kl_large(
                    input_length = 256,
                    **kwargs):
    model = ConditionalTranslatorKL(
        input_length=input_length,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
    )
    return model


@register_model
def translator_kl_f4(
                    input_length = 1024,
                    **kwargs):
    model = ConditionalTranslatorKL(
        input_length=input_length,
        embed_dim=768,
        depth=6,
        num_heads=12,
        mlp_ratio=4.,
        codebook_size = 8192
    )
    return model

@register_model
def translator_kl_8_256(
                    input_length = 1024,
                    **kwargs):
    model = ConditionalTranslatorKL(
        input_length=input_length,
        embed_dim=768,
        depth=6,
        num_heads=12,
        mlp_ratio=4.,
        codebook_size = 256
    )
    return model

@register_model
def translator_kl_8_256_large(
                    input_length = 1024,
                    **kwargs):
    model = ConditionalTranslatorKL(
        input_length=input_length,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        codebook_size = 256
    )
    return model