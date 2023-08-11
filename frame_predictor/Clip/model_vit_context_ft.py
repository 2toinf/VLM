import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from timm.models.registry import register_model
from timm.models.vision_transformer import Mlp, DropPath, PatchEmbed, trunc_normal_

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DecoderBlock(nn.Module):
    def __init__(self, d_model, nhead, mlp_ratio=4., context_dim=768,
                 attn_drop=0.0, proj_drop=0.0, drop_path=0.0, use_context=True,
                 context_drop_prob=0.0):
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
        self.c_drop_p = context_drop_prob

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
            tgt = tgt + drop_path(tgt2, self.c_drop_p, training=self.training)

        tgt_norm = self.norm2(tgt)
        q = k = v = tgt_norm
        tgt2 = self.self_attn(q+1e-8, k, v)[0]
        tgt = tgt + tgt2

        # ffn
        tgt = tgt + self.drop_path(self.mlp(self.norm3(tgt)))
        return tgt


class VitContext(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool=False,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.,
        class_token=True,
        drop_path_rate=0.,
        context_drop_prob=0.1,
        weight_init='',
        embed_layer=PatchEmbed,
        use_context_layer=-1,
        **kwargs,
    ):
        super().__init__()
        use_context_layer = use_context_layer or depth

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=True, 
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches + 1
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            DecoderBlock(
                d_model=embed_dim, 
                nhead=num_heads, 
                mlp_ratio=mlp_ratio, 
                context_dim=768,
                drop_path=dpr[i],
                use_context=i >= depth - use_context_layer,
                context_drop_prob=context_drop_prob
            )
            for i in range(depth)])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = nn.LayerNorm(embed_dim)
            del self.norm  # remove the original norm

        trunc_normal_(self.pos_embed, std=.02)
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
    
    def forward(self, x, context=None, use_mask=False, mask_ratio=0.75):
        B = x.shape[0]
        x = self.patch_embed(x)

        x = x + self.pos_embed[:, 1:, :]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = cls_tokens + self.pos_embed[:, :1, :]
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x, context)
        
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        x = self.head(outcome)

        return x

@register_model
def vit_context_base_c10(**kwargs):
    model = VitContext(patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                       mlp_ratio=4, use_context_layer=10, **kwargs)
    return model
