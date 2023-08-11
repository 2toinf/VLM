import nntplib
from turtle import forward
from fsspec import register_implementation
from sklearn.feature_selection import SelectFdr
import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp,DropPath,trunc_normal_
from einops import rearrange
from timm.models.registry import register_model

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


class Decoder(nn.Module):
    def __init__(self,
            embed_dim = 768,
            depth = 6,
            drop_path_rate=0,
            mlp_ratio=4.,
            num_heads=12
        ) -> None:
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            DecoderBlock(
                d_model=embed_dim, 
                nhead=num_heads, 
                mlp_ratio=mlp_ratio, 
                context_dim=embed_dim,
                drop_path=dpr[i],
                use_context=True
            )
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, query, value):
        for blk in self.blocks:
            query = blk(query, value)
        query = self.norm(query)
        return query


class VectorQuantizer(nn.Module):

    def __init__(self, n_e, e_dim, beta = 0.25):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

    def forward(self, z):
        '''
            z: Tensor(B, N, C)
        '''
        z_flattened = z.view(-1, self.e_dim)

        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(self.embedding.weight, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        return z_q, loss, min_encoding_indices


class VQImgTranslator(nn.Module):
    def __init__(
        self,
        codebook_size = 256,
        trans_codebook_size = 256,
        query_num = 32,
        embed_dim = 768,
        input_length = 256,
        encoder_depth = 6,
        decoder_depth = 6,
        drop_path_rate = 0.,
        mlp_ratio = 4., 
        num_heads=12,
    ) -> None:
        super().__init__()
        self.query_num = query_num
        self.embed_dim = embed_dim
        self.input_length = input_length
        self.code_embed = nn.Embedding(codebook_size, embed_dim)
        self.init_embed = nn.Parameter(torch.randn(1, 1, embed_dim)* .02)
        self.tar_embed = nn.Parameter(torch.randn(1, 1, embed_dim) * .02)
        self.pos_embed = nn.Parameter(torch.randn(1, input_length, embed_dim) *.02)
        self.query = nn.Parameter(torch.zeros(1, query_num, embed_dim))
        self.head = torch.nn.Linear(embed_dim, codebook_size)

        self.Encoder = Decoder(
            embed_dim=embed_dim,
            depth = encoder_depth,
            drop_path_rate=drop_path_rate,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads
        )
        self.Decoder = Decoder(
            embed_dim=embed_dim,
            depth = decoder_depth,
            drop_path_rate=drop_path_rate,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads
        )
        self.Quantizer = VectorQuantizer(
            n_e = trans_codebook_size,
            e_dim = embed_dim
        )
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
    
    def encode(self, init_emb, tar_emb):
        '''
            shape: B*L || B, L
        '''
        
        init_emb = init_emb + self.init_embed
        tar_emb = tar_emb + self.tar_embed
        B, _, _ = init_emb.shape
        query = self.query.repeat(B, 1, 1)
        query = self.Encoder(query, torch.cat((init_emb,tar_emb), dim = 1))

        z_q, loss, min_encoding_indices = self.Quantizer(query)
        return z_q, loss, min_encoding_indices


    def decode(self, init_emb, trans_emb):
        pred_emb = self.Decoder(init_emb, trans_emb)
        return self.head(pred_emb)

    def decode_indices(self, init_indices, trans_indices):
        init_emb = self.code_embed(init_indices)
        if len(init_indices.shape) == 1: 
            init_emb = init_emb.reshape(-1, self.input_length, self.embed_dim)
        init_emb = init_emb + self.pos_embed

        trans_emb = self.Quantizer.embedding(trans_indices)
        if len(trans_indices.shape) == 1:
            trans_emb = trans_emb.reshape(-1, self.query_num)
        return self.decode(init_emb, trans_emb)

    
    def forward_train(self, init_indices, tar_indices):

        init_emb = self.code_embed(init_indices)
        tar_emb = self.code_embed(tar_indices)
        if len(init_indices.shape) == 1: init_emb = init_emb.reshape(-1, self.input_length, self.embed_dim)
        if len(tar_indices.shape) == 1: tar_emb = tar_emb.reshape(-1, self.input_length, self.embed_dim)
        init_emb = init_emb + self.pos_embed 
        tar_emb = tar_emb + self.pos_embed

        z_q, quant_loss, _ = self.encode(init_emb, tar_emb)

        pred_logits = self.decode(init_emb, z_q)
        return pred_logits, quant_loss
    
    def forward(self, init_indices, tar_indices):
        return self.forward_train(init_indices, tar_indices)

@register_model
def vq_translator_base(pretrained = None, **kwargs):
    m = VQImgTranslator(
        codebook_size=256,
        trans_codebook_size=256,
        query_num = 32,
        embed_dim = 768,
        input_length = 256,
        encoder_depth = 6,
        decoder_depth = 6,
        drop_path_rate = 0.,
        mlp_ratio = 4., 
        num_heads=12,
    )
    return m

from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
if __name__ == "__main__":
    model = vq_translator_base()
    print(model)
    model.eval()
    i = torch.rand(1, model.input_length).long()
    flops = FlopCountAnalysis(model, (i, i))
    print(flop_count_table(flops))



