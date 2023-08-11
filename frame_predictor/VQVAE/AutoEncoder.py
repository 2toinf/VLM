import nntplib
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

from VQVAE.quantize import VectorQuantizer2 as VectorQuantizer
from timm.models.registry import register_model
from VQVAE.EncoderDecoder import Encoder, Decoder
import numpy as np
import yaml


class VQModel(nn.Module):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False, # tell vector quantizer to return indices as bhw
                 use_ema=False,
                 **kwargs
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")
    
    def freeze(self):
        self.encoder = self.encoder.eval()
        self.decoder = self.decoder.eval()
        self.quantize = self.quantize.eval()
        for param in self.parameters():
            param.requires_grad = False


    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def get_ind(self, h):
        _, _, (_,_,ind) = self.quantize(h)
        return ind

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embedding(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_,_,ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

@register_model
def VQVAE(
        name= "vq-f4",
        **kwargs
    ):
    model_config = f"/mnt/lustre/zhengjinliang/vision-language-model/ldm/configs/{name}/config.yaml"
    pretrained = f"/mnt/lustre/zhengjinliang/vision-language-model/ldm/configs/{name}/model.ckpt"
    with open(model_config, "r") as f:
        file = yaml.load(f.read(), Loader=yaml.FullLoader)
    model_kwargs = file['model']['params']
    m = VQModel(**model_kwargs)
    if pretrained is not None:
        state = torch.load(pretrained, map_location='cpu')
        state = state['state_dict']
        m.load_state_dict(state, strict=False)
    return m


