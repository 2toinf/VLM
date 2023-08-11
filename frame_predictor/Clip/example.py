import torch
import clip_text
import model_vit_context_ft

import timm
from timm.models import create_model


def main():
    device = 'cuda'
    clip_model = clip_text.FrozenCLIPEmbedder(device=device)
    backbone = create_model(
        'vit_context_base_c10', 
        context_drop_prob=0.1,
    )

    ckpt = torch.load('/mnt/lustre/liujihao/cache_ckpts/diffusion_vl/exp/mae_base_ori_dec_context_enc_c10_cdp5_full/ckpt_amp_500m_10ep/checkpoint.pth')
    ckpt = ckpt['model']
    new_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith('backbone.'):
            new_ckpt[k.replace('backbone.', '')] = v
    msg = backbone.load_state_dict(new_ckpt, strict=False)
    print(msg)

    backbone = backbone.to(device)
    clip_model = clip_model.to(device)
    backbone.eval()
    clip_model.eval()

    
    img = torch.randn(1, 3, 224, 224).to(device)
    caption = 'a dog'

    text_feat = clip_model(caption)

    out = backbone(img, context=text_feat)

    print(out.shape)


if __name__ == '__main__':
    main()