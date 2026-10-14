"""U-Net (ResNet-18 encoder) for SAR water segmentation. 25% of the score lives here."""
from __future__ import annotations

# 6 input channels: pre_VV, pre_VH, post_VV, post_VH (dB) + diff_VV, diff_VH (post - pre, dB)
IN_CHANNELS = 6


def build_model(encoder: str = "resnet18", in_channels: int = IN_CHANNELS, pretrained: bool = False):
    """pretrained=False by default: ImageNet weights are a pretrained model whose training data is
    NOT in the permitted-dataset list; the plan says to exclude such models unless verified.
    Training from scratch on Kuro Siwo is the safe default."""
    import segmentation_models_pytorch as smp
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                    in_channels=in_channels, classes=1)


def make_input(pre_vv_db, pre_vh_db, post_vv_db, post_vh_db):
    """Stack channels (numpy, dB). Shared by training AND inference => preprocessing parity."""
    import numpy as np
    return np.stack([pre_vv_db, pre_vh_db, post_vv_db, post_vh_db,
                     post_vv_db - pre_vv_db, post_vh_db - pre_vh_db]).astype("float32")
