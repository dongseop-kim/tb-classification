from typing import Dict

import timm
import torch
import torch.nn as nn

from trainer.models.decoder import BaseDecoder, build_decoder


def build_encoder(name: str, pretrained=True, **kwargs) -> timm.models._features.FeatureListNet:
    timm_list = timm.list_models(pretrained)
    assert name in timm_list, f'Unknown encoder name: {name}'
    return timm.create_model(model_name=name, pretrained=pretrained,
                             features_only=True, **kwargs)


class Model(nn.Module):
    def __init__(self,
                 num_classes: int,
                 encoder: Dict[str, any] = None,
                 decoder: Dict[str, any] = None,
                 header: Dict[str, any] = None):
        super().__init__()
        self.num_classes = num_classes

        self.encoder: nn.Module = build_encoder(**encoder)
        self.decoder: BaseDecoder = build_decoder(in_channels=self.encoder.feature_info.channels(),
                                                  in_strides=self.encoder.feature_info.reduction(),
                                                  **decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor | Dict[str, torch.Tensor]:
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.header(x)
        return x

    def load_weights(self, path: str, unwarp_key: str = 'model.'):
        weights: Dict = torch.load(path, map_location='cpu')
        weights: Dict = weights['state_dict'] if 'state_dict' in weights.keys() else weights
        weights = {key.replace(unwarp_key, ''): weight for key, weight in weights.items()}
        return self.load_state_dict(weights, strict=True)
