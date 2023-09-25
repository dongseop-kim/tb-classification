from typing import List, Union

from .linear import LinearDecoder
from .upsamplecat import UpsampleCat, UpsampleCatwithConv

available_decoders = {'linear': LinearDecoder,
                      'upsamplecat': UpsampleCat,
                      'upsamplecatwithconv': UpsampleCatwithConv,
                      }


def build_decoder(name: str,
                  in_channels: Union[int, List[int]],
                  in_strides: Union[int, List[int]],
                  **kwargs):
    return available_decoders[name](in_channels=in_channels,
                                    in_strides=in_strides,
                                    **kwargs)
