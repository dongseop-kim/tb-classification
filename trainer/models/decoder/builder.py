from .linear import LinearDecoder
from .upsamplecat import UpsampleCat

available_decoders = {"linear": LinearDecoder,
                      "upsamplecat": UpsampleCat}


def build_decoder(name, **kwargs):
    return available_decoders[name](**kwargs)
