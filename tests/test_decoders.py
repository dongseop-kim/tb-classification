import torch

from trainer.models.decoder.base import Identity
from trainer.models.decoder.upsample_concat import UpsampleConcat

INPUT_TENSOR = torch.randn(2, 64, 256, 256)
INPUT_LIST_TENSOR = [torch.randn(2, 64, 256, 256), torch.randn(2, 128, 128, 128),
                     torch.randn(2, 256, 64, 64), torch.randn(2, 512, 32, 32)]


def test_identity():
    decoder = Identity(in_channels=64, in_strides=4)
    output: torch.Tensor = decoder(INPUT_TENSOR)
    assert output.shape == INPUT_TENSOR.shape
    assert decoder.out_channels == 64
    assert decoder.out_strides == 4
    assert torch.allclose(output, INPUT_TENSOR, atol=1e-3)


def test_identity2():
    decoder = Identity(in_channels=[64, 128, 256, 512], in_strides=[4, 8, 16, 32])
    output: torch.Tensor = decoder(INPUT_LIST_TENSOR)
    assert len(output) == len(INPUT_LIST_TENSOR)
    assert list(decoder.out_channels) == [64, 128, 256, 512]
    assert list(decoder.out_strides) == [4, 8, 16, 32]
    for out, inp in zip(output, INPUT_LIST_TENSOR):
        assert torch.allclose(out, inp, atol=1e-3)


def test_upsampleconcat():
    decoder = UpsampleConcat(in_channels=[64, 128, 256, 512], in_strides=[4, 8, 16, 32])
    output: torch.Tensor = decoder(INPUT_LIST_TENSOR)
    assert output.shape == (2, 960, 256, 256)
    assert decoder.out_channels == 960
    assert decoder.out_strides == 4


def test_upsampleconcat2():
    decoder = UpsampleConcat(in_channels=[64, 128, 256, 512], in_strides=[4, 8, 16, 32], out_strides=8)
    output: torch.Tensor = decoder(INPUT_LIST_TENSOR)
    assert output.shape == (2, 960, 128, 128)
    assert decoder.out_channels == 960
    assert decoder.out_strides == 8
