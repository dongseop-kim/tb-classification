import torch

from trainer.models.decoder.deeplabv3 import DeepLabV3, DeepLabV3Plus
from trainer.models.decoder.fpn import FeaturePyramidNetwork
from trainer.models.decoder.linear import LinearDecoder
from trainer.models.decoder.upsamplecat import UpsampleCat, UpsampleCatwithConv

_INPUT_TENSOR = torch.randn(2, 64, 256, 256)
_INPUT_LIST_TENSOR = [torch.randn(2, 64, 256, 256), torch.randn(2, 128, 128, 128),
                      torch.randn(2, 256, 64, 64), torch.randn(2, 512, 32, 32)]


def test_linear_decoder():
    decoder = LinearDecoder(in_channels=64, in_strides=4)
    output: torch.Tensor = decoder(_INPUT_TENSOR)
    assert output.shape == _INPUT_TENSOR.shape
    assert decoder.out_channels == 64
    assert decoder.out_strides == 4
    assert torch.allclose(output, _INPUT_TENSOR, atol=1e-3)


def test_linear_decoder2():
    decoder = LinearDecoder(in_channels=[64, 128, 256, 512], in_strides=[4, 8, 16, 32])
    output: torch.Tensor = decoder(_INPUT_LIST_TENSOR)
    assert len(output) == len(_INPUT_LIST_TENSOR)
    assert decoder.out_channels == [64, 128, 256, 512]
    assert decoder.out_strides == [4, 8, 16, 32]
    for out, inp in zip(output, _INPUT_LIST_TENSOR):
        assert torch.allclose(out, inp, atol=1e-3)


def test_upsamplecat():
    decoder = UpsampleCat(in_channels=[64, 128, 256, 512], in_strides=[4, 8, 16, 32])
    output: torch.Tensor = decoder(_INPUT_LIST_TENSOR)
    assert output.shape == (2, 960, 256, 256)
    assert decoder.out_channels == 960
    assert decoder.out_strides == 4


def test_upsamplecat2():
    decoder = UpsampleCat(in_channels=[64, 128, 256, 512], in_strides=[4, 8, 16, 32], out_strides=8)
    output: torch.Tensor = decoder(_INPUT_LIST_TENSOR)
    assert output.shape == (2, 960, 128, 128)
    assert decoder.out_channels == 960
    assert decoder.out_strides == 8


def test_upsamplecatwithconv():
    decoder = UpsampleCatwithConv(in_channels=[64, 128, 256, 512], in_strides=[4, 8, 16, 32], out_channels=128)
    output: torch.Tensor = decoder(_INPUT_LIST_TENSOR)
    assert output.shape == (2, 128, 256, 256)
    assert decoder.out_channels == 128


def test_upsamplecatwithconv2():
    decoder = UpsampleCatwithConv(in_channels=[64, 128, 256, 512], in_strides=[4, 8, 16, 32],
                                  out_channels=128, out_strides=8)
    output: torch.Tensor = decoder(_INPUT_LIST_TENSOR)
    assert output.shape == (2, 128, 128, 128)
    assert decoder.out_channels == 128


def test_fpn():
    decoder = FeaturePyramidNetwork(in_channels=[64, 128, 256, 512], in_strides=[4, 8, 16, 32],
                                    out_channels=128, feature_aggregation=False)
    output: torch.Tensor = decoder(_INPUT_LIST_TENSOR)
    assert len(output) == len(_INPUT_LIST_TENSOR)
    assert decoder.out_channels == 128
    assert decoder.out_strides == [4, 8, 16, 32]
    assert output[0].shape == (2, 128, 256, 256)
    assert output[1].shape == (2, 128, 128, 128)
    assert output[2].shape == (2, 128, 64, 64)
    assert output[3].shape == (2, 128, 32, 32)


def test_fpn2():
    decoder = FeaturePyramidNetwork(in_channels=[128, 256, 512], in_strides=[8, 16, 32],
                                    out_channels=256, feature_aggregation=True)
    output: torch.Tensor = decoder(_INPUT_LIST_TENSOR[1:])
    assert output.shape == (2, 768, 128, 128)
    assert decoder.out_channels == 768

    decoder = FeaturePyramidNetwork(in_channels=[64, 128, 256, 512], in_strides=[4, 8, 16, 32],
                                    out_channels=256, feature_aggregation=True, out_strides=16)
    output: torch.Tensor = decoder(_INPUT_LIST_TENSOR)
    assert output.shape == (2, 1024, 64, 64)
    assert decoder.out_channels == 1024


def test_deeplabv3():
    decoder = DeepLabV3(in_channels=64, in_strides=4, out_channels=128)
    output: torch.Tensor = decoder(_INPUT_TENSOR)
    assert output.shape == (2, 128, 256, 256)
    assert decoder.out_channels == 128
    assert decoder.out_strides == 4


def test_deeplabv3_2():
    # this code raise warning
    decoder = DeepLabV3(in_channels=[64, 128, 256, 512], in_strides=[4, 8, 16, 32], out_channels=128)
    output: torch.Tensor = decoder(_INPUT_LIST_TENSOR)
    assert output.shape == (2, 128, 32, 32)
    assert decoder.out_channels == 128
    assert decoder.out_strides == 32


def test_deeplabv3plus():
    x = [torch.randn(2, 64, 256, 256), torch.randn(2, 128, 128, 128),
         torch.randn(2, 256, 64, 64), torch.randn(2, 512, 64, 64)]

    decoder = DeepLabV3Plus(in_channels=[64, 128, 256, 512], in_strides=[4, 8, 16, 16],
                            out_channels=128)

    output: torch.Tensor = decoder(x)
    assert output.shape == (2, 128, 256, 256)
    assert decoder.out_channels == 128
    assert decoder.out_strides == 4


def test_deeplabv3plus_2():
    x = [torch.randn(2, 64, 256, 256), torch.randn(2, 128, 128, 128),
         torch.randn(2, 256, 64, 64), torch.randn(2, 512, 64, 64)]

    decoder = DeepLabV3Plus(in_channels=[64, 128, 256, 512], in_strides=[4, 8, 16, 16],
                            out_channels=512)

    output: torch.Tensor = decoder(x)
    assert output.shape == (2, 512, 256, 256)
    assert decoder.out_channels == 512
    assert decoder.out_strides == 4
