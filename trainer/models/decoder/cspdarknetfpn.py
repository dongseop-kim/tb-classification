# from typing import List

# import torch
# import torch.nn as nn

# from ..base_module import ConvBNSwish, ConvBN, Concat


# class ConvBlock(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, expansion: float = 1.0):
#         super().__init__()
#         mid_channels = int(out_channels * expansion)
#         self.conv1 = ConvBNSwish(in_channels, mid_channels, 1, 1, 0)
#         self.conv2 = ConvBNSwish(mid_channels, out_channels, 3, 1, 1)

#         self.residual = nn.Identity()
#         if in_channels != out_channels:
#             self.residual = ConvBN(in_channels, out_channels, 1, 1, 0)

#     def forward(self, x: torch.Tensor):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.residual(x) + out
#         return out


# class CSPLayer(nn.Module):
#     def __init__(self,
#                  in_channels: int,
#                  out_channels: int,
#                  repeat: int = 1,
#                  expansion: float = 0.5):
#         """
#         Args:
#             in_channels (int): input channels.
#             out_channels (int): output channels.
#             repeat (int): number of Bottlenecks. Default value: 1.
#         """
#         super().__init__()
#         mid_channels = int(out_channels * expansion)  # hidden channels
#         self.conv_l = nn.Sequential(ConvBNSwish(in_channels, mid_channels, 1, 1, 0),
#                                     nn.Sequential(*[ConvBlock(mid_channels, mid_channels, 1.0) for _ in range(repeat)]))
#         self.conv_r = ConvBNSwish(in_channels, mid_channels, 1, 1, 0)
#         self.fusion = nn.Sequential(Concat(dim=1), ConvBNSwish(2 * mid_channels, out_channels, 1, 1, 0))

#     def forward(self, x: torch.Tensor):
#         x_l = self.conv_l(x)  # left branch
#         x_r = self.conv_r(x)  # right branch
#         out = self.fusion([x_l, x_r])  # aggregate each branch
#         return out


# class CSPDarknetFPN(nn.Module):
#     """
#     this module uses only last 3 feature maps that output from backbone
#     Args:
#         in_channels (list[int]): number of channels for each feature map that
#             is passed to the module
#         strides (list[int]): stride of each feature map that is passed to the module
#         out_channels (int): number of channels output by the module
#     """

#     def __init__(self,
#                  in_channels: List[int],
#                  strides: List[int],
#                  out_channels: int = 512,
#                  **kwargs):
#         super().__init__()

#         # reduction 3
#         reduction3 = [ConvBNSwish(in_channels[-1], in_channels[-2], 1, 1, 0)]
#         if int(strides[-2] // strides[-1]) != 1:
#             reduction3.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#         self.reduction3 = nn.Sequential(*reduction3)

#         # reduction 2
#         reduction2 = [CSPLayer(in_channels[-2] * 2, in_channels[-2], 3),
#                       ConvBNSwish(in_channels[-2], in_channels[-3], 1, 1, 0)]
#         if int(strides[-3] // strides[-2]) != 1:
#             reduction2.append(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True))
#         self.reduction2 = nn.Sequential(*reduction2)

#         # reduction 1
#         self.reduction1 = CSPLayer(in_channels[-3] * 2, out_channels, 3)
#         self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
#         self.out_channels = out_channels
#         self.out_stride = strides[-3] // 2

#     def forward(self, features: List[torch.Tensor]):
#         # channels[-1] -> channels[-2]
#         # reductions[-1] -> reductions[-2]
#         out3 = self.reduction3(features[-1])

#         # channels[-2]*2 -> channels[-3]
#         # reductions[-2] -> reductions[-3]
#         out2 = self.reduction2(torch.cat((features[-2], out3), dim=1))

#         # channels[-3]*2 -> channels[-3]
#         # reductions[-3], no upsample
#         out1 = self.reduction1(torch.cat((features[-3], out2), dim=1))

#         return self.upsample(out1)
