# from typing import Any, Dict, Optional, Union, List

# from ._base import BaseClassification

# _DATA_PATH = {'shenzhen': '/team/team_pxi/pxi-dataset/cxr/public/tb_shenzhen/tb_shenzhen_v1.2.json',
#               'montgomery': '/team/team_pxi/pxi-dataset/cxr/public/tb_montgomery/tb_montgomery_v1.2.json',
#               'tbx11k': '/team/team_pxi/pxi-dataset/cxr/public/tbx11k/tbx11k_v1.2.json',
#               }


# class CXRTBShenzhen(BaseClassification):
#     def __init__(self,
#                  data_dir,
#                  split: str,
#                  transforms: Optional[Dict[str, Dict]] = None,
#                  fold_val: int = -1,
#                  lesion_classes: List[int] = None,
#                  use_low_image: bool = False,
#                  do_windowing: bool = True,
#                  do_standardization: bool = True,
#                  mean: float = 0.0,
#                  std: float = 1.0):
#         super().__init__(data_dir=data_dir,
#                          split=split,
#                          transforms=transforms,
#                          fold_val=fold_val,
#                          lesion_classes=lesion_classes,
#                          use_low_image=use_low_image,
#                          do_windowing=do_windowing,
#                          do_standardization=do_standardization,
#                          mean=mean,
#                          std=std,
#                          annotation_json_fname=_DATA_PATH['shenzhen'])
#         if self.split == "val":
#             self.split = "valid"
#         self.annot_loader.filtering(keys=['etc', 'split'], condition=lambda x: x == self.split)


# class CXRTBMontgomery(BaseClassification):
#     def __init__(self,
#                  data_dir,
#                  split: str,
#                  transforms: Optional[Dict[str, Dict]] = None,
#                  fold_val: int = -1,
#                  lesion_classes: List[int] = None,
#                  use_low_image: bool = False,
#                  do_windowing: bool = True,
#                  do_standardization: bool = True,
#                  mean: float = 0.0,
#                  std: float = 1.0):
#         super().__init__(data_dir=data_dir,
#                          split=split,
#                          transforms=transforms,
#                          fold_val=fold_val,
#                          lesion_classes=lesion_classes,
#                          use_low_image=use_low_image,
#                          do_windowing=do_windowing,
#                          do_standardization=do_standardization,
#                          mean=mean,
#                          std=std,
#                          annotation_json_fname=_DATA_PATH['montgomery'])
#         if self.split == "val":
#             self.split = "valid"
#         self.annot_loader.filtering(keys=['etc', 'split'], condition=lambda x: x == self.split)


# class CXRTBX11K(BaseClassification):
#     def __init__(self,
#                  data_dir: str,
#                  split: str,
#                  transforms: Optional[Dict[str, Dict]] = None,
#                  fold_val: int = -1,
#                  lesion_classes: List[int] = None,
#                  use_low_image: bool = False,
#                  do_windowing: bool = True,
#                  do_standardization: bool = True,
#                  mean: float = 0,
#                  std: float = 1):
#         super().__init__(data_dir=data_dir,
#                          split=split,
#                          transforms=transforms,
#                          fold_val=fold_val,
#                          lesion_classes=lesion_classes,
#                          use_low_image=use_low_image,
#                          do_windowing=do_windowing,
#                          do_standardization=do_standardization,
#                          mean=mean,
#                          std=std,
#                          annotation_json_fname=_DATA_PATH['tbx11k'])
#         if self.split == "val":
#             self.split = "valid"
#         self.annot_loader.filtering(keys=['etc', 'split'], condition=lambda x: x == self.split)


# class CXRTBPublic(BaseClassification):
#     def __init__(self,
#                  data_dir: str,
#                  split: str,
#                  transforms: Optional[Dict[str, Dict]] = None,
#                  fold_val: int = -1,
#                  lesion_classes: List[int] = None,
#                  use_low_image: bool = False,
#                  do_windowing: bool = True,
#                  do_standardization: bool = True,
#                  mean: float = 0.0,
#                  std: float = 1.0):
#         super().__init__(data_dir=data_dir,
#                          split=split,
#                          transforms=transforms,
#                          fold_val=fold_val,
#                          lesion_classes=lesion_classes,
#                          use_low_image=use_low_image,
#                          do_windowing=do_windowing,
#                          do_standardization=do_standardization,
#                          mean=mean,
#                          std=std,
#                          annotation_json_fname=list(_DATA_PATH.values()))
#         if self.split == "val":
#             self.split = "valid"
#         self.annot_loader.filtering(keys=['etc', 'split'], condition=lambda x: x == self.split)
