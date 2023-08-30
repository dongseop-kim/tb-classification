import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from trainer.components.base import DummyData

from .base import BaseComponents
from .montgomery import _ANNOTATION as _MONTGOMERY_ANNOTATION
from .shenzhen import _ANNOTATION as _SHENZHEN_ANNOTATION


class SHMT(BaseComponents):
    """
    Shenzhen and Montgomery Tuberculosis dataset.

    Args:
        data_dir (Union[Path, str]): data directory
        split (str): train, val, test, trainval
        transforms (Dict[str, Dict]): transforms for train, val, test, trainval
        use_low_image (bool): use low resolution image
    """

    def __init__(self,
                 data_dir: Union[Path, str],
                 split: str,
                 transforms: Optional[Dict[str, Dict]] = None,
                 use_low_image: bool = True):
        super().__init__(data_dir, split)

        with open(_SHENZHEN_ANNOTATION) as f:
            annots = json.load(f)['annotations']
            split = 'valid' if self.split == 'val' else split
            annots = [annot for annot in annots if annot['etc']['split'] == split]
            for ann in annots:
                ann['image'] = '/data1/pxi-dataset/cxr/public/tb_shenzhen/' + ann['image']
            self.annots = annots

        with open(_MONTGOMERY_ANNOTATION) as f:
            annots = json.load(f)['annotations']
            split = 'valid' if self.split == 'val' else split
            annots += [annot for annot in annots if annot['etc']['split'] == split]
            for ann in annots:
                ann['image'] = '/data1/pxi-dataset/cxr/public/tb_montgomery/' + ann['image']
            self.annots += annots

        self.num_classes = 1
        self.use_low_image = use_low_image
        self.transforms = self._build_transforms(transforms)

    def __len__(self):
        return len(self.annots)

    def _load_data(self, idx) -> DummyData:
        data = self.annots[idx]
        path_image = str(Path(self.data_dir) / data['image'])
        path_image = path_image.replace('image_v1', 'image_v1_1024') if self.use_low_image else path_image
        image = self._load_image(path_image, is_cxr=True)
        age = data['property']['age']
        sex = data['etc']['sex']
        labels = 1 if data['objects'] else 0
        labels = np.array([labels,], dtype=np.int64)
        return DummyData(path_image=path_image,
                         image=image,
                         labels=labels,
                         etc={'age': age,
                              'sex': sex})
