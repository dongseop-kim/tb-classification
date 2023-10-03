import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from trainer.components.base import DummyData

from .base import BaseComponents

_ANNOTATION = "/team/team_pxi/pxi-dataset/cxr/public/tbx11k/tbx11k_v1.2_tmp.json"


class TBX11K(BaseComponents):
    """
    TBX11K Tuberculosis dataset.

    Args:
        data_dir (Union[Path, str]): data directory
        split (str): train, val, test, trainval
        transforms (Dict[str, Dict]): transforms for train, val, test, trainval
        use_low_image (bool): use low resolution image (1024x1024) 
    """

    def __init__(self,
                 data_dir: Union[Path, str],
                 split: str,
                 transforms: Optional[Dict[str, Dict]] = None,
                 use_low_image: bool = True):
        super().__init__(data_dir, split)

        with open(_ANNOTATION) as f:
            annots = json.load(f)['annotations']
            split = 'valid' if self.split == 'val' else split
            self.annots = [annot for annot in annots if annot['etc']['split'] == split]
            # split 이 test 일경우 etc id 기준으로 정렬
            if self.split == 'test':
                self.annots = sorted(self.annots, key=lambda x: x['etc']['test_id'])

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

        labels = 1 if data['objects'] else 0  # 0: normal, 1: active tb & latent tb
        labels = np.array([labels,], dtype=np.int64)

        # active = 1 if '15Tuberculosis-active' in [obj['lesion_name'] for obj in data['objects']] else 0

        return DummyData(path_image=path_image,
                         image=image,
                         labels=labels)
