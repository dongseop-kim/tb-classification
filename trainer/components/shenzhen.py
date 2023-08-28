import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from trainer.components.base import DummyData
from .base import BaseComponents

_ANNOTATION = "/team/team_pxi/pxi-dataset/cxr/public/tb_shenzhen/tb_shenzhen_v1.2.json"
class Shenzhen(BaseComponents):
    """
    Shenzhen Tuberculosis dataset.

    Args:
        data_dir (Union[Path, str]): data directory
        split (str): train, val, test, trainval
        transforms (Dict[str, Dict]): transforms for train, val, test, trainval
    """

    def __init__(self,
                 data_dir: Union[Path, str],
                 split: str,
                 transforms: Optional[Dict[str, Dict]] = None):
        super().__init__(data_dir, split, transforms)

        with open(_ANNOTATION) as f:
            annots = json.load(f)
            annots = annots['annotations']
            if self.split == 'val':
                split = 'valid'
            print(split)
            self.annots = [annot for annot in annots if annot['etc']['split'] == split]
        self.num_classes = 1 

    def __len__(self):
        return len(self.annots)
    
    def _load_data(self, idx) -> DummyData:
        data = self.annots[idx]
        path_image = str(Path(self.data_dir) / data['image'])
        image = self._load_image(path_image, is_cxr=True)
        age = data['property']['age']
        sex = data['etc']['sex']
        if data['objects']:
            labels = 1
        return DummyData(path_image=path_image,
                         image=image,
                         labels=labels,
                         etc={'age': age,
                              'sex': sex})
        
        
