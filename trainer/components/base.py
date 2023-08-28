
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from utils.cxr import load_cxr_image


@dataclass
class DummyData:
    path_image: Union[Path, str]
    image: np.ndarray
    labels: Any
    masks: Optional[np.ndarray] = None
    boxes: Optional[List[float]] = None
    boxes_label: Optional[List[int]] = None
    etc: Dict[str, Any] = field(default_factory=dict)



class BaseComponents(Dataset):
    """
    Base Componenets for all datasets

    Args:
        data_dir (Union[Path, str]): data directory
        split (str): train, val, test, trainval
        transforms (Dict[str, Dict]): transforms for train, val, test, trainval
    """

    def __init__(self,
                 data_dir: Union[Path, str],
                 split: str,
                 transforms: Optional[Dict[str, Dict]] = None):
        self.data_dir = data_dir
        assert split in ['train', 'val', 'test', 'trainval']
        self.split = split
        self.transforms = transforms

        self.num_classes = -1
        self.collate_fn = None

    def __getitem__(self, idx):
        dummy_data: DummyData = self._load_data(idx)
        transformed = self.transforms(image=dummy_data.image)
        image = F.to_tensor(transformed['image'])
        return {"image": image,  # C H W
                "target": {"labels": dummy_data.labels,
                           "path": dummy_data.path_image}}

    def __len__(self):
        # return len(self.annot_loader)
        return -1

    def _load_image(self, path_image: str, is_cxr: bool = True) -> np.ndarray:
        image: np.ndarray = cv2.imread(path_image, cv2.IMREAD_UNCHANGED)
        image = image.astype(np.float32)
        image = load_cxr_image(path_image) if is_cxr else image
        image = image * 255.0
        image = image.astype(np.uint8)
        return image

    @abstractmethod
    def _load_data(self, idx) -> DummyData:
        raise NotImplementedError
