
from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import albumentations as A
import cv2
import numpy as np
import torchvision.transforms.functional as F
from albumentations.core.serialization import Serializable
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
                 split: str):
        self.data_dir = data_dir
        assert split in ['train', 'val', 'test', 'trainval']
        self.split = split

        self.num_classes = -1
        self.collate_fn = None

    def __getitem__(self, idx):
        dummy_data: DummyData = self._load_data(idx)
        image: np.ndarray = dummy_data.image
        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed['image']
        image = F.to_tensor(image)
        return {"image": image,  # C H W
                "target": {"labels": dummy_data.labels,
                           "path": dummy_data.path_image}}


    def _load_image(self, path_image: str, is_cxr: bool = True) -> np.ndarray:
        image: np.ndarray = cv2.imread(path_image, cv2.IMREAD_UNCHANGED).astype(np.float32)
        image = load_cxr_image(path_image) if is_cxr else image
        return image

    @abstractmethod
    def _load_data(self, idx) -> DummyData:
        raise NotImplementedError

    def _build_transforms(self, transforms: Optional[List[Serializable]]=None, **kwargs):
        if transforms is None:
            return transforms
        else:
            return A.Compose(transforms, **kwargs)