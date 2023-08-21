
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from utils.cxr import load_cxr_image


class BaseClassification(Dataset):
    """
    Base Dataset for classification task

    Args:
        data_dir (Union[Path, str]): data directory
        split (str): train, val, test
        transforms (Dict[str, Dict]): transforms for train, val, test
        fold_val (int): fold number for validation
        lesion_classes (List[int]): list of lesion classes
        use_low_image (bool): use low image or not
        do_windowing (bool): do windowing or not
        do_standardization (bool): do standardization or not
        mean (float): mean for standardization
        std (float): std for standardization
        annotation_json_fname (str): annotation json file name
    """

    def __init__(self,
                 data_dir: Union[Path, str],
                 split: str,
                 transforms: Optional[Dict[str, Dict]] = None,
                 annotation_json_fname: Union[List[str], str] = None):
        self.data_dir = data_dir
        self.split = split
        # self.transforms = get_transform(transforms)
        self.transforms = transforms

        self.num_classes = -1
        self.collate_fn = None

    def __getitem__(self, idx):
        image, labels, path = self._load_data(idx)
        transformed = self.transforms(image=image)
        image = F.to_tensor(transformed['image'])
        return {"image": image,  # C H W
                "target": {"labels": labels,  # C
                           "path": path}}

    def __len__(self):
        # return len(self.annot_loader)
        return -1

    def _load_image(self, path_image: str, is_cxr: bool = True) -> np.ndarray:
        image: np.ndarray = cv2.imread(path_image, cv2.IMREAD_UNCHANGED)
        image = image.astype(np.float32)
        image = load_cxr_image(filename=path_image)
        image = image * 255.0
        image = image.astype(np.uint8)
        return image

    def _load_data(self, idx):
        annot: AnnotOutput = self.annot_loader[idx]

        # load image
        path_image: str = annot.path_image
        # TODO: 아래 변환을 변수로 처리하기
        path_image = path_image.replace("image_v1", "image_v1_1024") if self.use_low_image else path_image
        image = self._load_image(path_image, self.do_windowing, self.split)

        # load labels
        labels = annot.lesion_classes
        labels = self._class_to_onehot(labels)
        return image, labels, path_image

    def _class_to_onehot(self, lesion_classes: List[int]) -> np.ndarray:
        onehot = np.zeros(self.num_classes, dtype=np.int64)

        if not lesion_classes:
            onehot[0] = 1
            return onehot
        for class_id in lesion_classes:
            onehot[self.classid_to_trainid[class_id]] = 1
        return onehot
