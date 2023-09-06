from functools import partial
from typing import Any, Dict, List, Optional, Union

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from trainer.components.base import BaseComponents
from trainer.components.montgomery import Montgomery
from trainer.components.shenzhen import Shenzhen
from trainer.components.shmt import SHMT
from trainer.components.tbx11k import TBX11K
from trainer.transforms import parse_transforms

_AVAILABLE_DATASETS = {'montgomery': Montgomery,
                       'shenzhen': Shenzhen,
                       'shmt': SHMT,
                       'tbx11k': TBX11K}


class BaseDataModule(LightningDataModule):
    """
    Base DataModule for all datasets.

    Args:
        dataset (BaseComponents): dataset class
        data_dir (str): data directory
        batch_size (int): batch size
        transforms (Dict[str, Dict]): transforms for train, val, test, trainval
        stage (str): train, train_all, test
        dataset_config (Dict[str, Any]): keyword arguments for dataset
    """

    def __init__(self,
                 dataset: BaseComponents,
                 data_dir: str,
                 batch_size: int,
                 transforms: Dict[str, Dict],
                 stage: str = 'train',
                 dataset_config: Dict[str, Any] = {}):
        super().__init__()

        # hyperparameters defined by the yaml file
        self.dataset: BaseComponents = dataset
        self.data_dir: str = data_dir
        self.batch_size: int = batch_size
        self.transforms: Dict[str, Dict[str, Any]] = transforms
        self.stage: str = stage
        self.dataset_config: Dict[str, Any] = dataset_config  # keywork arguments for dataset

        # exception handling
        if self.stage not in ['train', 'train_all', 'test']:
            raise ValueError(f"stage must be one of ['train', 'train_all', 'test'], but got {self.stage}")
        if dataset not in _AVAILABLE_DATASETS.values():
            raise ValueError(f"dataset must be one of {_AVAILABLE_DATASETS.keys()}")

        # hyperparameters defined by the code in runtime
        self.num_workers = batch_size*2
        self.persistent_workers = True  # if self.num_workers > 0 else False

        # prepare BaseDataset instances for specific stage
        self.setup(stage)

        self.num_classes = self.data_train.num_classes

    def setup(self, stage: Optional[str] = None):
        self.data_train: BaseComponents = self.dataset(data_dir=self.data_dir,
                                                       split='trainval' if stage == 'trainall' else 'train',
                                                       transforms=parse_transforms(self.transforms['train']),
                                                       **self.dataset_config)
        self.data_valid: BaseComponents = self.dataset(data_dir=self.data_dir,
                                                       split='val' if stage == 'train' else 'test',
                                                       transforms=parse_transforms(self.transforms['val']),
                                                       **self.dataset_config)
        self.data_test: BaseComponents = self.dataset(data_dir=self.data_dir,
                                                      split='test',
                                                      transforms=parse_transforms(self.transforms['test']),
                                                      **self.dataset_config)

    def train_dataloader(self):
        return DataLoader(self.data_train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          sampler=None,
                          collate_fn=self.data_train.collate_fn,
                          num_workers=self.num_workers,
                          pin_memory=self.persistent_workers,
                          drop_last=True,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.data_valid,
                          batch_size=self.batch_size,
                          shuffle=False,
                          sampler=None,
                          collate_fn=self.data_valid.collate_fn,
                          num_workers=self.num_workers,
                          pin_memory=self.persistent_workers,
                          drop_last=False,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        return DataLoader(self.data_test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          sampler=None,
                          collate_fn=self.data_test.collate_fn,
                          num_workers=self.num_workers,
                          pin_memory=self.persistent_workers,
                          drop_last=False,
                          persistent_workers=self.persistent_workers)


DataModuleShenzhen = partial(BaseDataModule, dataset=Shenzhen)
DataModuleMontgomery = partial(BaseDataModule, dataset=Montgomery)
DataModuleSHMT = partial(BaseDataModule, dataset=SHMT)
DataModuleTBX11K = partial(BaseDataModule, dataset=TBX11K)
