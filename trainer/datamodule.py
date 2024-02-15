from functools import partial
from typing import Any, Optional

import albumentations as A
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from univdt.components import PublicTuberculosis
from univdt.transforms.builder import AVAILABLE_TRANSFORMS

DEFAULT_HEIGHT = 768
DEFAULT_WIDTH = 768
DEFAULT_TRAIN_TRANSFORMS = A.Compose([AVAILABLE_TRANSFORMS['random_resize'](height=DEFAULT_HEIGHT,
                                                                            width=DEFAULT_WIDTH,
                                                                            pad_val=0, p=1.0),
                                      A.HorizontalFlip(p=0.5),
                                      AVAILABLE_TRANSFORMS['random_blur'](magnitude=0.2, p=0.5),
                                      AVAILABLE_TRANSFORMS['random_brightness'](magnitude=0.2, p=0.5),
                                      AVAILABLE_TRANSFORMS['random_contrast'](magnitude=0.2, p=0.5),
                                      AVAILABLE_TRANSFORMS['random_gamma'](magnitude=0.2, p=0.5),
                                      AVAILABLE_TRANSFORMS['random_noise'](magnitude=0.2, p=0.5),
                                      AVAILABLE_TRANSFORMS['random_windowing'](magnitude=0.5, p=0.5),
                                      AVAILABLE_TRANSFORMS['random_zoom'](scale=0.2, pad_val=0, p=0.5),
                                      A.Affine(rotate=(-45, 45), p=0.5),
                                      A.Affine(translate_percent=(0.01, 0.1), p=0.5),
                                      ])

DEFAULT_VALID_TRANSFORMS = A.Compose([A.Resize(height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH, p=1.0)])
DEFAULT_TEST_TRANSFORMS = A.Compose([A.Resize(height=DEFAULT_HEIGHT, width=DEFAULT_WIDTH, p=1.0)])


class TBDataModule(LightningDataModule):
    """
    Base dataloader for all datasets

    Args:
        data_dir : root directory for dataset
        datasets : dataset names to load. if multiple datasets are given, they will be concatenated
        batch_size : batch size for dataloader. if batch_size is given, all batch_size_* will be ignored
        num_workers : number of workers for dataloader
        additional_keys (optional) : additional keys to load dataset
        split_train (optional) : split name for training. default is 'train'. either 'train' or 'trainval'
        split_val (optional) : split name for validation. default is 'val'. either 'val' or 'test'
        split_test (optional) : split name for testing. default is 'test'. either 'val' or 'test'

    """

    def __init__(self, data_dir: str,
                 datasets: str | list[str],
                 batch_size: Optional[int] = None,
                 num_workers: Optional[int] = None,

                 split_train: Optional[str] = 'train',
                 split_val: Optional[str] = 'val',
                 split_test: Optional[str] = 'test',
                 additional_keys: Optional[list[str]] = [],

                 transforms_train: Optional[dict[str, Any]] = None,
                 transforms_val: Optional[dict[str, Any]] = None,
                 transforms_test: Optional[dict[str, Any]] = None,

                 datasets_train: Optional[str | list[str]] = None,
                 datasets_val: Optional[str | list[str]] = None,
                 datasets_test: Optional[str | list[str]] = None,

                 batch_size_train: Optional[int] = None,
                 batch_size_val: Optional[int] = None,
                 batch_size_test: Optional[int] = None):
        super().__init__()
        self.data_dir = data_dir
        """ Load all datasets for training"""
        self.datasets = datasets if isinstance(datasets, list) else [datasets]

        # get splits and check
        self.split_train = split_train if split_train is not None else 'train'
        self.split_val = split_val if split_val is not None else 'val'
        self.split_test = split_test if split_test is not None else 'test'
        assert self.split_train in ['train', 'trainval'], f'Invalid split for training: {self.split_train}'
        assert self.split_val in ['val', 'test'], f'Invalid split for validation: {self.split_val}'
        assert self.split_test in ['val', 'test'], f'Invalid split for testing: {self.split_test}'
        self.additional_keys = additional_keys

        # get transforms
        self.transforms_train = transforms_train if transforms_train is not None else DEFAULT_TRAIN_TRANSFORMS
        self.transforms_val = transforms_val if transforms_val is not None else DEFAULT_VALID_TRANSFORMS
        self.transforms_test = transforms_test if transforms_test is not None else DEFAULT_TEST_TRANSFORMS

        # set hyperparameters for dataloader
        self.batch_size_train = batch_size_train or batch_size
        self.batch_size_val = batch_size_val or batch_size
        self.batch_size_test = batch_size_test or batch_size
        self.num_workers = num_workers if num_workers is not None else 0
        self.persistent_workers = True if self.num_workers > 0 else False
        self.pin_memory = True  # if self.num_workers > 0 else False

        self.dataset_train: Dataset = None
        self.dataset_val: Dataset = None
        self.dataset_test: Dataset = None

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        # stage must be in ['fit', 'validate', 'test', 'predict']
        assert stage in ['fit', 'validate', 'test', 'predict'], f'Invalid stage: {stage}'

        # TODO: Multiple dataset 사용 가능하도록 수정하기.
        assert len(self.datasets) <= 1, "Multiple datasets are not supported yet"
        dataset = partial(PublicTuberculosis, root_dir=self.data_dir, additional_keys=self.additional_keys)
        match stage:
            case 'fit':
                self.dataset_train = dataset(dataset=self.datasets[0], split=self.split_train,
                                             transform=self.transforms_train)
                self.dataset_val = dataset(dataset=self.datasets[0], split=self.split_val,
                                           transform=self.transforms_train)
            case 'validate':
                self.dataset_val = dataset(dataset=self.datasets[0], split=self.split_val,
                                           transform=self.transforms_train)
            case 'test':
                self.dataset_test = dataset(dataset=self.datasets[0], split=self.split_test,
                                            transform=self.transforms_test)
            case 'predict':
                pass

    def teardown(self, stage: str):
        """Called at the end of fit (train + validate), validate, test, or predict."""
        super().teardown(stage)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size_train, shuffle=True, num_workers=self.num_workers,
                          drop_last=True, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers,
                          collate_fn=self.dataset_train.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size_val, shuffle=False, num_workers=self.num_workers,
                          drop_last=False, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers,
                          collate_fn=self.dataset_val.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size_test, shuffle=False, num_workers=self.num_workers,
                          drop_last=False, pin_memory=self.pin_memory, persistent_workers=self.persistent_workers,
                          collate_fn=self.dataset_test.collate_fn)

    def predict_dataloader(self):
        pass
