from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch

from trainer.components.montgomery import Montgomery
from trainer.components.shenzhen import Shenzhen


def test_shenzhen():
    sh_train = Shenzhen("/data1/pxi-dataset/cxr/public/tb_shenzhen", 'train', {})
    sh_test = Shenzhen("/data1/pxi-dataset/cxr/public/tb_shenzhen", 'test', {})
    sh_valid = Shenzhen("/data1/pxi-dataset/cxr/public/tb_shenzhen", 'val', {})
    assert len(sh_train) == 530
    assert len(sh_test) == 66
    assert len(sh_valid) == 66 

def test_montgomery():
    mt_train = Montgomery("/data1/pxi-dataset/cxr/public/tb_montgomery", 'train', {})
    mt_test = Montgomery("/data1/pxi-dataset/cxr/public/tb_montgomery", 'test', {})
    mt_valid = Montgomery("/data1/pxi-dataset/cxr/public/tb_montgomery", 'val', {})
    assert len(mt_train) == 110
    assert len(mt_test) == 14
    assert len(mt_valid) == 14