from typing import List, Union

from .classification import BasicClassifier, ClassifierWith2Head

available_headers = {'base_classifier': BasicClassifier,
                     '2head_classifier': ClassifierWith2Head}


def build_header(name: str, num_classes: int,
                 in_channels: Union[int, List[int]],
                 in_strides: Union[int, List[int]],
                 **kwargs):
    return available_headers[name](num_classes=num_classes, in_channels=in_channels,
                                   in_strides=in_strides, **kwargs)
