from typing import List, Union

from .classification import BasicClassifier

available_headers = {'base_classifier': BasicClassifier}


def build_header(name: str,
                 in_channels: Union[int, List[int]],
                 in_strides: Union[int, List[int]],
                 **kwargs):
    return available_headers[name](in_channels=in_channels,
                                   in_strides=in_strides,
                                   **kwargs)
