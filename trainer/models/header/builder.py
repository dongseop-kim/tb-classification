from typing import List, Union

available_headers = {}


def build_header(name: str,
                 in_channels: Union[int, List[int]],
                 in_strides: Union[int, List[int]],
                 **kwargs):
    return available_headers[name](in_channels=in_channels,
                                   in_strides=in_strides,
                                   **kwargs)
