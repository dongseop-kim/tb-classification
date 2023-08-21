import random

import albumentations.core.bbox_utils as bbox_utils
import cv2
from albumentations.augmentations.geometric import functional as geo_f
from albumentations.core.transforms_interface import DualTransform


class RandomRatio(DualTransform):
    """
    Apply specific ratio resize & padding

        default ratio: [8, 12], [11, 14],[14, 17], [25, 30]
                       (1.5, 1.272727, 1.214285, 1.2)
    """

    def __init__(self,
                 height=768,
                 width=768,
                 #  ratio=(1.5, 1.272727, 1.214285, 1.2),
                 ratio=(0.785714, 0.823529, 0.833333, 0.666666),
                 interpolation=cv2.INTER_LANCZOS4,
                 rand_border_mode=False,
                 border_mode=cv2.BORDER_REFLECT,
                 always_apply=False,
                 p=1):
        super().__init__(always_apply, p)
        # target height width
        self.height = height
        self.width = width

        # self.ratio_list = [[17, 14], [11, 14], [8, 12], [25, 30]]
        self.ratio_list = ratio

        self.value = 0
        self.mask_value = 0

        self.rand_border_mode = rand_border_mode
        self.interpolation = interpolation
        self.border_mode = border_mode

    def get_params(self):
        return {"scale": random.choice(self.ratio_list)}

    def update_params(self, params, **kwargs):
        params = super().update_params(params, **kwargs)

        # input image shape
        height = params["rows"]
        width = params["cols"]
        scale = params["scale"]
        # print(height, width, scale)

        if random.uniform(0, 1) < 0.5:
            scale_h = self.height/height*scale
            scale_w = self.width/width
        else:
            scale_h = self.height/height
            scale_w = self.width/width*scale
        rows = int(height*scale_h)
        cols = int(width*scale_w)
        # print(rows, cols, scale_h, scale_w)

        if rows < self.height:
            h_pad_top = int((self.height - rows) / 2.0)
            h_pad_bottom = self.height - rows - h_pad_top
        else:
            h_pad_top = 0
            h_pad_bottom = 0

        if cols < self.width:
            w_pad_left = int((self.width - cols) / 2.0)
            w_pad_right = self.width - cols - w_pad_left
        else:
            w_pad_left = 0
            w_pad_right = 0

        # now support only center
        # h_pad_top, h_pad_bottom, w_pad_left, w_pad_right = self.__update_position_params(
        #    h_top=h_pad_top, h_bottom=h_pad_bottom, w_left=w_pad_left, w_right=w_pad_right
        # )

        params.update({"rows": rows,
                       "cols": cols,
                       "scale_h": scale_h,
                       "scale_w": scale_w,
                       "pad_top": h_pad_top,
                       "pad_bottom": h_pad_bottom,
                       "pad_left": w_pad_left,
                       "pad_right": w_pad_right})
        return params

    def apply(self,
              img,
              rows,
              cols,
              pad_top=0,
              pad_bottom=0,
              pad_left=0,
              pad_right=0,
              interpolation=cv2.INTER_LINEAR,
              **params):
        img = geo_f.resize(img, rows, cols, interpolation)
        if self.rand_border_mode:
            self.border_mode = random.choice([cv2.BORDER_REFLECT, cv2.BORDER_CONSTANT])
        return geo_f.pad_with_params(img,
                                     pad_top,
                                     pad_bottom,
                                     pad_left,
                                     pad_right,
                                     border_mode=self.border_mode,
                                     value=self.value)

    def apply_to_mask(self,
                      img,
                      rows,
                      cols,
                      pad_top=0,
                      pad_bottom=0,
                      pad_left=0,
                      pad_right=0,
                      **params):
        img = geo_f.resize(img, rows, cols, cv2.INTER_NEAREST)

        if self.rand_border_mode:
            self.border_mode = random.choice([cv2.BORDER_REFLECT, cv2.BORDER_CONSTANT])
        return geo_f.pad_with_params(img,
                                     pad_top,
                                     pad_bottom,
                                     pad_left,
                                     pad_right,
                                     border_mode=self.border_mode,
                                     value=self.mask_value)

    def apply_to_bbox(self,
                      bbox,
                      rows=0,
                      cols=0,
                      pad_top=0,
                      pad_bottom=0,
                      pad_left=0,
                      pad_right=0,
                      **params):
        x_min, y_min, x_max, y_max = bbox_utils.denormalize_bbox(bbox, rows, cols)
        bbox = x_min+pad_left, y_min+pad_top, x_max+pad_left, y_max+pad_top
        return bbox_utils.normalize_bbox(bbox, rows+pad_top+pad_bottom, cols+pad_left+pad_right)

    # not confirmed
    def apply_to_keypoint(self,
                          keypoint,
                          rows=0,
                          cols=0,
                          pad_top=0,
                          pad_bottom=0,
                          pad_left=0,
                          pad_right=0,
                          **params):
        keypoint = geo_f.keypoint_scale(keypoint, rows/self.height, cols/self.width)
        x, y, angle, scale = keypoint
        return x+pad_left, y+pad_top, angle, scale
