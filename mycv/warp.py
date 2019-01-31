from typing import Dict, Tuple, Any

import numpy as np
import cv2
import matplotlib.pyplot as plt

from . import image


class Warp(object):
    def __init__(self, mapping:Dict[Tuple[Any, Any], Tuple[Any, Any]]):
        '''
        initializes the warp perspective transformation from a mapping of
        four source points to target points
        '''
        if len(mapping) != 4:
            raise ValueError('source-to-target mapping must contain exactly four points')
        image_points, object_points = zip(*mapping.items())
        self.transform_matrix = cv2.getPerspectiveTransform(np.array(image_points, dtype=np.float32), np.array(object_points, dtype=np.float32))
    
    def transform(self, img:np.array) -> np.array:
        return cv2.warpPerspective(img, self.transform_matrix,
            (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)


_birds_eye_image_points = [(600.5522583218831, 446.8418731439211), (680.3359558530209, 446.8418731439211), (1040.9582686937638, 675.5551393998494), (265.46072869110424, 676.618922033598)]
_x_left = 265
_x_right = 1041
_y_top = 0
_y_bottom = 719
_birds_eye_object_points = [(_x_left, _y_top), (_x_right, _y_top), (_x_right, _y_bottom), (_x_left, _y_bottom)]
_birds_eye_warp = Warp(dict(zip(_birds_eye_image_points, _birds_eye_object_points)))

def birds_eye(img:np.array) -> np.array:
    return _birds_eye_warp.transform(img)


if __name__ == '__main__':
    # when run as a module, warp starts a helper point picker for choosing image points
    import sys
    image_file = sys.argv[1]

    img = image.read(image_file, matplotlib=True)
    plt.imshow(img)
    print(plt.ginput(4))