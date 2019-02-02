import os
from typing import List, Optional, Union, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt


def convert_to_gray(img:np.array, rgb:bool=False) -> np.array:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if rgb else cv2.COLOR_BGR2GRAY)


def convert_to_hls(img:np.array, rgb:bool=False) -> np.array:
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS if rgb else cv2.COLOR_BGR2HLS)


def read(filename:str, rgb:bool=False) -> np.array:
    if rgb:
        return plt.imread(filename)
    return cv2.imread(filename)


def write(img:np.array, filename:str, rgb:bool=False):
    dirs = os.path.dirname(filename)
    os.makedirs(dirs, exist_ok=True)
    if rgb:
        plt.imsave(filename, img)
    cv2.imwrite(filename, img)


def read_dir(directory:str,
    rgb:bool=False,
    filenames:bool=False) -> Union[List[np.array], List[Tuple[str, np.array]]]:
    files = os.listdir(directory)
    images: List[np.array] = list()
    if filenames:
        names: List[str] = list()
    for f in files:
        try:
            name = os.path.join(directory, f)
            img = read(name, rgb=rgb)
            if img is not None:
                images.append(img)
                if filenames:
                    names.append(name)
        except Exception as e:
            print(e)
            continue
    if filenames:
        return list(zip(names, images))
    return images


def preview(filename:str, gray:bool=False):
    show(read(filename), gray=gray)


def show(img:np.array, gray:bool=False):
    cmap = 'gray' if gray else None
    plt.imshow(img, cmap=cmap)


def crop_to_bottom_half(img:np.array) -> np.array:
    height = img.shape[0]
    # width = img.shape[1]
    return img[height//2:, :]


def like(img:np.array, num_channels:Optional[int]=None) -> np.array:
    '''like returns a new image of the same size as the input image, with the option to specify a different number of channels'''
    if num_channels is None:
        return np.copy(img)
    img = np.squeeze(img)
    if len(img.shape) > 2:
        raise Exception('image must be single channel to use num_channels option')
    return np.tile(img, (num_channels, 1, 1))


def add(img1:np.array, img2:np.array,
        alpha:float=1, beta:float=0.3, gamma:float=0) -> np.array:
    return cv2.addWeighted(img1, alpha, img2, beta, gamma)