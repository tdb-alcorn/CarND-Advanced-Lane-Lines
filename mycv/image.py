import os
from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt


def convert_to_gray(img:np.array, rgb:bool=False) -> np.array:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if rgb else cv2.COLOR_BGR2GRAY)


def read(filename:str) -> np.array:
    return cv2.imread(filename)


def read_dir(directory:str) -> List[np.array]:
    files = os.listdir(directory)
    images: List[np.array] = list()
    for f in files:
        try:
            img = read(os.path.join(directory, f))
            if img is not None:
                images.append(img)
        except Exception as e:
            print(e)
            continue
    return images


def preview(filename:str, gray:bool=False):
    show(read(filename), gray=gray)


def show(image:np.array, gray:bool=False):
    cmap = 'gray' if gray else None
    plt.imshow(image, cmap=cmap)