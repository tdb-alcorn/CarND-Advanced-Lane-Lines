import cv2
import numpy as np

from . import image


def sobel1D(img:np.array, y:bool=False, kernel_size:int=3) -> np.array:
    '''
    computes the Sobel gradient in either the x or y direction.
    
    img must be grayscale
    '''
    if y:
        return cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)
    return cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)


def sobel2D(img:np.array, kernel_size:int=3) -> np.array:
    '''img must be grayscale'''
    sobelx = sobel1D(img, y=False, kernel_size=kernel_size)
    sobely = sobel1D(img, y=True, kernel_size=kernel_size)
    sobel_mag = np.sqrt(np.square(sobelx), np.square(sobely))
    # we abs value the gradient from each axis to restrict the direction of the
    # gradient to the top-right quadrant (x>0, y>0), since we don't care about
    # rotations of 90deg. e.g. if it looks like a lane, rotating it 90 degrees
    # it will still look like a lane.
    sobel_dir = np.arctan2(np.abs(sobely), np.abs(sobelx))

    return sobel_mag, sobel_dir


def abs(img:np.array) -> np.array:
    return np.absolute(img)


def eightBit(img:np.array) -> np.array:
    '''img should be grayscale'''
    minValue = np.min(img)
    valueRange = np.max(img) - minValue
    return (255 * (img - minValue) / valueRange).astype(np.uint8)


def threshold(img:np.array, tmin:int=0, tmax:int=255) -> np.array:
    '''
    img is expected to have one channel i.e. grayscale

    allowable values are those that fall in the range [tmin, tmax) i.e. exclusive of tmax
    '''
    tbin = np.zeros_like(img, dtype=np.uint8)
    tbin[(img >= tmin) & (img < tmax)] = 1
    return tbin


def grayscale(img:np.array, rgb:bool=False) -> np.array:
    return image.convert_to_gray(img, rgb=rgb)


def saturation(img:np.array, rgb:bool=False) -> np.array:
    hls_img = image.convert_to_hls(img, rgb=rgb)
    return hls_img[:,:,2]


def main(img:np.array, rgb:bool=False) -> np.array:
    g = grayscale(img, rgb=rgb)
    sobel_mag, sobel_dir = sobel2D(g, kernel_size=3)
    sobel_mask = threshold(sobel_mag, tmin=20, tmax=100) | threshold(sobel_dir, tmin=0.7, tmax=1.3)

    s = saturation(img, rgb=rgb)
    s_mask = threshold(s, tmin=170, tmax=255)

    return sobel_mask | s_mask