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


def hue(img:np.array, rgb:bool=False) -> np.array:
    hls_img = image.convert_to_hls(img, rgb=rgb)
    return histogram_equalization(hls_img[:,:,0])


def red(img:np.array, rgb:bool=False) -> np.array:
    if rgb:
        return img[:,:,0]
    return img[:,:,2]


def green(img:np.array, rgb:bool=False) -> np.array:
    return img[:,:,1]


def blue(img:np.array, rgb:bool=False) -> np.array:
    if rgb:
        return img[:,:,2]
    return img[:,:,0]


def histogram_equalization(img:np.array) -> np.array:
    '''
    assumes a single channel image
    '''
    return cv2.equalizeHist(img)
    # hist, _ = np.histogram(img, bins=256, range=(0, 255))
    # cdf = np.cumsum(hist)
    # cdfmin = np.min(cdf)
    # size = img.shape[0]*img.shape[1]
    # h = np.vectorize(lambda v: np.round((cdf[v]-cdfmin)/(size-cdfmin)*255))
    # return h(img)


def main(img:np.array, rgb:bool=False,
        sobel_mag_min:int=30, sobel_mag_max:int=200,
        sobel_dir_min:float=0.5, sobel_dir_max:float=1.571,
        saturation_min:int=120, saturation_max:int=255,
        hue_min:int=32, hue_max:int=60,
        hue_saturation_min:int=120, hue_saturation_max:int=255,
        red_min:int=120, red_max:int=255,
        green_min:int=120, green_max:int=255,
        ) -> np.array:
    # print(sobel_mag_min, sobel_mag_max, sobel_dir_min, sobel_dir_max, saturation_min, saturation_max)
    g = grayscale(img, rgb=rgb)
    sobel_mag, sobel_dir = sobel2D(g, kernel_size=3)
    sobel_mask = threshold(sobel_mag, tmin=sobel_mag_min, tmax=sobel_mag_max
        ) & threshold(sobel_dir, tmin=sobel_dir_min, tmax=sobel_dir_max)

    s = saturation(img, rgb=rgb)
    s_mask = threshold(s, tmin=saturation_min, tmax=saturation_max)

    h = hue(img, rgb=rgb)
    hue_pre_mask = threshold(h, tmin=hue_min, tmax=hue_max)

    hue_sat_mask = threshold(s, tmin=hue_saturation_min, tmax=hue_saturation_max)
    h_mask = hue_pre_mask & hue_sat_mask

    r = red(img, rgb=rgb)
    r_mask = threshold(r, tmin=red_min, tmax=red_max)

    g = green(img, rgb=rgb)
    g_mask = threshold(r, tmin=green_min, tmax=green_max)

    return sobel_mask | h_mask | s_mask | r_mask | g_mask
    # return sobel_mask | s_mask | r_mask | g_mask
    # return sobel_mask | h_mask | r_mask



if __name__ == '__main__':
    import sys
    import matplotlib.pyplot as plt

    image_file = sys.argv[1]
    img = image.read(image_file, rgb=True)
    plt.imshow(img)
    point = plt.ginput(1, show_clicks=True)[0]
    print(point)
    hls_img = image.convert_to_hls(img, rgb=True)
    print(hls_img.shape)
    print(hls_img[int(point[0])][int(point[1])])
