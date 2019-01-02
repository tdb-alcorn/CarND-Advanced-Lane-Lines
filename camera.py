import cv2
import numpy as np
from typing import List

class CalibrationError(Exception):
    pass


class Camera(object):
    def __init__(self):
        self.camera_matrix = np.zeros((3, 3))
        self.distortion_coeffs = np.zeros(5)
    
    def undistort(self, img:np.array) -> np.array:
        return cv2.undistort(img, self.camera_matrix, self.distortion_coeffs)
    
    def calibrate(self, images:List[np.array], nx:int, ny:int):
        '''
        calibrate calibrates the camera given a set of chessboard images with a fixed number of chessboard corners to be detected.

        images = list of chessboard images
        nx = number of x chessboard corners
        ny = number of y chessboard corners
        '''
        sizex = 0
        sizey = 0
        corners = list()
        for image in images:
            gray = convert_to_gray(image)
            ret, img_corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if not ret:
                raise CalibrationError('failed to find chessboard corners in given image')
            corners.extend(img_corners)
            # Draw and display the corners
            # cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            # plt.imshow(img)
        imgpoints = np.array(corner.reshape(-1) for corner in corners, dtype=np.float32)

        objpoints = generate_object_points(sizex, sizey, nx, ny, offsetx=100, offsety=100)

        ret, mtx, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        if not ret:
            raise CalibrationError('failed to get camera matrix and/or distortion coefficients')


def generate_object_points(sizex:int, sizey:int, nx:int, ny:int, offsetx:int=0, offsety:int=0) -> np.array:
    dx = (sizex - 2*offsetx)/nx
    dy = (sizey - 2*offsety)/ny
    object_points = list()
    for y in ny:
        for x in nx:
            objpoints.append((offsetx + x*dx, offsety + y*dy))
    return np.array(object_points, dtype=np.float32)


def convert_to_gray(img:np.array, rgb:bool=False) -> np.array:
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if rgb else cv2.COLOR_BGR2GRAY)

def read_image(filename:str) -> np.array:
    return cv2.imread(filename)