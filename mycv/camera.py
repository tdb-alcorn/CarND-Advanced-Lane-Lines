import os
from typing import List

import cv2
import numpy as np
import matplotlib.pyplot as plt

from . import image

class CalibrationError(Exception):
    pass


class Camera(object):
    def __init__(self):
        self.camera_matrix = np.zeros((3, 3))
        self.distortion_coeffs = np.zeros(5)
    
    def undistort(self, img:np.array) -> np.array:
        return cv2.undistort(img, self.camera_matrix, self.distortion_coeffs)
    
    def calibrate(self, images:List[np.array], nx:int, ny:int, display:bool=False):
        '''
        calibrate calibrates the camera given a set of chessboard images with a fixed number of chessboard corners to be detected.

        images = list of chessboard images
        nx = number of x chessboard corners
        ny = number of y chessboard corners
        '''
        sizex = 0
        sizey = 0
        corners = list()
        idx = 0
        for img in images:
            gray = image.convert_to_gray(img)

            if sizex == 0 or sizey == 0:
                sizex, sizey = gray.shape[0:2]
                print(sizex, sizey)

            if gray.shape[0] != sizex or gray.shape[1] != sizey:
                print(idx)
                raise CalibrationError('all calibration images must be the same size. expected: (%d, %d) got: (%d, %d)' % (sizex, sizey, gray.shape[0], gray.shape[1]))
            
            # do some thresholding to help cv2 find chessboard corners
            # dark_pixels = gray < 100
            # gray[dark_pixels] = 0
            # gray[~dark_pixels] = 255
            # image.show(gray, gray=True)

            ret, img_corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            if not ret:
                image.show(img)
                raise CalibrationError('failed to find chessboard corners in given image')
            corners.extend(img_corners)

            # Draw and display the corners
            if display:
                cv2.drawChessboardCorners(img, (nx, ny), img_corners, ret)
                image.show(img)
            
            idx += 1

        imgpoints = np.array([corner.reshape(-1) for corner in corners], dtype=np.float32)

        objpoints = generate_object_points(sizex, sizey, nx, ny, offsetx=100, offsety=100)

        ret, mtx, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (sizey, sizex), None, None)

        if not ret:
            raise CalibrationError('failed to get camera matrix and/or distortion coefficients')

        _, _ = rvecs, tvecs
        self.camera_matrix = mtx
        self.distortion_coeffs = distCoeffs


def generate_object_points(sizex:int, sizey:int, nx:int, ny:int, offsetx:int=0, offsety:int=0) -> np.array:
    dx = (sizex - 2*offsetx)/nx
    dy = (sizey - 2*offsety)/ny
    object_points = list()
    for y in range(ny):
        for x in range(nx):
            object_points.append((offsetx + x*dx, offsety + y*dy))
    return np.array(object_points, dtype=np.float32)
