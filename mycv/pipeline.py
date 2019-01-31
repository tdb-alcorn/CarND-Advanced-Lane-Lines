from collections import namedtuple

import numpy as np
import cv2

from . import camera, image, warp, filters, lanes


class Pipeline(object):
    def __init__(self, calibration:camera.CalibrationParameters=camera.default_calibration):
        self.camera = camera.Camera()
        self.camera.calibrate(parameters=calibration, display=False)
        self.lane_detector = lanes.LaneDetector()
    
    def step(self, img:np.array) -> np.array:
        '''
        1. undistort image
        2. warp to birds eye perspective
        3. filter to mask
        4. run lane detection, get polynomial fit
        5. plot on warped unfiltered image
        6. unwarp
        7. render text
        '''
        import matplotlib.pyplot as plt
        n_show = 6
        _fig, axes = plt.subplots(n_show, 1, figsize=(15, 8*n_show))
        ctr = 0
        def show(img):
            nonlocal ctr
            axes[ctr].imshow(img)
            ctr += 1

        undistorted = self.camera.undistort(img)
        show(undistorted)
        warped = warp.birds_eye.transform(undistorted)
        show(warped)
        filtered = filters.main(warped)
        show(filtered)
        leftx, lefty, rightx, righty = self.lane_detector.update(filtered)
        vis = lanes.visualize(warped,
            self.lane_detector.left, self.lane_detector.right,
            leftx, lefty, rightx, righty)
        show(vis)
        unwarped_vis = warp.birds_eye.inverse_transform(vis)
        show(unwarped_vis)
        summed = image.add(undistorted, unwarped_vis)
        show(summed)
        return summed
