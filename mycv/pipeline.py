from collections import namedtuple

from moviepy.editor import VideoFileClip
import numpy as np
import cv2

from . import camera, image, warp, filters, lanes


class Pipeline(object):
    def __init__(self, calibration:camera.CalibrationParameters=camera.default_calibration):
        self.camera = camera.Camera()
        self.camera.calibrate(parameters=calibration, display=False)
        self.lane_detector = lanes.LaneDetector()
    
    def reset(self):
        self.lane_detector.reset()
    
    def run(self, video_file:str, output_file:str):
        clip = VideoFileClip(video_file)

        # img = clip.get_frame(0)
        # out = p.step(img)
        # image.write(out, 'output.jpg', rgb=True)

        processed_clip = clip.fl_image(self.step).subclip(0, 5)
        processed_clip.write_videofile(output_file, audio=False)
    
    def debug(self, img:np.array) -> np.array:
        undistorted = self.camera.undistort(img)
        warped = warp.birds_eye.transform(undistorted)
        filtered = filters.main(warped)
        leftx, lefty, rightx, righty, debug_img = self.lane_detector.debug(filtered)
        vis = lanes.visualize(warped,
            self.lane_detector.left, self.lane_detector.right,
            leftx, lefty, rightx, righty)
        unwarped_vis = warp.birds_eye.inverse_transform(vis)
        summed = image.add(undistorted, unwarped_vis)

        import matplotlib.pyplot as plt

        n_show = 7
        _fig, axes = plt.subplots(n_show, 1, figsize=(15, 8*n_show))
        ctr = 0

        def show(img):
            nonlocal ctr
            axes[ctr].imshow(img)
            ctr += 1

        show(undistorted)
        show(warped)
        show(filtered)
        show(debug_img)
        show(vis)
        show(unwarped_vis)
        show(summed)
    
    def step(self, img:np.array, display:bool=False) -> np.array:
        '''
        1. undistort image
        2. warp to birds eye perspective
        3. filter to mask
        4. run lane detection, get polynomial fit
        5. plot on warped unfiltered image
        6. unwarp
        7. render text
        '''
        undistorted = self.camera.undistort(img)
        warped = warp.birds_eye.transform(undistorted)
        filtered = filters.main(warped)
        leftx, lefty, rightx, righty = self.lane_detector.update(filtered)
        vis = lanes.visualize(warped,
            self.lane_detector.left, self.lane_detector.right,
            leftx, lefty, rightx, righty)
        unwarped_vis = warp.birds_eye.inverse_transform(vis)
        summed = image.add(undistorted, unwarped_vis)

        if display:
            import matplotlib.pyplot as plt

            n_show = 6
            _fig, axes = plt.subplots(n_show, 1, figsize=(15, 8*n_show))
            ctr = 0

            def show(img):
                nonlocal ctr
                axes[ctr].imshow(img)
                ctr += 1

            show(undistorted)
            show(warped)
            show(filtered)
            show(vis)
            show(unwarped_vis)
            show(summed)

        return summed


if __name__ == '__main__':
    # main runs the pipeline on a video input file
    import sys
    input_file, video_file = sys.argv[1:3]
    p = Pipeline()
    p.run(input_file, video_file)

    # input_dir = sys.argv[1]
    # images = image.read_dir(input_dir, filenames=True)
    # for filename, img in images:
    #     image.write(p.step(img), filename + '_out.jpg')
    #     p.reset()
