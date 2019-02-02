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

        # Define conversions in x and y from pixels space to meters
        self.metres_per_pixel_y = 30/720 # meters per pixel in y dimension
        self.metres_per_pixel_x = 3.7/700 # meters per pixel in x dimension

        # Filter parameters
        self.filter_params = {
            'sobel_mag_min': 50,
            'sobel_mag_max': 150,
            'sobel_dir_min': 0.4,
            'sobel_dir_max': 0.8,
            'saturation_min': 120,
            'saturation_max': 255,
        }
    
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
            self.metres_per_pixel_x, self.metres_per_pixel_y,
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
    
    def step(self, img:np.array, rgb:bool=False, display:bool=False) -> np.array:
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
        filtered = filters.main(warped, rgb=rgb, **self.filter_params)
        leftx, lefty, rightx, righty = self.lane_detector.update(filtered)
        vis, radius_of_curvature, lane_deviation = lanes.visualize(warped,
            self.lane_detector.left, self.lane_detector.right,
            self.metres_per_pixel_x, self.metres_per_pixel_y,
            leftx, lefty, rightx, righty)
        unwarped_vis = warp.birds_eye.inverse_transform(vis)
        summed = image.add(undistorted, unwarped_vis)
   
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Output radius of curvature
        cv2.putText(summed, 'Radius of Curvature = %.2fm' % radius_of_curvature,
            (50,100), font, 2, (255,255,255), 2, cv2.LINE_AA)

        # Output lane deviation
        deviation_msg = 'Vehicle is centered'
        if lane_deviation < -0.1:
            deviation_msg = 'Vehicle is %.2fm left of center' % abs(lane_deviation)
        elif lane_deviation > 0.1:
            deviation_msg = 'Vehicle is %.2fm right of center' % abs(lane_deviation)
        cv2.putText(summed, deviation_msg,
            (50,200), font, 2, (255,255,255), 2, cv2.LINE_AA)

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
    # main runs the pipeline on a given input file
    import sys
    import os

    inp, out = sys.argv[1:3]
    inp_basename = os.path.basename(inp)
    inp_ext = ''
    inp_basename_parts = inp_basename.split('.')
    if len(inp_basename_parts) > 1:
        inp_ext = inp_basename_parts[-1]

    p = Pipeline()
    
    if os.path.isdir(inp):
        # process a folder of images
        input_dir, output_dir = inp, out
        images = image.read_dir(input_dir, filenames=True)
        for filename, img in images:
            basename = os.path.basename(filename)
            image.write(p.step(img), os.path.join(output_dir, basename))
            p.reset()
    elif inp_ext == 'mp4':
        # process a video
        input_file, video_file = inp, out
        p.run(input_file, video_file)
    elif inp_ext == 'jpg' or inp_ext == 'png':
        # process a single image
        img = image.read(inp)
        image.write(p.step(img), out)
    else:
        print('Unknown input type: input must be either directory, mp4, or image file (jpg, png)')
        print('Usage: python -m mycv.pipeline <input> <output>')
