from collections import namedtuple

from moviepy.editor import VideoFileClip
import numpy as np
import cv2

from . import camera, image, warp, filters, lanes

_metres_per_pixel_y = 30/720
_metres_per_pixel_x = 3.7/700

Parameter = namedtuple('Parameter', ['value', 'min', 'max', 'type'])

def make_slider(param:Parameter):
    from ipywidgets import IntSlider, FloatSlider, Layout
    if param.type is int:
        return IntSlider(value=param.value, min=param.min, max=param.max, layout=Layout(width='100%'))
    elif param.type is float:
        return FloatSlider(value=param.value, min=param.min, max=param.max, layout=Layout(width='100%'))
    return ValueError('Unknown parameter type %s in %s' % (param.type, param))

# Filter parameters
filter_params = {
    'sobel_mag_min': Parameter(90, 0, 255, int),
    'sobel_mag_max': Parameter(200, 0, 255, int),
    'sobel_dir_min': Parameter(0.7, 0, np.pi/2, float),
    'sobel_dir_max': Parameter(1.5, 0, np.pi/2, float),
    'saturation_min': Parameter(240, 0, 255, int),
    'saturation_max': Parameter(255, 0, 255, int),
    'hue_min': Parameter(20, 0, 255, int),
    'hue_max': Parameter(130, 0, 255, int),
    'hue_saturation_min': Parameter(120, 0, 255, int),
    'hue_saturation_max': Parameter(255, 0, 255, int),
    'red_min': Parameter(220, 0, 255, int),
    'red_max': Parameter(255, 0, 255, int),
    'green_min': Parameter(220, 0, 255, int),
    'green_max': Parameter(255, 0, 255, int),
}

class Pipeline(object):
    def __init__(self, calibration:camera.CalibrationParameters=camera.default_calibration):
        self.camera = camera.Camera()
        self.camera.calibrate(parameters=calibration, display=False)
        self.lane_detector = lanes.LaneDetector()

        # Define conversions in x and y from pixels space to meters
        self.metres_per_pixel_y = _metres_per_pixel_y # meters per pixel in y dimension
        self.metres_per_pixel_x = _metres_per_pixel_x # meters per pixel in x dimension

        # Filter parameters
        self.filter_params = dict([(p, filter_params[p].value) for p in filter_params])
    
    def reset(self):
        self.lane_detector.reset()
    
    def run(self, video_file:str, output_file:str):
        clip = VideoFileClip(video_file)

        # img = clip.get_frame(0)
        # out = p.step(img)
        # image.write(out, 'output.jpg', rgb=True)

        #.subclip(0,5)
        # clip = clip.subclip(21, 27)  # TODO remove this
        # clip = clip.subclip(37, 43)  # TODO remove this
        processed_clip = clip.fl_image(lambda frame: self.step(frame, rgb=True))
        processed_clip.write_videofile(output_file, audio=False)
    
    def debug(self, img:np.array) -> np.array:
        undistorted = self.camera.undistort(img)
        warped_unfiltered = warp.birds_eye.transform(undistorted)
        filtered = filters.main(undistorted, **self.filter_params)
        warped = warp.birds_eye.transform(filtered)
        leftx, lefty, rightx, righty, debug_img = self.lane_detector.debug(warped)
        vis, radius_of_curvature, lane_deviation = lanes.visualize(warped_unfiltered,
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
        if lane_deviation < -0.01:
            deviation_msg = 'Vehicle is %.2fm left of center' % abs(lane_deviation)
        elif lane_deviation > 0.01:
            deviation_msg = 'Vehicle is %.2fm right of center' % abs(lane_deviation)
        cv2.putText(summed, deviation_msg,
            (50,200), font, 2, (255,255,255), 2, cv2.LINE_AA)

        # import matplotlib.pyplot as plt

        # n_show = 7
        # _fig, axes = plt.subplots(n_show, 1, figsize=(15, 8*n_show))
        ctr = 0

        # def show(img):
        #     nonlocal ctr
        #     axes[ctr].imshow(img)
        #     ctr += 1

        def show(img):
            nonlocal ctr
            image.write(img, 'output_images/pipeline_%d.jpg' % ctr)
            ctr += 1

        show(undistorted)
        show(warped)
        show(filtered*255)
        show(debug_img)
        show(vis)
        show(unwarped_vis)
        show(summed)

        return summed
    
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
        warped_unfiltered = warp.birds_eye.transform(undistorted)
        filtered = filters.main(undistorted, rgb=rgb, **self.filter_params)
        # filtered = filtered.astype(np.int32) * 255
        # return np.stack([filtered, filtered, filtered], axis=-1)
        warped = warp.birds_eye.transform(filtered)
        leftx, lefty, rightx, righty = self.lane_detector.update(warped)
        vis, radius_of_curvature, lane_deviation = lanes.visualize(warped_unfiltered,
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

            def show(img, cmap=None):
                nonlocal ctr
                if cmap is not None:
                    axes[ctr].imshow(img, cmap=cmap)
                else:
                    axes[ctr].imshow(img)
                ctr += 1

            show(undistorted)
            show(filtered, cmap='gray')
            show(warped, cmap='gray')
            show(vis)
            show(unwarped_vis)
            show(summed)

        return summed


def main():
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


if __name__ == '__main__':
    # import sys
    # inp = sys.argv[1]
    # p = Pipeline()
    # img = image.read(inp)
    # p.debug(img)
    main()
