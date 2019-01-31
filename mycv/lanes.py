from typing import Tuple

import cv2
import numpy as np

from . import poly, image


class LaneDetector(object):
    def __init__(self, minpix_poly:int=200):
        self.left: poly.Fit = poly.zero()
        self.right: poly.Fit = poly.zero()

        self.minpix_poly = minpix_poly
    
    def update(self, img:np.array):
        '''
        `minpix` is the minimum number of nonzero pixels that must be found in the
        search area for the search to succeed
        '''
        leftx, lefty, rightx, righty = poly_search(img, self.left, self.right, margin=100)

        if len(leftx) < self.minpix_poly or len(rightx) < self.minpix_poly:
            leftx, lefty, rightx, righty = window_search(img, nwindows=9, margin=100, minpix=50)

        # Fit new polynomials
        self.left = poly.fit(lefty, leftx)
        self.right = poly.fit(righty, rightx)


def draw_lanes(img:np.array, left:poly.Fit, right:poly.Fit) -> np.array:
    '''
    draw_polyfit fills the area between two polynomial fits
    
    assumes img is a top-down perspective image
    '''
    out_img = image.like(img, num_channels=3)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])

    left_fitx = poly.compute(left, ploty)
    right_fitx = poly.compute(right, poly)

    # # Plots the left and right polynomials on the lane lines
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')

    return out_img


def window_search(binary_warped:np.array, nwindows:int=9, margin:int=100, minpix:int=50):
    '''
    window_search takes a top-down image of a road and finds pixels
    corresponding to lane lines by performing a window-based search. It
    starts with a window centred on the peak activation in the bottom half of
    the image and searches upward, updating the position of the next window
    using the mean position of active pixels found in the previous window

    `nwindows` is the number of sliding windows
    `margin` is half the width of the windows
    `minpix` is the threshold for number of pixels needed to cause the window
    to be recentered
    '''
    # Take a histogram of the bottom half of the image
    histogram = np.sum(image.crop_to_bottom_half(binary_warped), axis=0)

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height

        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        in_window_left = np.zeros(nonzerox.shape, dtype=np.bool_)
        in_window_left[(nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high) & (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)] = True
        in_window_right = np.zeros(nonzerox.shape, dtype=np.bool_)
        in_window_right[(nonzerox >= win_xright_low) & (nonzerox < win_xright_high) & (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)] = True
        good_left_inds = [(nonzerox[i], nonzeroy[i]) for i in range(len(in_window_left)) if in_window_left[i]]
        good_right_inds = [(nonzerox[i], nonzeroy[i]) for i in range(len(in_window_right)) if in_window_right[i]]
        
        # Append these indices to the lists
        left_lane_inds.extend(good_left_inds)
        right_lane_inds.extend(good_right_inds)
        
        ### If we found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if np.sum(in_window_left.astype(np.int32)) > minpix:
            leftx_current = np.round(np.mean(nonzerox[in_window_left])).astype(np.int32)
        if np.sum(in_window_right.astype(np.int32)) > minpix:
            rightx_current = np.round(np.mean(nonzerox[in_window_right])).astype(np.int32)

    # Extract left and right line pixel positions
    leftx, lefty = zip(*left_lane_inds)
    rightx, righty = zip(*right_lane_inds)

    return leftx, lefty, rightx, righty


def poly_search(
    img:np.array,
    left_fit:poly.Fit,
    right_fit:poly.Fit,
    margin:int=100) -> Tuple[bool, poly.Fit, poly.Fit]:
    '''
    poly_search takes a top-down image of the road and finds pixels
    corresponding to lanes using a polynomial-based search. Given the
    polynomial of each lane from the previous frame, it gathers all active
    pixels within `margin` of it and fits a new polynomial to them.

    `margin` is the width around the previous polynomials to search
    '''
    # Grab activated pixels
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = poly.in_window(left_fit, nonzerox, nonzeroy, margin)
    right_lane_inds = poly.in_window(right_fit, nonzerox, nonzeroy, margin)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty
    