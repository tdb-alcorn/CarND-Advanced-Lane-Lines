## Writeup

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[pipeline0]: ./output_images/pipeline_0.jpg "Undistorted"
[pipeline1]: ./output_images/pipeline_1.jpg "Perspective transform"
[pipeline2]: ./output_images/pipeline_2.jpg "Filtered"
[pipeline3]: ./output_images/pipeline_3.jpg "Window search"
[pipeline4]: ./output_images/pipeline_4.jpg "Polynomial fit"
[pipeline5]: ./output_images/pipeline_5.jpg "Unwarped"
[pipeline6]: ./output_images/pipeline_6.jpg "Final output"
[undistort]: ./output_images/undistorted_calibration.jpg "Calibration"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in `mycv/camera.py`.

As in the coursework, `objpoints` contains many copies of the same object points based on the assumption that we are undistorting to a fixed-z chessboard.  `imgpoints` contains all the corners detected in the calibration images.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][undistort]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][pipeline0]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of saturation, hue and Sobel gradient thresholds to generate a binary image, which can be viewed in `mycv/filters.py`. I used Jupyter's interactive widgets to fine tune the values of the thresholds used, ultimately arriving at the values listed in `filters.py#main()`. I also used histogram equalization to improve the contrast in the hue channel before thresholding. The hue channel is also post-filtered with a second saturation filter to help remove background noise. Here's an example of my output for this step:

![alt text][pipeline2]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform appears in `mycv/warp.py`. The birds-eye perspective transform parameters are hardcoded as follows:

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 603, 458      | 450, 0        | 
| 683, 461      | 850, 0        |
| 1063, 719     | 850, 719      |
| 249, 719      | 450, 719        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image. Here is an example of a perspective transformed image:

![alt text][pipeline1]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?


I used a combination of window search and polynomial searching to identify lane pixels, which can be seen in `mycv/lanes.py`. The lane detector first uses window search, and then subsequently uses polynomial search. It reverts back to window search if it cannot find enough lane pixels within the polynomial search area. Here is an example of the results of a window search:

![alt text][pipeline3]

Here is an example of the results of a polynomial search:

![alt text][pipeline4]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In `mycv/poly.py` I implemented `curvature()` and `convert_units()` functions that compute the curvature formula and change units for a given polynomial respectively. I applied these function in `mycv/lanes.py#visualize()`, which uses runs them on the polynomial fits for each lane line. I then computed the vehicles position by measuring the centre of the lane and calculating the image centre's deviation from the centre.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implented `mycv/warp.py#inverse_transform()` which computes the inverse transform for a given transform. I then use this in `mycv/pipeline.py` to unwarp the output of the lane search method, like so:

![alt text][pipeline5]

I then sum this with the original image and draw the analysis output on top, as in:

![alt text][pipeline6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I took the same basic approach that was outlined in the coursework but I tried to modularize it as much as possible to make debugging and improving the pipeline easier. Tuning the filter parameters using Jupyter notebook interactive widgets made it much easier to get a good result on the project video, but this also means that my filters are probably overfit for these particular images and may not be as effective in other conditions. 

Looking at the results of my pipeline on the [challenge video](./challenge_video_output.mp4) it is clear that the filters are unable to exclude non-lane markings on the road, which confuses the lane detector. However, it is hard to see how this filter-based approach can do better without lots of tedious fine-tuning, which is again prone to overfitting.

If I were to pursue this project further I would try to think of entirely new types of filters that might enable cleaner inputs to the lane fitter.