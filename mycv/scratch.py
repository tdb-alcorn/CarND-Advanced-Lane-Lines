def corners_unwarp(img, nx, ny, mtx, dist):
    undist = cv2.undistort(img, mtx, dist)
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny))
    if ret:
        cv2.drawChessboardCorners(undist, (nx, ny), corners, ret)
        corner_idx = [0, nx-1, nx*ny-1, nx*(ny-1)]
        imgpoints = np.array([corners[i].reshape([-1]) for i in corner_idx], dtype=np.float32)
        x_size = undist.shape[1]
        y_size = undist.shape[0]
        offset = 100
        objpoints = np.array([(offset,offset), (x_size-offset, offset), (x_size-offset, y_size-offset), (offset, y_size-offset)], dtype=np.float32)
        M = cv2.getPerspectiveTransform(imgpoints, objpoints)
        warped = cv2.warpPerspective(undist, M, (undist.shape[1], undist.shape[0]))
    else:
        M = None
        warped = np.copy(img) 
    return warped, M

M = cv2.getPerspectiveTransform(src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)
warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)


mask = threshold(sobel_mag, tmin=20, tmax=100) | threshold(sobel_dir, tmin=0.7, tmax=1.3)

def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = compute_poly(left_fit, ploty)
    right_fitx = compute_poly(right_fit, ploty)
    
    return left_fitx, right_fitx, ploty

