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