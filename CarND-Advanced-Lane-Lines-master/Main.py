import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

#Camera Calibration
CalibrationImages = glob.glob('camera_cal/calibration*.jpg')


img = cv2.imread('camera_cal/calibration2.jpg')
plt.figure(1)
plt.imshow(img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.figure(2)
plt.imshow(gray, cmap = 'gray')

#Number of corners on chessboard
nx = 9
ny = 6

# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

#Initialize objpoints/imgpoints array
objpoints = []
imgpoints = []

objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

# If found, draw corners
if ret == True:
    # Draw and display the corners
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    plt.imshow(img)

    #Add objpoints/imgpoints to initialized array
    imgpoints.append(corners)
    objpoints.append(objp)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)





plt.show()