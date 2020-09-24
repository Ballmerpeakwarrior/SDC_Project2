import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

##### ----- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images ----- #####

#Import Calibration images and initialize variables
CalibrationImages = glob.glob('camera_cal/calibration*.jpg')
objpoints = [] #Object Points Array initialization
imgpoints = [] #Image Points Array initialization
nx = 9 #Number of x corners on chessboard
ny = 6 #Nubmer of y corners on chessboard
objp = np.zeros((nx*ny,3), np.float32) #To be inserted into objpoints array
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) #Array with all the object points

for fname in CalibrationImages:
    #Import image
    img = cv2.imread('camera_cal/calibration2.jpg')
    plt.figure(1)
    plt.imshow(img)

    #Convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    plt.figure(2)
    plt.imshow(gray, cmap = 'gray')

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret == True:
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        plt.imshow(img)

        #Add objpoints/imgpoints to initialized array
        imgpoints.append(corners)
        objpoints.append(objp)

#Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#Undistortion of chessboard image
undist = cv2.undistort(img, mtx, dist, None, mtx)
plt.figure(3)
plt.imshow(undist)

##### ----- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images ----- #####


plt.show()