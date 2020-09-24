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
    img = cv2.imread(fname)
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

##### ----- Apply a distortion correction to raw images ----- #####
TestImages = glob.glob('test_images/test*.jpg')
for fname in TestImages:
    #Import test image
    img = cv2.imread(fname)

    #Undistort test image
    undist = cv2.undistort(img, mtx, dist, None, mtx)

##### ----- Use color transforms, gradients, etc., to create a thresholded binary image ----- #####
    #Grayscale & HLS Color Space
    gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
    hls = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
    S = hls[:,:,2]

    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(S)
    s_binary[(S >= s_thresh_min) & (S <= s_thresh_max)] = 1

    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    plt.figure(4)
    plt.imshow(combined_binary, cmap = 'gray')




plt.show()