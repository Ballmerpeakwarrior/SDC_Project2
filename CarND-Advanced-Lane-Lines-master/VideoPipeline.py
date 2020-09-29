import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip

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
def process_image(image):
    #Undistort test image
    undist = cv2.undistort(image, mtx, dist, None, mtx)

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

    trap = np.array([[[240,686],[1055,675],[690,450],[587,450]]], np.int32)
    wwww = np.array([[[300,img.shape[1]],[950,img.shape[1]],[950,0],[300,0]]], np.int32)
    src = np.float32(trap)
    dst = np.float32(wwww)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(combined_binary, M, (combined_binary.shape[1],combined_binary.shape[0]), flags=cv2.INTER_LINEAR)

    bottom_half = []
    bottom_half = warped[warped.shape[0]//2:,:]
    histogram = np.sum(bottom_half, axis=0)

    midpoint = np.int(histogram.shape[0]//2) # Width of the image/2
    leftx_base = np.argmax(histogram[:midpoint]) # Index of point in histogram that has max value from 0 - midpoint
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint # Index of point in histogram that has max value from midpoint to end. Midpoint is added to get index with reference to 0

    nwindows = 9
    margin = 100
    minpix = 50
    out_img = np.dstack((warped, warped, warped))

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero (i.e. activated) pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped.shape[0] - (window+1)*window_height
        win_y_high = warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700

    left_curverad = ((1 + (2*left_fitx[0]*y_eval*ym_per_pix + left_fitx[1])**2)**1.5) / np.absolute(2*left_fitx[0])
    right_curverad = ((1 + (2*right_fitx[0]*y_eval*ym_per_pix + right_fitx[1])**2)**1.5) / np.absolute(2*right_fitx[0])
    print(left_curverad, right_curverad)

    vehicle_l = midpoint - left_fitx[719]
    vehicle_r = right_fitx[719] - midpoint

    print('Distance from left', vehicle_l)
    print('Distance from right', vehicle_r)

    uM = cv2.getPerspectiveTransform(dst, src)
    unwarped = cv2.warpPerspective(out_img, uM, (out_img.shape[1],out_img.shape[0]), flags=cv2.INTER_LINEAR)

    return unwarped

white_output = 'output_images/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)





