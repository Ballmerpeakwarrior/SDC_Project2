import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

#Camera Calibration

img = mpimg.imread('camera_cal/calibration1.jpg')
plt.imshow(img)


#Number of corners on chessboard
nx = 9
ny = 6


plt.show()