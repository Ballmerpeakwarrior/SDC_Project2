import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
 
img = plt.imread('test_images/straight_lines1.jpg')
imgmod = plt.imread('test_images/straight_lines1.jpg')
 
trap = np.array([[[240,686],[1055,675],[690,450],[587,450]]], np.int32)
wwww = np.array([[[300,img.shape[1]],[950,img.shape[1]],[950,0],[300,0]]], np.int32)
img_mod = cv2.polylines(imgmod, [trap], True, (0,255,0),3)
img_mod = cv2.polylines(img_mod, [wwww], True, (0,0,255),3)

plt.figure(1)
plt.imshow(img_mod)

src = np.float32(trap)
dst = np.float32(wwww)
M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]), flags=cv2.INTER_LINEAR)

plt.figure(2)
plt.imshow(warped)

cv2.waitKey(0)
cv2.destroyAllWindows()

plt.show()