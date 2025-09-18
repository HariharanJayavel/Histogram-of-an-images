import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('parrot.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.show()

# Display the images
plt.hist(img.ravel(),256,range = [0, 256])
plt.title('Original Image')
plt.show()

# Equalize histogram
img_eq = cv2.equalizeHist(img)

# Display the images.
plt.hist(img_eq.ravel(), 256, range = [0, 256])
plt.title('Equalized Histogram')

# Display the images.
plt.imshow(img_eq, cmap='gray')
plt.title('Original Image')
plt.show()

# Read the color image.
img = cv2.imread('parrot.jpg', cv2.IMREAD_COLOR)

# Convert to HSV.
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Perform histogram equalization only on the V channel, for value intensity.
img_hsv[:,:,2] = cv2.equalizeHist(img_hsv[:, :, 2])

# Convert back to BGR format.
img_eq = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

plt.imshow(img_eq[:,:,::-1]) 
plt.title('Equalized Image')
plt.show()

# Display the images.
#plt.figure(figsize = (20,10))
plt.subplot(221)
plt.imshow(img[:, :, ::-1])
plt.title('Original Color Image')
plt.subplot(222)
plt.imshow(img_eq[:, :, ::-1])
plt.title('Equalized Image')
plt.subplot(223)
plt.hist(img.ravel(),256,range = [0, 256])
plt.title('Original Image')
plt.subplot(224)
plt.hist(img_eq.ravel(),256,range = [0, 256])
plt.title('Histogram Equalized');plt.show()

# Display the histograms.
plt.figure(figsize = [15,4])
plt.subplot(121)
plt.hist(img.ravel(),256,range = [0, 256])
plt.title('Original Image')
plt.subplot(122)
plt.hist(img_eq.ravel(),256,range = [0, 256])
plt.title('Histogram Equalized')
