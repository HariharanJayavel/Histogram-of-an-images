# Histogram of an images
## Aim
To obtain a histogram for finding the frequency of pixels in an Image with pixel values ranging from 0 to 255. Also write the code using OpenCV to perform histogram equalization.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Read the gray and color image using imread()

### Step2:
Print the image using imshow().



### Step3:
Use calcHist() function to mark the image in graph frequency for gray and color image.

### step4:
Use calcHist() function to mark the image in graph frequency for gray and color image.

### Step5:
The Histogram of gray scale image and color image is shown.


## Program:
```python
# Developed By HARIHARAN J 
# Register Number: 212223240047

import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('parrot.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')
plt.title('Original Image')
plt.show()
```
#### Display the images
```python
plt.hist(img.ravel(),256,range = [0, 256]);
plt.title('Original Image')
plt.show()
```
#### Equalize histogram
```python
img_eq = cv2.equalizeHist(img)
```
#### Display the images.
```python
plt.hist(img_eq.ravel(), 256, range = [0, 256])
plt.title('Equalized Histogram')
```
#### Display the images.
```python
plt.imshow(img_eq, cmap='gray')
plt.title('Original Image')
plt.show()
```
#### Read the color image.
```
img = cv2.imread('parrot.jpg', cv2.IMREAD_COLOR)
```
#### Convert to HSV.
```
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
```
#### Perform histogram equalization only on the V channel, for value intensity.
```
img_hsv[:,:,2] = cv2.equalizeHist(img_hsv[:, :, 2])
```
#### Convert back to BGR format.
```
img_eq = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
```
#### Display the Equalized Image.
```
plt.imshow(img_eq[:,:,::-1]) 
plt.title('Equalized Image')
plt.show()
```
#### Histogram Equalized
```
plt.hist(img_eq.ravel(),256,range = [0, 256])
plt.title('Histogram Equalized')
plt.show()
```
#### Display the images.
```
plt.figure(figsize = (20,10))
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
```
#### Display the histograms.
```
plt.figure(figsize = [15,4])
plt.subplot(121)
plt.hist(img.ravel(),256,range = [0, 256])
plt.title('Original Image')
plt.subplot(122)
plt.hist(img_eq.ravel(),256,range = [0, 256])
plt.title('Histogram Equalized')
```
## Output:

### Histogram Equalized

<img width="1270" height="626" alt="image" src="https://github.com/user-attachments/assets/7bf0f48f-7a01-4185-a103-c3bb0084b36f" />


## Result: 
Thus the histogram for finding the frequency of pixels in an image with pixel values ranging from 0 to 255 is obtained. Also,histogram equalization is done for the gray scale image using OpenCV.
