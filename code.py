import cv2
import numpy as np
import matplotlib.pyplot as plt 
image = cv2.imread('SDC/Image_Database/635948.jpg')

lane_image = np.copy(image)
# Convert to a supported data type
lane_image = lane_image.astype(np.float32)
resized = cv2.resize(lane_image, (800, 450), interpolation=cv2.INTER_AREA)

def Canny(c1):
    grey_image = cv2.cvtColor(c1, cv2.COLOR_RGB2GRAY)
    Blur_image = cv2.GaussianBlur(grey_image, (5, 5), 0)
    return cv2.Canny(Blur_image, 50, 150)

def regionOfInterest(c2):
    heightImage = c2.shape[0]
    widthImage = c2.shape[1]
    triangle = np.array([(380, heightImage), (widthImage, heightImage),(widthImage, heightImage-65), (392, 215)])
    mask = np.zeros_like(c2)
    cv2.fillPoly(mask, [triangle], 255)
    masked_image = cv2.bitwise_and(c2, mask)
    return masked_image

canny_image = Canny(resized)
cropped_image = regionOfInterest(canny_image)
RoadLaneLines = cv2.HoughLinesP()
cv2.imshow('result', cropped_image)
cv2.waitKey(0)
