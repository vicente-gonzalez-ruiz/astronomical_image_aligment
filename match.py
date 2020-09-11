import numpy as np
import cv2 as cv
import os

images = []

# Read the images
for root, dirs, files in os.walk("."):
    for filename in files:
        print("Processing ", filename)
        images.append(cv.imread(filename),
                      cv2.IMREAD_UNCHANGED).astype(np.float32)

first_image = images[0]
rest_images = []
for i in range(len(images)-1):
    rest_images.append(images[i+1])

def estimate_motion(first, second):
    return cv.calcOpticalFlowFarneback(first, second, None, 0.5, 3,
                                       15, 3, 5, 1.2, 0)

def project(second, motion):
    height, width = motion.shape[:2]
    map_x = np.tile(np.arange(width), (height, 1))
    map_y = np.swapaxes(np.tile(np.arange(height), (width, 1)), 0, 1)
    map_xy = (motion + np.dstack((map_x, map_y))).astype('float32')

    return cv2.remap(second, map_xy, None,
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)

projections = []
projections.append(first_image)

for image in images:
    motion = estimate_motion(first_image, image)
    projections.append(project(image, motion))

accumulated = first_image
for projection in projections:
    accumulated += projection

accumulated =/ len(images)

cv.imwrite("output.png", accumulated)
