import numpy as np
import cv2 as cv
import os

images = []

# Read the images
for root, dirs, files in os.walk("./input/"):
    for filename in files:
        fn = "./input/" + filename
        print("Reading", fn)
        img = cv.imread(fn, cv.IMREAD_UNCHANGED)
        if img is None:
            print("Error reading", fn)
        images.append(img.astype(np.float32))

first_image = images[0]
rest_images = []
for i in range(len(images)-1):
    rest_images.append(images[i+1])

def estimate_motion(prev, next):
    prev = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    next = cv.cvtColor(next, cv.COLOR_BGR2GRAY)
    # prev, net, flow, pyr_scale, levels, winsize, iterations, poly_n,
    # poly_sigma, flags
    return cv.calcOpticalFlowFarneback(prev, next, None, 0.5, 3,
                                       3, 5, 10, 1.5, 0)


def project(image, motion):
    height, width = motion.shape[:2]
    map_x = np.tile(np.arange(width), (height, 1))
    map_y = np.swapaxes(np.tile(np.arange(height), (width, 1)), 0, 1)
    map_xy = (motion + np.dstack((map_x, map_y))).astype(np.float32)

    return cv.remap(image, map_xy, None,
                    interpolation=cv.INTER_LINEAR,
                    borderMode=cv.BORDER_REPLICATE)


projections = []
counter = 0
for image in rest_images:
    print("Processing ...", end=' ', flush=True)
    motion = estimate_motion(first_image, image)
    #motion = np.random.rand(first_image.shape[0], first_image.shape[1], 2)*10
    #motion = np.zeros((first_image.shape[0], first_image.shape[1], 2))
    print("max_motion =", motion.max(), "min_motion =", motion.min())
    projection = project(image, motion)
    projection[:,:,1] = projection[:,:,2] = 0
    cv.imwrite("{:03d}.tif".format(counter), projection)
    counter += 1
    projections.append(projection)
    

print("Projected", len(projections), "images")
    
accumulated = first_image
for projection in projections:
    accumulated += projection
accumulated /= len(images)
accumulated = accumulated.astype(np.int8)

cv.imwrite("output.tif", accumulated)

print("done")
