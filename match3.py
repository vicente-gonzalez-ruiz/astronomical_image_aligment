import numpy as np
import cv2 as cv
import os
import astroalign as aa

MAX_NUMBER_OF_IMAGES = 20

def read_image(fn):
    print("Reading", fn, end= ' ', flush=True)
    img = cv.imread("./input/" + fn, cv.IMREAD_UNCHANGED)
    if img is None:
        print("Error reading", fn)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print("with shape", img.shape)
    img = img.astype(np.float32)
    print(img.max(), img.min())
    return img

for root, dirs, files in os.walk("./input/"):
    files.sort()
    first = files.pop(0)
    target = read_image(first)
    accumulated = target
    counter = 1
    for filename in files:
        source = read_image(filename)
        print("Projecting", filename)
        projection, footprint = aa.register(source, target,
                                            detection_sigma = 2,
                                            min_area = 9)
        cv.imwrite("{:03d}.tif".format(counter), projection.astype(np.uint8))
        accumulated += projection
        #target = source
        target = accumulated
        counter += 1
        if counter > MAX_NUMBER_OF_IMAGES:
            break

accumulated /= counter
print(accumulated.max(), accumulated.min())
accumulated = accumulated.astype(np.uint8)

cv.imwrite("output.tif", accumulated)

print("done")
