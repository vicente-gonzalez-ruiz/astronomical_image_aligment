import numpy as np
import cv2 as cv
import os
import astroalign as aa

images = []

# Read the images
for root, dirs, files in os.walk("./input/"):
    files.sort()
    for filename in files:
        fn = "./input/" + filename
        print("Reading", fn)
        img = cv.imread(fn, cv.IMREAD_UNCHANGED)
        if img is None:
            print("Error reading", fn)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        images.append(img.astype(np.float32))

first_image = images[0]
rest_images = []
for i in range(len(images)-1):
    rest_images.append(images[i+1])

projections = []
counter = 0
for image in rest_images:
    print("Processing ...", end=' ', flush=True)
    source = image #cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    target = first_image #cv.cvtColor(first_image, cv.COLOR_BGR2GRAY)
    projection, footprint = aa.register(source, target)
    cv.imwrite("{:03d}.tif".format(counter), projection)
    counter += 1
    projections.append(projection)
    
print("Projected", len(projections), "images")
    
accumulated = first_image
for projection in projections:
    accumulated += projection
accumulated /= len(images)
accumulated = accumulated.astype(np.uint8)

cv.imwrite("output.tif", accumulated)

print("done")
