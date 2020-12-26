#import sys
import os
import cv2 as cv
import numpy as np
import image_io
#from matplotlib import pyplot as plt

# INPUT_DIR/template.tiff
# INPUT_DIR/originals/*.tiff

#INPUT_DIR = "$HOME/Pictures/jupiter-saturno 2020-12-24-c/"
INPUT_DIR = "/Users/vruiz/Pictures/jupiter-saturno 2020-12-24-c/"
#INPUT_DIR = "./"
EXTENSION = ".tiff"

print("Loading template ... ", end='')
template = image_io.read(INPUT_DIR + 'template' + EXTENSION)
template_Y = cv.cvtColor(template, cv.COLOR_BGR2GRAY).astype(np.float32)
w = template_Y.shape[0]
h = template_Y.shape[1]
print(f"done (size={template_Y.shape})")

#def normalize(image):
#    max = image.max()
#    min = image.min()
#    max_min = max - min
#    normal = (image - min) / max_min
#    #normal *= 65535
#    #normal = normal.astype(np.uint16)
#    return normal, max, min

prefix = INPUT_DIR + "full_size/"
for root, dirs, files in os.walk(prefix):
    files.sort()
    counter = 0
    for target_name in files:
        image_fn = prefix + target_name
        image = image_io.read(image_fn)
        print("Searching template in ...", image_fn)
        image_Y = cv.cvtColor(image, cv.COLOR_BGR2GRAY).astype(np.float32)
        res = cv.matchTemplate(image_Y, template_Y, cv.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        top_left = max_loc
        print("Template found at", top_left)
        extraction = image[top_left[1]:top_left[1]+w,
                           top_left[0]:top_left[0]+h]
        extract_fn = INPUT_DIR + f"extractions/{counter}" + EXTENSION
        print("Writting ...", extract_fn)
        #cv.imwrite(extract_fn, extraction.astype(np.uint16))
        cv.imwrite(extract_fn, extraction)
        counter += 1
        #bottom_right = (top_left[0] + h, top_left[1] + w)
        #cv.rectangle(image, top_left, bottom_right, 65535, 2)
        #normal = (normalize(image)[0]*255).astype(np.int)
        #plt.subplot(111), plt.imshow(normal, cmap = 'gray')
        #plt.show()
