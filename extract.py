import sys
import os
import cv2 as cv
import numpy as np
import image_io

# INPUT_DIR/template.tiff
# INPUT_DIR/originals/*.tiff

#INPUT_DIR = "$HOME/Pictures/jupiter-saturno 2020-12-24-c/"
INPUT_DIR = "/Users/vruiz/Pictures/jupiter-saturno 2020-12-24-c/"
#INPUT_DIR = "./"
EXTENSION = ".tiff"

print("Loading template ... ", end='')
template = image_io.read(INPUT_DIR + 'template' + EXTENSION)
template_Y = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
print(f"done (size={template_Y.shape})")

print("Searching template in ... ", end='')
for root, dirs, files in os.walk(INPUT_DIR + "originals/"):
    files.sort()
    image_name = files.pop(0)
    print(image_name, end='')
    image = image_io.read(INPUT_DIR + "originals/" + image_name)
    counter = 0
    for target_name in files:
        image_Y = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        res = cv.matchTemplate(image_Y, template_Y, cv.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        print("found at ", min_loc, max_loc)
        extraction = image[min_loc[0]:min_loc[1], max_loc[0]:max_loc[1]]
        print(extraction.shape)
        cv.imwrite(INPUT_DIR + f"extractions/{counter}" + EXTENSION, extraction)
        counter += 1
