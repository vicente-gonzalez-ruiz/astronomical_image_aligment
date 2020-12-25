import os
import cv2 as cv
import numpy as np
from io_image import read as read
from io_image import read_image as read_image

MAX_NUMBER_OF_IMAGES = 500

template = read('template')
template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

for root, dirs, files in os.walk("./originals/"):
    files.sort()
    print(files)

    '''
    image_name = files.pop(0)
    image = read_image(image_name)
    counter = 1
    for target_name in files:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        res = cv.matchTemplate(image, template, cv.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        print(min_val, max_val, min_loc, max_loc)
    '''
