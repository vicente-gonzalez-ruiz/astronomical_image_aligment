import numpy as np
import cv2 as cv

FILE_EXTENSION = "png"

def read(fn):
    image_name = fn + '.' + FILE_EXTENSION
    print("Reading", image_name, end= ' ', flush=True)
    img = cv.imread(image_name, cv.IMREAD_UNCHANGED)
    if img is None:
        print("Error reading", fn)
    #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    print("with shape", img.shape)
    img = img.astype(np.float32)
    print(img.max(), img.min())
    return img

def read_image(fn):
    fn = "./images/" + fn
    return read(fn)
