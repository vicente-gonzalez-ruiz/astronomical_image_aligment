import numpy as np
import cv2 as cv

def read(fn):
    #print("Reading", fn, end= ' ', flush=True)
    img = cv.imread(fn, cv.IMREAD_UNCHANGED)
    if img is None:
        print("Error reading", fn)
    #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #print("with shape", img.shape)
    img = img.astype(np.float32)
    #print(img.max(), img.min())
    return img
