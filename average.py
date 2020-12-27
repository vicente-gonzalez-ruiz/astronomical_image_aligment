import numpy as np
import cv2 as cv
import os
import astroalign as aa
import image_io

MAX_NUMBER_OF_IMAGES = 500
INPUT_DIR = "/Users/vruiz/Pictures/jupiter-saturno 2020-12-24-c/"
EXTENSION = ".tiff"

def normalize(image):
    max = image.max()
    min = image.min()
    max_min = max - min
    normal = (image - min) / max_min
    #normal *= 65535
    normal *= 255
    #normal = normal.astype(np.uint16)
    normal = normal.astype(np.uint8)
    return normal, max, min

prefix = INPUT_DIR + "extracted/resized/"
print("prefix =", prefix)
for root, dirs, files in os.walk(prefix):
    files.sort()
    accumulated_name = files.pop(0)
    accumulated_image = image_io.read(prefix + accumulated_name).astype(np.float32)
    counter = 1
    for next_name in files:
        next_image = image_io.read(prefix + next_name).astype(np.float32)
        accumulated_image += next_image
        counter += 1
        output_image = accumulated_image / counter
        print("Normalizing", end=' ', flush=True)
        output_image, max, min = normalize(output_image)
        print(max, min)
        print(f"Writting output_{counter-2}.tiff")
        cv.imwrite(f"output_{counter-2}.tiff", output_image)
        if counter > MAX_NUMBER_OF_IMAGES:
            break

print("done")
