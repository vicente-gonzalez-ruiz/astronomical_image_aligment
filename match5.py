import numpy as np
import cv2 as cv
import os
import astroalign as aa

MAX_NUMBER_OF_IMAGES = 500

def read(fn):
    print("Reading", fn, end= ' ', flush=True)
    img = cv.imread(fn, cv.IMREAD_UNCHANGED)
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

# Dark image
dark = read("dark.tiff")

for root, dirs, files in os.walk("./images/"):
    files.sort()
    first = files.pop(0)
    target = read_image(first)
    target -= dark
    target_luma = cv.cvtColor(target, cv.COLOR_BGR2GRAY)
    accumulated = target
    counter = 1
    for filename in files:
        source = read_image(filename)
        print("Projecting", filename)
        source_luma = cv.cvtColor(source, cv.COLOR_BGR2GRAY)
        try:
            transf, (source_list, target_list) = aa.find_transform(source_luma, target_luma)
            projection_0, footprint = aa.apply_transform(transf, source[:,:,0], target[:,:,0])
            projection_1, footprint = aa.apply_transform(transf, source[:,:,1], target[:,:,1])
            projection_2, footprint = aa.apply_transform(transf, source[:,:,2], target[:,:,2])
            projection = np.stack([projection_0, projection_1, projection_2], axis=2)
            cv.imwrite("{:03d}.tif".format(counter), projection.astype(np.uint8))
            accumulated += projection
            target = accumulated
            counter += 1
        except aa.MaxIterError:
            print("Unable to align", filename)
        if counter > MAX_NUMBER_OF_IMAGES:
            break

accumulated /= counter
print(accumulated.max(), accumulated.min())
accumulated = accumulated.astype(np.uint8)

cv.imwrite("output.tif", accumulated)

print("done")
