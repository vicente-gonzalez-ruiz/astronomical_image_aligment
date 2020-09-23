import numpy as np
import cv2 as cv
import os
import astroalign as aa

MAX_NUMBER_OF_IMAGES = 500
MAX_CONTROL_POINTS = 50

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
dark_image = read("dark.tiff")

def normalize(image):
    max = image.max()
    min = image.min()
    max_min = max - min
    normal = (image - min) / max_min
    normal *= 65535
    normal = normal.astype(np.uint16)
    return normal, max, min

for root, dirs, files in os.walk("./images/"):
    files.sort()
    source_name = files.pop(0)
    source_image = read_image(source_name)
    source_image -= dark_image
    counter = 1
    for target_name in files:
        source_image_luma = cv.cvtColor(source_image, cv.COLOR_BGR2GRAY)
        target_image = read_image(target_name)
        target_image -= dark_image
        accumulated_image = target_image
        print("Projecting", source_name, "to", target_name)
        target_image_luma = cv.cvtColor(target_image, cv.COLOR_BGR2GRAY)
        try:
            transf, (source_list, target_list) = aa.find_transform(source = source_image_luma, target = target_image_luma, max_control_points = MAX_CONTROL_POINTS)
            projection_0, footprint = aa.apply_transform(transf, source_image[:,:,0], target_image[:,:,0])
            projection_1, footprint = aa.apply_transform(transf, source_image[:,:,1], target_image[:,:,1])
            projection_2, footprint = aa.apply_transform(transf, source_image[:,:,2], target_image[:,:,2])
            projection_image = np.stack([projection_0, projection_1, projection_2], axis=2)
            cv.imwrite("{:03d}.tif".format(counter), projection_image.astype(np.uint8))
            accumulated_image += projection_image
            source_image = target_image
            counter += 1
            output_image = accumulated_image / counter
            print("Normalizing", end=' ', flush=True)
            output_image, max, min = normalize(output_image)
            print(max, min)
            print(f"Writting output_{counter-2}.tif")
            cv.imwrite(f"output_{counter-2}.tif", output_image)
        except aa.MaxIterError:
            print("Unable to align", source_name)
        if counter > MAX_NUMBER_OF_IMAGES:
            break

print("done")
