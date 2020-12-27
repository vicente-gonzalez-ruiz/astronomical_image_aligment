import numpy as np
import cv2 as cv
import os
import astroalign as aa
import image_io

MAX_NUMBER_OF_IMAGES = 500
MAX_CONTROL_POINTS = 50
INPUT_DIR = "/Users/vruiz/Pictures/jupiter-saturno 2020-12-24-c/"
EXTENSION = ".tiff"

# Dark image. This image registerizing tool supposes that all the dark
# image is the same for all the input images.
dark_image = image_io.read(INPUT_DIR + "dark" + EXTENSION).astype(np.float32)

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

prefix = INPUT_DIR + "full_size/"
for root, dirs, files in os.walk(prefix):
    files.sort()
    source_name = files.pop(0)
    source_image = image_io.read(prefix + source_name).astype(np.float32)
    source_image -= dark_image
    counter = 1
    for target_name in files:
        source_image_luma = normalize(cv.cvtColor(source_image, cv.COLOR_BGR2GRAY))[0]
        #_, source_image_luma = cv.threshold(src=source_image_luma,
        #                                    thresh=1,
        #                                    maxval=65535,
        #                                    type=cv.THRESH_OTSU)
        # source_image_luma = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2
        cv.imwrite("{:03d}_source.tiff".format(counter), source_image_luma)
        target_image = image_io.read(prefix + target_name).astype(np.float32)
        #target_image = image_io.read(prefix + source_name).astype(np.float32)
        target_image -= dark_image
        accumulated_image = target_image
        print("Projecting", source_name, "to", target_name)
        target_image_luma = normalize(cv.cvtColor(target_image, cv.COLOR_BGR2GRAY))[0]
        cv.imwrite("{:03d}_target.tiff".format(counter), target_image_luma)
        #_, target_image_luma = cv.threshold(src=target_image_luma,
        #                                    thresh=1,
        #                                    maxval=65535,
        #                                    type=cv.THRESH_OTSU)
        try:
            transf, (source_list, target_list) = aa.find_transform(source = source_image_luma, target = target_image_luma, max_control_points = MAX_CONTROL_POINTS)
            projection_0, footprint = aa.apply_transform(transf, source_image[:,:,0], target_image[:,:,0])
            projection_1, footprint = aa.apply_transform(transf, source_image[:,:,1], target_image[:,:,1])
            projection_2, footprint = aa.apply_transform(transf, source_image[:,:,2], target_image[:,:,2])
            projection_image = np.stack([projection_0, projection_1, projection_2], axis=2)
            cv.imwrite("{:03d}.tiff".format(counter), projection_image.astype(np.uint8))
            accumulated_image += projection_image
            source_image = target_image
            counter += 1
            output_image = accumulated_image / counter
            print("Normalizing", end=' ', flush=True)
            output_image, max, min = normalize(output_image)
            print(max, min)
            print(f"Writting output_{counter-2}.tiff")
            cv.imwrite(f"output_{counter-2}.tiff", output_image)
        except aa.MaxIterError:
            print("Unable to align", source_name)
        if counter > MAX_NUMBER_OF_IMAGES:
            break

print("done")
