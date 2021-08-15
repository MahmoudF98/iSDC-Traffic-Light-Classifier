# Helper functions

import os
import glob # library for loading images from a directory
import matplotlib.image as mpimg
import cv2
import numpy as np



# This function loads in images and their labels and places them in a list
# The list contains all images and their associated labels
# For example, after data is loaded, im_list[0][:] will be the first image-label pair in the list
def load_dataset(image_dir):
    
    # Populate this empty image list
    im_list = []
    image_types = ["red", "yellow", "green"]
    
    # Iterate through each color folder
    for im_type in image_types:
        
        # Iterate through each image file in each image_type folder
        # glob reads in any image with the extension "image_dir/im_type/*"
        for file in glob.glob(os.path.join(image_dir, im_type, "*")):
            
            # Read in the image
            im = mpimg.imread(file)
            
            # Check if the image exists/if it's been correctly read-in
            if not im is None:
                # Append the image, and it's type (red, green, yellow) to the image list
                im_list.append((im, im_type))

    return im_list


def image_brightness_mask(cropped_image):
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    lower_value = 130
    upper_value = 255
    mask = cv2.inRange(v, lower_value, upper_value)
    masked_image = np.copy(cropped_image)
    masked_image[mask == 0] = 0

    return masked_image


def image_crop(rgb_image):
    cropped = np.copy(rgb_image)
    row_crop = 5
    col_crop = 12
    cropped = cropped[row_crop:-row_crop, col_crop:-col_crop, :]
    return cropped


def image_area_brightness(cropped_image):
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    vRed = sum(np.sum(v[:7, :], axis=0))
    vYellow = sum(np.sum(v[7:15, :], axis=0))
    vGreen = sum(np.sum(v[15:, :], axis=0))

    return vRed, vYellow, vGreen


def green_hue_mask(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    lower_green = np.array([36,50,50])
    upper_green = np.array([100,255,255])
    
    mask = cv2.inRange(hsv, lower_green, upper_green)

    masked_image = np.copy(rgb_image)
    masked_image[mask == 0] = 0

    return masked_image


def yellow_hue_mask(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([10,50,50])
    upper_yellow = np.array([35,255,255])
    
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    masked_image = np.copy(rgb_image)
    masked_image[mask == 0] = 0

    return masked_image


def red_hue_mask(rgb_image):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([8, 255, 255])
    mask0 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([165, 28, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask0 + mask1
    masked_image = np.copy(rgb_image)
    masked_image[mask == 0] = 0

    return masked_image


def avg_brightness(masked_image):
    hsv = cv2.cvtColor(masked_image, cv2.COLOR_RGB2HSV)
    sum_brightness = np.sum(hsv[:, :, 2])
    sum_saturation = np.sum(hsv[:, :, 1])

    avg = (sum_brightness + sum_saturation) / (32 * 32.0)
    return avg / 2


def get_brightest_area(red_brightness, yellow_brightness, green_brightness):
    if (red_brightness == green_brightness) and (red_brightness == yellow_brightness):
        return [1, 0, 0]
    if (red_brightness > yellow_brightness):
        if red_brightness > green_brightness:
            feature = [1, 0, 0]
        else:
            feature = [0, 0, 1]
    else:
        if yellow_brightness > green_brightness:
            feature = [0, 1, 0]
        else:
            feature = [0, 0, 1]

    return feature