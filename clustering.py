import glob
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

K = 4

def seperate_digits(img):
    output = np.zeros_like(img)
    
    locations = np.argwhere(img > 100)

    loc_float = locations.astype(np.float32)

    cv2.imshow("cluser", img)
    cv2.waitKey(0)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center=cv2.kmeans(loc_float, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    red = np.array([255, 0, 0])
    green = np.array([0, 255, 0])
    blue = np.array([0, 0, 255])
    yellow = np.array([255, 255, 0])
    cyan = np.array([0, 255, 255])
    magenta = np.array([255, 0, 255])

    colors = np.array([red, green, blue, yellow, cyan, magenta])

    for color in range (K):
        t = label[:, 0]
        mask = np.argwhere(t == color)
        loc_to_use = locations[mask]
        loc_squeezed = loc_to_use[:, 0]

        output[loc_squeezed[:,0],loc_squeezed[:,1]] = colors[color % len(colors)]
    return output

if __name__ == "__main__":
    for image in glob.glob("labeled_images\*.jpg"):
        filename = os.path.splitext(os.path.basename(image))[0]
        img = cv2.imread(image)
        output = seperate_digits(img)
        cv2.imwrite(f"seperated_images/{filename}.jpg", output)