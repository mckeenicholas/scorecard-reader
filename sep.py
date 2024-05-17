import cv2
import numpy as np
import os
import glob

def digit_sep2(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mask = gray > 10

    top_border = np.argmax(mask, axis=0)
    neg_diff = np.argwhere(np.diff(top_border) > 0)
    pos_diff = np.argwhere(np.diff(top_border) < 0)

    print(neg_diff)
    print(pos_diff)


if __name__ == "__main__":
    for image in glob.glob("labeled_images\\19A1_499.jpg"):
        filename = os.path.splitext(os.path.basename(image))[0]
        img = cv2.imread(image)
        digit_sep2(img)