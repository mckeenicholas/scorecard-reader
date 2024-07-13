import glob
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def seperate_digits(img):
    vis = img.copy()
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(img)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    cv2.polylines(vis, hulls, 1, (0, 255, 0))
    # for i, contour in enumerate(hulls):
    #     cv2.drawContours(vis, [contour], -1, (255, 255, 255), -1)
    #     # x, y, w, h = cv2.boundingRect(contour)

    cv2.imshow("img", vis)
    cv2.waitKey()

    return vis
        

if __name__ == "__main__":
    for image in glob.glob("labeled_images\*.jpg"):
        filename = os.path.splitext(os.path.basename(image))[0]
        img = cv2.imread(image)
        output = seperate_digits(img)
        cv2.imwrite(f"seperated_images\{filename}.jpg", output)