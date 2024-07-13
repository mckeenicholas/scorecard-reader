import os
from glob import glob
import cv2

def crop_images(src, height: int):
    x, y, _ = src.shape
    aspect_ratio = y / x
    new_width = int(height * aspect_ratio)
    return cv2.resize(src, (height, new_width))
    

if __name__ == "__main__":
    for image in glob("labeled_images\\19A1_499.jpg"):
        filename = os.path.splitext(os.path.basename(image))[0]
        img = cv2.imread(image)
        crop_images(img, 28)
    