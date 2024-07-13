import cv2
import numpy as np
import imutils

# Constants for image dimensions
WIDTH = 600
HEIGHT = 800

PIXEL_MAX_VALUE = 255
crop_offset_y = 5

def reorder_coordinates(coords):
    """Reorder coordinates based on the sum of x and y values."""
    order = np.argsort(np.sum(coords, axis=1))
    reordered_coords = coords[order]
    return reordered_coords

def crop_empty(img):

    empty_cols = np.argwhere(np.sum(img, axis=0) > PIXEL_MAX_VALUE * 2)

    if empty_cols.size == 0:
        return None

    low = max(empty_cols[0, 0] - 1, 0)
    high = min(empty_cols[-1, 0] + 1, img.shape[1] - 1)
    
    cols = np.arange(low, high)

    cropped_image = img[:, cols]

    return cropped_image

def process_image(name: str):
    # Load and preprocess image
    image = cv2.imread(name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)
    canny = cv2.Canny(blur, 0, 200)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 6)

    # Find contours
    contours = cv2.findContours(canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Find paper outline
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approximation) == 4:
            paper_outline = approximation
            break

    # Perspective transformation
    source = reorder_coordinates(np.array(paper_outline[:, 0, :], dtype=np.float32))
    dest = np.array([[0, 0], [WIDTH - 1, 0], [0, HEIGHT - 1], [WIDTH - 1, HEIGHT - 1]], dtype=np.float32)
    homography = cv2.getPerspectiveTransform(source, dest)
    warped = cv2.warpPerspective(thresh, homography, (WIDTH, HEIGHT), flags=cv2.INTER_LINEAR)

    # Morphological operations
    kernel_length = np.array(warped).shape[1] // 80
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Detect vertical and horizontal lines
    img_temp1 = cv2.erode(warped, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    img_temp2 = cv2.erode(warped, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

    # Combine vertical and horizontal lines
    alpha = 0.5
    beta = 1.0 - alpha
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)

    _, img_final_bin = cv2.threshold(img_final_bin, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Find and save contours
    contours, _ = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours_sorted = sorted(zip(contours, [cv2.boundingRect(c) for c in contours]), key=lambda b: b[1][0])[0]

    images = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 350 > w > 80 and h > 40:
            new_img = warped[y + crop_offset_y:y+h - crop_offset_y, x:x+w]

            cropped = crop_empty(new_img)

            images.append(cropped)

    return list(reversed(images[1:]))


if __name__ == "__main__":
    imgs = process_image("raw_images/107.jpg")
    for idx, img in enumerate(imgs):
        cv2.imwrite(f"output-{idx}.jpg", img)
