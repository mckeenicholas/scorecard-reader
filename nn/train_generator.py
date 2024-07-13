import torch
import torchvision.transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import cv2
import numpy as np
import torch

# Load the MNIST dataset
transform = T.Compose([T.ToTensor()])
mnist_dataset = MNIST(root='.', train=True, download=True, transform=transform)
mnist_loader = DataLoader(mnist_dataset, batch_size=1, shuffle=True)

# Function  perspective transform
def apply_perspective_transform(image, distortion_scale=0.5):
    transform = T.RandomPerspective(distortion_scale=distortion_scale, p=1.0)
    return transform(image)

# Function to create a canvas and place  gaps
def place_images_on_canvas(images, gap_range):
    height, width = images[0].shape[2], images[0].shape[3]
    canvas_width = width * len(images) + gap_range[1] * (len(images) - 1)
    canvas = torch.zeros((height, canvas_width))

    current_x = 0
    for img in images:
        im_squeeze = img.squeeze((0, 1))

        coords = torch.sum(im_squeeze, dim=0)
        bounds = torch.argwhere(coords)
        left_min ,= bounds[0]
        right_min ,= bounds[-1]

        gap = np.random.randint(*gap_range)
        current_x = max(current_x + gap, 0)  # Ensure start_x is non-negative
        width = right_min - left_min
        end = current_x + width

        canvas[:, current_x:end] += im_squeeze[:, left_min:right_min]
        current_x += width

    return canvas

# Function to generate a string of numbers
def generate_random_string_of_numbers(length, gap_range):
    images = []
    labels = []
    for _ in range(length):
        image, label = next(iter(mnist_loader))
        transformed_image = apply_perspective_transform(image)
        images.append(transformed_image)
        labels.append(str(label.item()))

    canvas = place_images_on_canvas(images, gap_range)
    return canvas, ''.join(labels)

#  string of numbers with their stitched image
length_of_string = 5  # for example, a string of 5 numbers
gap_range = (-2, 5)
stitched_image, random_string = generate_random_string_of_numbers(length_of_string, gap_range)

# Remove the singleton dimension
stitched_image_np = stitched_image.squeeze(0).numpy()

# Debugging: Check the range of pixel values
print("Min pixel value:", np.min(stitched_image_np))
print("Max pixel value:", np.max(stitched_image_np))

# Normalize the image to [0, 255] and convert to uint8 for saving
stitched_image_np = (stitched_image_np * 255).astype(np.uint8)

end ,= np.argwhere(np.sum(stitched_image_np, axis=0))[-1]
crop = stitched_image_np[:, :end]

print(crop.shape)

cv2.imwrite("test_stitched.png", crop)

print("Generated string of numbers:", random_string)
