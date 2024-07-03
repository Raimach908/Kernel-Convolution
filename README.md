# Kernel-Convolution

## Overview

This Python script applies various image kernels (e.g., sharpening, blurring) to an image. It supports both color and grayscale images. The script utilizes OpenCV and NumPy for image processing.

## Features

- Load an image from a specified path.
- Resize the image to 400x450 pixels.
- Convert the image to RGB if it is a color image.
- Apply a specified kernel to the image (e.g., sharpening).
- Display both the original and filtered images.

## Requirements

- Python 3.x
- OpenCV (`cv2`): Install via `pip install opencv-python`
- NumPy: Install via `pip install numpy`

## Kernels Provided

- **Blur**: Smooths the image.
- **Bottom Sobel**: Detects vertical edges.
- **Emboss**: Creates an emboss effect.
- **Identity**: Placeholder kernel with no effect.
- **Left Sobel**: Detects horizontal edges.
- **Outline**: Enhances edges.
- **Right Sobel**: Detects vertical edges.
- **Sharpen**: Enhances details and sharpness.
- **Top Sobel**: Detects horizontal edges.
- **Sobel**: Basic Sobel edge detection.
