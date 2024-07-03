import numpy as np
import cv2 as cv

# All kernels
blur = np.array([[0.0625, 0.125, 0.0625],
                 [0.125, 0.25, 0.125],
                 [0.0625, 0.125, 0.0625]])
bottom_sobel = np.array([-1, -2, -1, 0, 0, 0, 1, 2, 1]).reshape(3, 3)
emboss = np.array([-2, -1, 0, -1, 1, 1, 0, 1, 2]).reshape(3, 3)
identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
left_sobel = np.array([1, 0, -1, 2, 0, -2, 1, 0, -1]).reshape(3, 3)
outline = np.array([-1, -1, -1, -1, 8, -1, -1, -1, -1]).reshape(3, 3)
right_sobel = np.array([-1, 0, 1, -2, 0, 2, -1, 0, 1]).reshape(3, 3)
sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
top_sobel = np.array([1, 2, 1, 0, 0, 0, -1, -2, -1]).reshape(3, 3)
sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

def applyKernal(image, kernal):
    if len(image.shape) == 2:  # Grayscale or binary image
        updated = convolute(image, kernal)
        updated = cv.convertScaleAbs(updated)
    else:  # Color image
        R, G, B = cv.split(image)
        red = convolute(R, kernal)
        green = convolute(G, kernal)
        blue = convolute(B, kernal)
        updated = cv.merge((red, green, blue))
        updated = cv.convertScaleAbs(updated)
    return updated

def convolute(image, kernal):
    height, width = image.shape
    kheight, kwidth = kernal.shape
    padded_array = np.pad(image, pad_width=kwidth // 2)
    resultant_image = np.zeros((height, width), dtype=int)

    for i in range(height):
        for j in range(width):
            mat = padded_array[i:i+kheight, j:j+kwidth]
            if mat.shape == kernal.shape:
                resultant_image[i, j] = np.sum(mat * kernal)

    return resultant_image

path = input("Enter path of image: ")
image = cv.imread(path)

if image is None:
    print("Error: Could not open or find the image.")
else:
    image = cv.resize(image, (400, 450))  
    print("\n\t\tWait.....")
    if len(image.shape) == 3:  # Color image
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    new_image = applyKernal(image, sharpen)
    new_image = cv.resize(new_image, (400, 450))  

    # Display images using OpenCV
    if len(image.shape) == 3:  # Color image
        cv.imshow('Original', cv.cvtColor(image, cv.COLOR_RGB2BGR))
        cv.imshow('Filtered', cv.cvtColor(new_image, cv.COLOR_RGB2BGR))
    else:  # Grayscale or binary image
        cv.imshow('Original', image)
        cv.imshow('Filtered', new_image)

    # Wait indefinitely until a key is pressed
    cv.waitKey(0)
    # Destroy all OpenCV windows
    cv.destroyAllWindows()
