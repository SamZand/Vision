import numpy as np
import scipy
from scipy import ndimage as ndi
from skimage import data, feature, filters
from skimage.util import random_noise
from skimage.viewer import ImageViewer

# from skimage.viewer import ImageViewer


def edgeDetection1():
    image = data.camera()

    sobel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    sobel_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    edge_x = scipy.ndimage.convolve(image, sobel_x)
    edge_y = scipy.ndimage.convolve(image, sobel_y)
    newimage = scipy.sqrt(edge_x**2 + edge_y**2)
    viewer = ImageViewer(newimage)
    viewer.show()

    return newimage


def edgeDetection2():
    image = data.camera()
    scharr_x = [[3, 0, -3], [10, 0, -10], [3, 0, -3]]
    scharr_y = [[3, 10, 3], [0, 0, 0], [-3, -10, -3]]

    edge_x = scipy.ndimage.convolve(image, scharr_x)
    edge_y = scipy.ndimage.convolve(image, scharr_y)
    newimage = scipy.sqrt(edge_x**2 + edge_y**2)
    viewer = ImageViewer(newimage)
    viewer.show()


def edgeDetection3():
    image = data.camera()
    prewitt_x = [[1, 1, 1], [0, 0, 0], [-1, -1, -1]]
    prewitt_y = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]

    edge_x = scipy.ndimage.convolve(image, prewitt_x)
    edge_y = scipy.ndimage.convolve(image, prewitt_y)
    newimage = scipy.sqrt(edge_x**2 + edge_y**2)
    viewer = ImageViewer(newimage)
    viewer.show()


def main():
    image = data.camera()

    viewer = ImageViewer(image)
    viewer.show()

    # Filter 1: Sobel filter
    edgeDetection1()
    image = data.camera()
    image = filters.sobel(image)
    viewer = ImageViewer(image)
    viewer.show()

    # Filter 2: Scharr filter
    edgeDetection2()
    image = data.camera()
    image = filters.scharr(image)
    viewer = ImageViewer(image)
    viewer.show()

    # Filter 3: Prewitt filter
    edgeDetection3()
    image = data.camera()
    image = filters.prewitt(image)
    viewer = ImageViewer(image)
    viewer.show()

    # Filter 4: Canny filter
    image = data.camera()
    newimage = feature.canny(
        image, sigma=3, low_threshold=10, high_threshold=20)

    viewer = ImageViewer(newimage)
    viewer.show()


if __name__ == "__main__":
    main()
