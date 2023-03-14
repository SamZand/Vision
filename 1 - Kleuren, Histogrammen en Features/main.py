from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage import color, io


def preserve_color_range(image, color_range):
    """
    Returns een kopie van de image met alle pixels buiten het opgegeven kleurbereik ingesteld op grijstinten.
    """

    hue = color.rgb2hsv(image)[:, :, 0]
    grayscale = color.gray2rgb(color.rgb2gray(image))

    color_mask = np.logical_and(hue >= color_range[0], hue <= color_range[1])
    grayscale[color_mask] = image[color_mask]

    return grayscale


def plot_hue_histogram(ax, image, title):
    """
    Plots een histogram van de tintwaarden in het image.
    """
    bins = 20
    range_ = (0, 1)

    image_hue = color.rgb2hsv(image)[:, :, 0].flatten()
    ax.hist(image_hue, bins=bins, range=range_)
    ax.set_title(title)


def main():
    fig, ((img1, img2, img3), (hist1, hist2, hist3)) = plt.subplots(2, 3)

    image = io.imread(
        "1 - Kleuren, Histogrammen en Features\color.jpg").astype(float)

    img1.imshow(image.astype(int))
    plot_hue_histogram(hist1, image, "Original")

    red_grayscale = preserve_color_range(image, (345 / 360, 360 / 360))
    img2.imshow(red_grayscale.astype(int))
    plot_hue_histogram(hist2, red_grayscale, "Red filter")

    blue_grayscale = preserve_color_range(image, (150 / 360, 210 / 360))
    img3.imshow(blue_grayscale.astype(int))
    plot_hue_histogram(hist3, blue_grayscale, "Blue filter")

    plt.show()


if __name__ == '__main__':
    main()
