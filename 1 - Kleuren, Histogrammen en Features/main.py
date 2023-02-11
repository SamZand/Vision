import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.viewer import ImageViewer


def progress_bar(current, total):
    """ Prints progress bar. """
    percentage = int(100 * current / total)
    print("\r[", "#" * (percentage // 2), " " * (50 - percentage // 2), "]",
          f" {percentage}%", sep="", end="")


def colorFilter(image, lower, upper):

    # Create a binary mask to select pixels in the specified color range
    mask = np.logical_and(lower <= image, image <= upper).all(axis=-1)

    # Convert the selected pixels to their original RGB values
    image[mask] = np.round(image[mask]).astype(int)

    # Convert the rest of the pixels to grayscale
    gray = np.dot(image[~mask], [0.2126, 0.7152, 0.0722])
    gray = np.round(gray).astype(int)
    gray = np.stack([gray, gray, gray], axis=-1)

    # Combine the grayscale and color pixels to form the final output
    image[~mask] = gray

    return image


# def hue_histogram(image):
def hue_histogram(image):
    # Split the image into R, G and B channels
    red_channel = image[:, :, 0]
    green_channel = image[:, :, 1]
    blue_channel = image[:, :, 2]

    # Get the total number of pixels in the image
    total_pixels = image.shape[0] * image.shape[1]

    # Create a list to store the count of pixels for each intensity value in the channels
    red_count = [0] * 256
    green_count = [0] * 256
    blue_count = [0] * 256

    # Loop through each pixel in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Increment the count for each intensity value in the R, G and B channels
            red_count[red_channel[i, j]] += 1
            green_count[green_channel[i, j]] += 1
            blue_count[blue_channel[i, j]] += 1

    # Normalize the count of pixels for each intensity value to get the probability density
    red_density = [count / total_pixels for count in red_count]
    green_density = [count / total_pixels for count in green_count]
    blue_density = [count / total_pixels for count in blue_count]

    # Plot the RGB histograms
    plt.figure(figsize=(10, 5))
    plt.plot(red_density, color='red', label='Red Channel')
    plt.plot(green_density, color='green', label='Green Channel')
    plt.plot(blue_density, color='blue', label='Blue Channel')
    plt.xlabel('Intensity Value')
    plt.ylabel('Pixel Count')
    plt.legend()
    plt.show()


def main():
    # get the color.jpg image
    img = io.imread("1 - Kleuren, Histogrammen en Features/color.jpg")
    edited_img = img.copy()

    # Define the color range to preserve (lower and upper bounds for each channel)
    lower = np.array([222, 0, 0])
    upper = np.array([255, 70, 15])
    edited_img = colorFilter(edited_img, lower, upper)

    viewer = ImageViewer(img)
    viewer.show()
    hue_histogram(img)
    viewer = ImageViewer(edited_img)
    viewer.show()
    hue_histogram(edited_img)


if __name__ == '__main__':
    main()
