import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage import data


def main():
    fig, ((img1, img2, img3, img4)) = plt.subplots(1, 4)
    image = data.astronaut()

    img1.imshow(image)

    # draaien van de image
    rotation_matrix = skimage.transform.AffineTransform(
        rotation=np.deg2rad(30))
    rotated_image = skimage.transform.warp(image, rotation_matrix.params)
    img2.imshow(rotated_image)

    # Translatie
    translation_matrix = skimage.transform.AffineTransform(
        translation=(50, -30))
    translated_image = skimage.transform.warp(image, translation_matrix.params)
    img3.imshow(translated_image)

    # Stretching
    stretch_matrix = skimage.transform.AffineTransform(scale=(1.2, 0.8))
    stretched_image = skimage.transform.warp(image, stretch_matrix.params)
    img4.imshow(stretched_image)

    plt.show()


if __name__ == '__main__':
    main()
