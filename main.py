import os

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_image(image_path: str) -> np.ndarray:
    return np.array(Image.open(image_path))


def create_histogram(
        image: np.ndarray,
        bins=256,
        filename: str = "histogram.png",
        cumulative: bool = False,
) -> None:

    plt.figure(figsize=(16, 6))
    plt.hist(
        image.ravel(),
        bins=bins,
        range=(0, 255),
        color='gray',
        alpha=0.7,
        density=True,
        cumulative=cumulative,
    )

    title = 'Cumulative ' if cumulative else ''
    plt.title(f'{title}Grayscale Image Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Density')

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
        print(50*"-", f"\nHistogram saved to {filename}")

    plt.show()


def compute_cdf(
        image: np.ndarray,
        is_normalized: bool = False,
) -> np.ndarray :

    hist, bins = np.histogram(
        image.ravel(),
        bins=256,
        range=(0, 255)
    )
    cdf = hist.cumsum()

    return cdf if not is_normalized else ((cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())).astype(np.uint8)


def apply_histogram_equalization(
        image_array: np.ndarray,
        image_cdf: np.ndarray,
) -> np.ndarray:
    return image_cdf[image_array]


def save_image(
        image_array: np.ndarray,
        filename: str
) -> None:
    Image.fromarray(image_array).save(filename)
    return print(50*"-",f"\nImage saved to {filename}")


def main():
    base_image_path = "Images"
    if not os.path.exists(base_image_path):
        os.mkdir(base_image_path)

    # Load the image
    image_path = os.path.join(base_image_path, "Low-Contrast.tif")
    image_array = load_image(image_path)

    # Print image shape
    print("Image Shape:", image_array.shape)

    # Create and display histogram
    create_histogram(
        image=image_array,
        bins=256,
        filename=os.path.join(base_image_path, "low_contrast_histogram.png")
    )

    create_histogram(
        image=image_array,
        bins=256,
        filename=os.path.join(base_image_path, "low_contrast_cumulative_histogram.png"),
        cumulative=True
    )

    normalized_cdf = compute_cdf(
        image=image_array,
        is_normalized=True,
    )
    print("Normalized CDF:", normalized_cdf)

    equalized_image_array = apply_histogram_equalization(
        image_array=image_array,
        image_cdf=normalized_cdf,
    )
    print("Equalized Image:", equalized_image_array)

    create_histogram(
        image=equalized_image_array,
        bins=256,
        filename=os.path.join(base_image_path, "equalized_histogram.png")
    )
    create_histogram(
        image=equalized_image_array,
        bins=256,
        filename=os.path.join(base_image_path, "equalized_cumulative_histogram.png"),
        cumulative=True
    )
    save_image(
        image_array=equalized_image_array,
        filename=os.path.join(base_image_path, "equalized_image.png")
    )

if __name__ == '__main__':
    main()