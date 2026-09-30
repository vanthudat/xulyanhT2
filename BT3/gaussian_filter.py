"""Bai 1a - Bo loc Gaussian tu cai dat va doi chieu cv2.GaussianBlur."""
from pathlib import Path
import math

import cv2
import numpy as np


def labeled_panels(images: list[np.ndarray], labels: list[str]) -> np.ndarray:
    """Them nhan phia tren tung anh de dung truc tiep trong bao cao."""
    panels = []
    for image, label in zip(images, labels):
        height, width = image.shape
        panel = np.full((height + 35, width), 255, dtype=np.uint8)
        panel[35:, :] = image
        cv2.putText(panel, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, 0, 2, cv2.LINE_AA)
        panels.append(panel)
    return np.hstack(panels)


def gaussian_kernel(kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    if kernel_size <= 0 or kernel_size % 2 == 0 or sigma <= 0:
        raise ValueError("kernel_size le duong va sigma duong.")
    radius = kernel_size // 2
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float64)
    for y in range(-radius, radius + 1):
        for x in range(-radius, radius + 1):
            kernel[y + radius, x + radius] = math.exp(-(x*x + y*y) / (2*sigma*sigma))
    return kernel / kernel.sum()


def gaussian_filter(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    kernel = gaussian_kernel(kernel_size, sigma)
    radius = kernel_size // 2
    height, width = image.shape
    result = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            value = 0.0
            for ky in range(-radius, radius + 1):
                for kx in range(-radius, radius + 1):
                    yy = min(max(y + ky, 0), height - 1)
                    xx = min(max(x + kx, 0), width - 1)
                    value += int(image[yy, xx]) * kernel[ky + radius, kx + radius]
            result[y, x] = np.uint8(np.clip(round(value), 0, 255))
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    original = cv2.imread(str(root / "nhieuhat.jpg"), cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError("Khong doc duoc nhieuhat.jpg")

    manual = gaussian_filter(original, 5, 1.0)
    opencv = cv2.GaussianBlur(original, (5, 5), 1.0, borderType=cv2.BORDER_REPLICATE)
    difference = cv2.absdiff(manual, opencv)
    comparison = labeled_panels(
        [original, manual, opencv, difference],
        ["Anh goc", "Tu cai dat", "OpenCV", "Sai khac"],
    )
    cv2.imwrite(str(root / "gaussian_comparison.png"), comparison)
    print("Da luu gaussian_comparison.png")
    print("Sai khac lon nhat (manual - OpenCV):", difference.max())
