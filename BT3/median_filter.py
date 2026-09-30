"""Bai 1a - Bo loc Median tu cai dat va doi chieu cv2.medianBlur."""
from pathlib import Path

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


def median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Sap xep cua so va lay phan tu giua; khong dung ham median co san."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Kich thuoc kernel phai la so le duong.")

    radius = kernel_size // 2
    height, width = image.shape
    result = np.zeros_like(image)
    middle = (kernel_size * kernel_size) // 2
    for y in range(height):
        for x in range(width):
            values = []
            for ky in range(-radius, radius + 1):
                for kx in range(-radius, radius + 1):
                    yy = min(max(y + ky, 0), height - 1)
                    xx = min(max(x + kx, 0), width - 1)
                    values.append(int(image[yy, xx]))
            values.sort()
            result[y, x] = values[middle]
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    original = cv2.imread(str(root / "nhieumuoitieu.jpg"), cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError("Khong doc duoc nhieumuoitieu.jpg")

    manual = median_filter(original, 3)
    opencv = cv2.medianBlur(original, 3)
    difference = cv2.absdiff(manual, opencv)
    comparison = labeled_panels(
        [original, manual, opencv, difference],
        ["Anh goc", "Tu cai dat", "OpenCV", "Sai khac"],
    )
    cv2.imwrite(str(root / "median_comparison.png"), comparison)
    print("Da luu median_comparison.png")
    print("Sai khac lon nhat (manual - OpenCV):", difference.max())
