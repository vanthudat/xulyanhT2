"""Bai 1a - Bo loc Mean tu cai dat va doi chieu cv2.blur.

Chay: py mean_filter.py
Anh ket qua: mean_comparison.png (goc | tu cai dat | OpenCV | sai khac)
"""
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


def mean_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Loc trung binh cong tu cai dat, xu ly bien bang replicate."""
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Kich thuoc kernel phai la so le duong.")

    radius = kernel_size // 2
    height, width = image.shape
    result = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            total = 0
            for ky in range(-radius, radius + 1):
                for kx in range(-radius, radius + 1):
                    yy = min(max(y + ky, 0), height - 1)
                    xx = min(max(x + kx, 0), width - 1)
                    total += int(image[yy, xx])
            result[y, x] = round(total / (kernel_size * kernel_size))
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    original = cv2.imread(str(root / "nhieuhat.jpg"), cv2.IMREAD_GRAYSCALE)
    if original is None:
        raise FileNotFoundError("Khong doc duoc nhieuhat.jpg")

    manual = mean_filter(original, 3)
    opencv = cv2.blur(original, (3, 3), borderType=cv2.BORDER_REPLICATE)
    difference = cv2.absdiff(manual, opencv)
    comparison = labeled_panels(
        [original, manual, opencv, difference],
        ["Anh goc", "Tu cai dat", "OpenCV", "Sai khac"],
    )
    cv2.imwrite(str(root / "mean_comparison.png"), comparison)
    print("Da luu mean_comparison.png")
    print("Sai khac lon nhat (manual - OpenCV):", difference.max())
