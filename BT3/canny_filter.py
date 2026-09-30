"""Bai 1b - Canny tu cai dat tren ma tran 10x10 va doi chieu cv2.Canny.

Khong dung cv2.Canny trong phan tu cai dat. cv2.Canny chi duoc dung o phan doi chieu.
Chay: py canny_filter.py
"""
from pathlib import Path

import cv2
import numpy as np


I = np.array([
    [0, 0, 0, 0, 200, 200, 0, 0, 0, 0],
    [0, 0, 0, 0, 200, 200, 0, 0, 0, 0],
    [0, 0, 0, 200, 200, 200, 200, 0, 0, 0],
    [0, 0, 200, 200, 200, 200, 200, 200, 0, 0],
    [200, 200, 200, 200, 200, 200, 200, 200, 200, 200],
    [200, 200, 200, 200, 200, 200, 200, 200, 200, 200],
    [0, 0, 200, 200, 200, 200, 200, 200, 0, 0],
    [0, 0, 0, 200, 200, 200, 200, 0, 0, 0],
    [0, 0, 0, 0, 200, 200, 0, 0, 0, 0],
    [0, 0, 0, 0, 200, 200, 0, 0, 0, 0],
], dtype=np.float64)

KX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
KY = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


def labeled_panels(images: list[np.ndarray], labels: list[str]) -> np.ndarray:
    """Them nhan tren tung o anh de phan biet ro khi chen vao bao cao."""
    panels = []
    for image, label in zip(images, labels):
        height, width = image.shape
        panel = np.full((height + 35, width), 255, dtype=np.uint8)
        panel[35:, :] = image
        cv2.putText(panel, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, 0, 2, cv2.LINE_AA)
        panels.append(panel)
    return np.hstack(panels)


def convolution_3x3(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Tu tich chap 3x3, zero-padding o bien."""
    height, width = image.shape
    output = np.zeros((height, width), dtype=np.float64)
    for y in range(height):
        for x in range(width):
            for ky in range(3):
                for kx in range(3):
                    yy, xx = y + ky - 1, x + kx - 1
                    if 0 <= yy < height and 0 <= xx < width:
                        output[y, x] += image[yy, xx] * kernel[ky, kx]
    return output


def non_maximum_suppression(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Chi giu diem cuc dai theo huong gradient (4 huong 0/45/90/135)."""
    height, width = magnitude.shape
    result = np.zeros_like(magnitude)
    direction = (np.rad2deg(angle) + 180) % 180
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            a = direction[y, x]
            if a < 22.5 or a >= 157.5:
                before, after = magnitude[y, x - 1], magnitude[y, x + 1]
            elif a < 67.5:
                before, after = magnitude[y - 1, x + 1], magnitude[y + 1, x - 1]
            elif a < 112.5:
                before, after = magnitude[y - 1, x], magnitude[y + 1, x]
            else:
                before, after = magnitude[y - 1, x - 1], magnitude[y + 1, x + 1]
            if magnitude[y, x] >= before and magnitude[y, x] >= after:
                result[y, x] = magnitude[y, x]
    return result


def hysteresis(thinned: np.ndarray, low: float, high: float) -> np.ndarray:
    """Giu bien manh va bien yeu lien thong 8 huong voi bien manh."""
    height, width = thinned.shape
    output = np.zeros((height, width), dtype=np.uint8)
    strong = [(y, x) for y in range(height) for x in range(width) if thinned[y, x] >= high]
    stack = strong[:]
    for y, x in strong:
        output[y, x] = 255
    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                yy, xx = y + dy, x + dx
                if (0 <= yy < height and 0 <= xx < width and output[yy, xx] == 0
                        and low <= thinned[yy, xx] < high):
                    output[yy, xx] = 255
                    stack.append((yy, xx))
    return output


def canny_manual(image: np.ndarray, low: float = 200, high: float = 500) -> np.ndarray:
    """Canny: Sobel -> magnitude/huong -> NMS -> nguong kep -> hysteresis."""
    gx = convolution_3x3(image.astype(np.float64), KX)
    gy = convolution_3x3(image.astype(np.float64), KY)
    magnitude = np.hypot(gx, gy)
    thinned = non_maximum_suppression(magnitude, np.arctan2(gy, gx))
    return hysteresis(thinned, low, high)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    low, high = 200, 500
    manual = canny_manual(I, low, high)
    opencv = cv2.Canny(I.astype(np.uint8), low, high, apertureSize=3, L2gradient=True)
    difference = cv2.absdiff(manual, opencv)

    # Phong to de quan sat ma tran 10x10; 4 o: I | tu cai dat | OpenCV | sai khac.
    panels = [I.astype(np.uint8), manual, opencv, difference]
    enlarged = [cv2.resize(panel, (500, 500), interpolation=cv2.INTER_NEAREST) for panel in panels]
    # Anh dung cho muc 1b: chi ma tran goc va ket qua Canny tu cai dat.
    manual_only = labeled_panels(
        enlarged[:2], ["Ma tran goc I", "Canny tu cai dat"]
    )
    cv2.imwrite(str(root / "canny_manual_matrix.png"), manual_only)

    # Anh dung cho muc 1c: doi chieu voi OpenCV.
    comparison = labeled_panels(
        enlarged, ["Ma tran goc I", "Canny tu cai dat", "Canny OpenCV", "Sai khac"]
    )
    cv2.imwrite(str(root / "canny_matrix_comparison.png"), comparison)
    np.savetxt(root / "canny_values.txt", manual, fmt="%3d")
    print("Ma tran Canny tu cai dat (0/255):\n", manual)
    print("Da luu canny_manual_matrix.png, canny_matrix_comparison.png va canny_values.txt")
    print("So pixel khac cv2.Canny:", np.count_nonzero(difference))

    # Tach bien tren anh dua vao bai (goc | tu cai dat | OpenCV | sai khac).
    original = cv2.imread(str(root / "nhieuhat.jpg"), cv2.IMREAD_GRAYSCALE)
    if original is not None:
        manual_image = canny_manual(original, 100, 200)
        opencv_image = cv2.Canny(original, 100, 200, apertureSize=3, L2gradient=True)
        image_difference = cv2.absdiff(manual_image, opencv_image)
        comparison = labeled_panels(
            [original, manual_image, opencv_image, image_difference],
            ["Anh goc", "Canny tu cai dat", "Canny OpenCV", "Sai khac"],
        )
        cv2.imwrite(str(root / "canny_opencv_comparison.png"), comparison)
        print("Da luu canny_opencv_comparison.png")
