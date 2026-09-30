import cv2
import numpy as np


# Ma tran 10x10 doc tu anh bai tap.
I = np.array(
    [
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
    ],
    dtype=np.int32,
)

KX = np.array(
    [
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ],
    dtype=np.int32,
)

KY = np.array(
    [
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1],
    ],
    dtype=np.int32,
)


def tich_chap_3x3(anh, kernel):
    """Tinh tich chap 3x3 thu cong, bien ngoai anh duoc xem bang 0."""
    chieu_cao, chieu_rong = anh.shape
    ket_qua = np.zeros((chieu_cao, chieu_rong), dtype=np.int32)

    for y in range(chieu_cao):
        for x in range(chieu_rong):
            tong = 0

            for ky in range(3):
                for kx in range(3):
                    yy = y + ky - 1
                    xx = x + kx - 1

                    # Zero-padding: pixel ngoai ma tran co gia tri 0.
                    if 0 <= yy < chieu_cao and 0 <= xx < chieu_rong:
                        tong += int(anh[yy, xx]) * int(kernel[ky, kx])

            ket_qua[y, x] = tong

    return ket_qua


def sobel_thu_cong(anh):
    """Tra ve Gx, Gy va do lon gradient; khong dung cv2.Sobel."""
    gx = tich_chap_3x3(anh, KX)
    gy = tich_chap_3x3(anh, KY)
    do_lon = np.sqrt(gx.astype(np.float64) ** 2 + gy.astype(np.float64) ** 2)
    return gx, gy, np.rint(do_lon).astype(np.int32)


def chuan_hoa_anh(ma_tran):
    """Chuan hoa ma tran khong am ve anh 8-bit trong khoang [0, 255]."""
    gia_tri_lon_nhat = int(ma_tran.max())
    if gia_tri_lon_nhat == 0:
        return np.zeros_like(ma_tran, dtype=np.uint8)

    return np.rint(ma_tran * 255.0 / gia_tri_lon_nhat).astype(np.uint8)


def ghep_anh_co_nhan(cac_anh, cac_nhan):
    """Them nhan tren tung o anh de phan biet ket qua trong bao cao."""
    panels = []
    for anh, nhan in zip(cac_anh, cac_nhan):
        chieu_cao, chieu_rong = anh.shape
        panel = np.full((chieu_cao + 35, chieu_rong), 255, dtype=np.uint8)
        panel[35:, :] = anh
        cv2.putText(panel, nhan, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, 0, 2, cv2.LINE_AA)
        panels.append(panel)
    return np.hstack(panels)


if __name__ == "__main__":
    gx, gy, do_lon = sobel_thu_cong(I)

    print("Ma tran Gx:")
    print(gx)
    print("\nMa tran Gy:")
    print(gy)
    print("\nDo lon gradient |G|:")
    print(do_lon)
    print("\nGia tri |G| lon nhat:", do_lon.max())

    # Tao anh bien Sobel: chuan hoa |G| ve [0, 255].
    anh_bien = chuan_hoa_anh(do_lon)

    # Phong to moi phan tu 80 lan de nhin ro ma tran 10x10.
    anh_goc_lon = cv2.resize(I.astype(np.uint8), (800, 800), interpolation=cv2.INTER_NEAREST)
    anh_bien_lon = cv2.resize(anh_bien, (800, 800), interpolation=cv2.INTER_NEAREST)
    anh_so_sanh = ghep_anh_co_nhan(
        [anh_goc_lon, anh_bien_lon], ["Ma tran goc I", "Sobel tu cai dat"]
    )

    cv2.imwrite("sobel_edge_map_python.png", anh_bien_lon)
    cv2.imwrite("sobel_comparison_python.png", anh_so_sanh)

    # Luu cac ma tran so de chen vao bao cao neu can.
    with open("sobel_values_python.txt", "w", encoding="utf-8") as tep:
        tep.write("Gx:\n" + np.array2string(gx) + "\n\n")
        tep.write("Gy:\n" + np.array2string(gy) + "\n\n")
        tep.write("|G| = round(sqrt(Gx^2 + Gy^2)):\n")
        tep.write(np.array2string(do_lon) + "\n")

    print("\nDa luu sobel_edge_map_python.png")
    print("Da luu sobel_comparison_python.png")

    # Doi chieu voi cv2.Sobel tren cung ma tran, cung quy uoc zero-padding.
    cv_gx = cv2.Sobel(I.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3,
                      borderType=cv2.BORDER_CONSTANT).astype(np.int32)
    cv_gy = cv2.Sobel(I.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3,
                      borderType=cv2.BORDER_CONSTANT).astype(np.int32)
    sai_khac = np.abs(gx - cv_gx) + np.abs(gy - cv_gy)
    print("Sai khac Gx/Gy voi cv2.Sobel:", sai_khac.max())

    # So sanh tren anh nhieu: goc | tu cai dat | OpenCV | anh sai khac.
    anh_nhieu = cv2.imread("nhieuhat.jpg", cv2.IMREAD_GRAYSCALE)
    if anh_nhieu is not None:
        _, _, bien_thu_cong = sobel_thu_cong(anh_nhieu.astype(np.int32))
        cv_x = cv2.Sobel(anh_nhieu, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_CONSTANT)
        cv_y = cv2.Sobel(anh_nhieu, cv2.CV_64F, 0, 1, ksize=3, borderType=cv2.BORDER_CONSTANT)
        bien_opencv = np.rint(np.sqrt(cv_x * cv_x + cv_y * cv_y)).astype(np.int32)
        manual_8bit = chuan_hoa_anh(bien_thu_cong)
        opencv_8bit = chuan_hoa_anh(bien_opencv)
        comparison = ghep_anh_co_nhan(
            [anh_nhieu, manual_8bit, opencv_8bit, cv2.absdiff(manual_8bit, opencv_8bit)],
            ["Anh goc", "Sobel tu cai dat", "Sobel OpenCV", "Sai khac"],
        )
        cv2.imwrite("sobel_opencv_comparison.png", comparison)
        print("Da luu sobel_opencv_comparison.png")
