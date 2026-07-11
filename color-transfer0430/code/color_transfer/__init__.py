# # RGB格式

# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# def color_transfer(source, target, clip=True, preserve_paper=True, debug=False):
#     # 生成目标图像的白色区域掩膜（基于BGR颜色空间）
#     lower_white = np.array([220, 220, 220], dtype=np.uint8)
#     upper_white = np.array([255, 255, 255], dtype=np.uint8)
#     white_mask = cv2.inRange(target, lower_white, upper_white)
#
#     if debug:
#         cv2.imshow("[DEBUG 1] Target White Mask", white_mask)
#         cv2.waitKey(0)
#
#     # 获取非白色区域的统计信息（传入BGR图像）
#     src_stats = non_white_region_stats(source, debug=debug)
#     tar_stats = non_white_region_stats(target, debug=debug)
#
#     (bMeanSrc, gMeanSrc, rMeanSrc, bStdSrc, gStdSrc, rStdSrc) = src_stats
#     (bMeanTar, gMeanTar, rMeanTar, bStdTar, gStdTar, rStdTar) = tar_stats
#     print(bMeanSrc, gMeanSrc, rMeanSrc, bStdSrc, gStdSrc, rStdSrc)
#     print(bMeanTar, gMeanTar, rMeanTar, bStdTar, gStdTar, rStdTar)
#
#     # 拆分通道并进行颜色转移
#     b, g, r = cv2.split(target.astype("float32"))
#     b -= bMeanTar
#     g -= gMeanTar
#     r -= rMeanTar
#
#     if preserve_paper:
#         b = (bStdTar / (bStdSrc + 1e-8)) * b  # 避免除以零
#         g = (gStdTar / (gStdSrc + 1e-8)) * g
#         r = (rStdTar / (rStdSrc + 1e-8)) * r
#     else:
#         b = (bStdSrc / (bStdTar + 1e-8)) * b
#         g = (gStdSrc / (gStdTar + 1e-8)) * g
#         r = (rStdSrc / (rStdTar + 1e-8)) * r
#
#     b += bMeanSrc
#     g += gMeanSrc
#     r += rMeanSrc
#
#     # 调整并限制范围
#     b = _scale_array(b, clip=clip)
#     g = _scale_array(g, clip=clip)
#     r = _scale_array(r, clip=clip)
#
#     # 合并通道并恢复白色区域
#     transfer_bgr = cv2.merge([b, g, r]).astype("uint8")
#
#     # 绘制颜色迁移后的非白色部分的RGB三通道直方图
#     non_white_mask = cv2.bitwise_not(white_mask)
#     b_hist = cv2.calcHist([b.astype("uint8")], [0], non_white_mask, [256], [0, 256])
#     g_hist = cv2.calcHist([g.astype("uint8")], [0], non_white_mask, [256], [0, 256])
#     r_hist = cv2.calcHist([r.astype("uint8")], [0], non_white_mask, [256], [0, 256])
#
#     plt.figure()
#     plt.title("Non-White Region Histogram After Color Transfer")
#     plt.xlabel("Pixel Value")
#     plt.ylabel("Frequency")
#     plt.plot(b_hist, color="b", label="Blue")
#     plt.plot(g_hist, color="g", label="Green")
#     plt.plot(r_hist, color="r", label="Red")
#     plt.legend()
#     plt.xlim([0, 256])
#     plt.show()
#
#     if debug:
#         # 显示颜色转移后的BGR图像（恢复白色区域前）
#         cv2.imshow("[DEBUG 3] Before White Recovery", transfer_bgr)
#         cv2.waitKey(0)
#
#     # 恢复白色区域的原始BGR值
#     transfer_bgr[white_mask != 0] = target[white_mask != 0]
#
#     if debug:
#         cv2.imshow("[DEBUG 4] Final Result", transfer_bgr)
#         cv2.waitKey(0)
#
#     return transfer_bgr
#
#
# def non_white_region_stats(image, debug=False):
#     # 创建白色背景掩膜（基于BGR图像）
#     lower_white = np.array([220, 220, 220], dtype=np.uint8)
#     upper_white = np.array([255, 255, 255], dtype=np.uint8)
#     white_mask = cv2.inRange(image, lower_white, upper_white)
#
#     if debug:
#         cv2.imshow("[DEBUG 2] White Mask in Stats", white_mask)
#         cv2.waitKey(0)
#
#     # 生成非白色区域的掩膜
#     non_white_mask = cv2.bitwise_not(white_mask)
#
#     # 找到非白色区域的边界坐标
#     coords = np.argwhere(non_white_mask > 0)
#     if len(coords) == 0:
#         return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), ([], [], [])
#
#     y1, x1 = np.min(coords, axis=0)
#     y2, x2 = np.max(coords, axis=0)
#
#     # 扩展边界
#     expand_pixels = 10
#     x1_exp = max(0, x1 - expand_pixels)
#     y1_exp = max(0, y1 - expand_pixels)
#     x2_exp = min(image.shape[1], x2 + expand_pixels)
#     y2_exp = min(image.shape[0], y2 + expand_pixels)
#
#     if debug:
#         # 在原图上绘制扩展区域框
#         debug_image = image.copy()
#         cv2.rectangle(debug_image, (x1_exp, y1_exp), (x2_exp, y2_exp), (0, 255, 0), 2)
#         cv2.imshow("[DEBUG 2] Expanded Region", debug_image)
#         cv2.waitKey(0)
#
#     region_bgr = cv2.bitwise_and(image, image, mask=non_white_mask)
#
#     if debug:
#         cv2.imshow("[DEBUG 2] Non-White Region", region_bgr)
#         cv2.waitKey(0)
#
#     # 分离通道并计算统计信息
#     b, g, r = cv2.split(region_bgr.astype("float32"))
#
#     # 只计算非白色区域的统计信息
#     b = b[non_white_mask > 0]
#     g = g[non_white_mask > 0]
#     r = r[non_white_mask > 0]
#
#     # 计算直方图
#     b_hist = cv2.calcHist([b.astype("uint8")], [0], None, [256], [0, 256])
#     g_hist = cv2.calcHist([g.astype("uint8")], [0], None, [256], [0, 256])
#     r_hist = cv2.calcHist([r.astype("uint8")], [0], None, [256], [0, 256])
#
#     # 绘制直方图
#     plt.figure()
#     plt.title("Non-White Region Histogram")
#     plt.xlabel("Pixel Value")
#     plt.ylabel("Frequency")
#     plt.plot(b_hist, color="b", label="Blue")
#     plt.plot(g_hist, color="g", label="Green")
#     plt.plot(r_hist, color="r", label="Red")
#     plt.legend()
#     plt.xlim([0, 256])
#     plt.show()
#
#     return (b.mean(), g.mean(), r.mean(), b.std(), g.std(), r.std())
#
#
# def _scale_array(arr, clip=True):
#     if clip:
#         return np.clip(arr, 0, 255)
#     else:
#         scale_range = (max(arr.min(), 0), min(arr.max(), 255))
#         return _min_max_scale(arr, scale_range)
#
#
# def _min_max_scale(arr, new_range=(0, 255)):
#     mn = arr.min()
#     mx = arr.max()
#     return (new_range[1] - new_range[0]) * (arr - mn) / (mx - mn + 1e-8) + new_range[0]


# LAB格式

import numpy as np
import cv2
# import matplotlib.pyplot as plt
def color_transfer(source, ssmask, target, tsmask, clip=True, preserve_paper=True, debug=False):

    if debug:
        cv2.imshow("[DEBUG 1] Target White Mask", ssmask)
        cv2.waitKey(0)

    # 转换到LAB颜色空间
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")
    target_lab_original = target_lab.copy()  # 保存原始LAB值用于恢复
    # 获取非白色区域的统计信息（传入BGR图像）
    src_stats = skin_region_stats(source_lab, ssmask, debug=debug)
    tar_stats = skin_region_stats(target_lab, tsmask, debug=debug)

    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = src_stats
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = tar_stats
    print(lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc)
    print(lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar)

    
    # 拆分通道并进行颜色转移
    (l, a, b) = cv2.split(target_lab)
    if debug:
        show_hist(l, a, b, tsmask, "before")

    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    if preserve_paper:
        l = (lStdTar / (lStdSrc + 1e-8)) * l  # 避免除以零
        a = (aStdTar / (aStdSrc + 1e-8)) * a
        b = (bStdTar / (bStdSrc + 1e-8)) * b
    else:
        l = (lStdSrc / (lStdTar + 1e-8)) * l
        a = (aStdSrc / (aStdTar + 1e-8)) * a
        b = (bStdSrc / (bStdTar + 1e-8)) * b

    l += lMeanSrc 
    a += aMeanSrc
    b += bMeanSrc

    # 调整并限制范围
    l = _scale_array(l, clip=clip)
    a = _scale_array(a, clip=clip)
    b = _scale_array(b, clip=clip)

    # 合并通道并恢复白色区域
    transfer_lab = cv2.merge([l, a, b])
    
    # 绘制颜色迁移后的非白色部分的RGB三通道直方图
    if debug:
        show_hist(l, a, b, tsmask, "after")

    if debug:
        # 显示颜色转移后的LAB图像（恢复白色区域前）
        debug_transfer = cv2.cvtColor(transfer_lab.astype("uint8"), cv2.COLOR_LAB2BGR)
        cv2.imshow("[DEBUG 3] Before White Recovery", debug_transfer)
        cv2.waitKey(0)

    # 转换回BGR颜色空间
    transfer = cv2.cvtColor(transfer_lab.astype("uint8"), cv2.COLOR_LAB2BGR)

    if debug:
        cv2.imshow("[DEBUG 4] Final Result", transfer)
        cv2.waitKey(0)

    return transfer

def skin_region_stats(image, mask, debug=False):

    if debug:
        cv2.imshow("[DEBUG 2] Mask in Stats", mask)
        cv2.waitKey(0)

    # 分离通道并计算统计信息
    l, a, b = cv2.split(image.astype("float32"))

    # 只计算非白色区域的统计信息
    l = l[mask > 0]
    a = a[mask > 0]
    b = b[mask > 0]

    return (l.mean(), l.std(), a.mean(), a.std(), b.mean(), b.std())

# def non_white_region_stats(image, debug=False):
#     # 创建白色背景掩膜（基于BGR图像）
#     lower_white = np.array([220, 220, 220], dtype=np.uint8)
#     upper_white = np.array([255, 255, 255], dtype=np.uint8)
#     white_mask = cv2.inRange(image, lower_white, upper_white)

#     if debug:
#         cv2.imshow("[DEBUG 2] White Mask in Stats", white_mask)
#         cv2.waitKey(0)

#     # 生成非白色区域的掩膜
#     non_white_mask = cv2.bitwise_not(white_mask)

#     # 找到非白色区域的边界坐标
#     coords = np.argwhere(non_white_mask > 0)
#     if len(coords) == 0:
#         return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), ([], [], [])

#     y1, x1 = np.min(coords, axis=0)
#     y2, x2 = np.max(coords, axis=0)

#     # 扩展边界
#     expand_pixels = 10
#     x1_exp = max(0, x1 - expand_pixels)
#     y1_exp = max(0, y1 - expand_pixels)
#     x2_exp = min(image.shape[1], x2 + expand_pixels)
#     y2_exp = min(image.shape[0], y2 + expand_pixels)

#     if debug:
#         # 在原图上绘制扩展区域框
#         debug_image = image.copy()
#         cv2.rectangle(debug_image, (x1_exp, y1_exp), (x2_exp, y2_exp), (0, 255, 0), 2)
#         cv2.imshow("[DEBUG 2] Expanded Region", debug_image)
#         cv2.waitKey(0)

#     region_bgr = cv2.bitwise_and(image, image, mask=non_white_mask)

#     if debug:
#         cv2.imshow("[DEBUG 2] Non-White Region", region_bgr)
#         cv2.waitKey(0)

#     region_lab = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2LAB)
#     # 分离通道并计算统计信息
#     l, a, b = cv2.split(region_lab.astype("float32"))

#     # 只计算非白色区域的统计信息
#     l = l[non_white_mask > 0]
#     a = a[non_white_mask > 0]
#     b = b[non_white_mask > 0]

#     # 计算直方图
#     l_hist = cv2.calcHist([l.astype("uint8")], [0], None, [256], [0, 256])
#     a_hist = cv2.calcHist([a.astype("uint8")], [0], None, [256], [0, 256])
#     b_hist = cv2.calcHist([b.astype("uint8")], [0], None, [256], [0, 256])

#     # 绘制直方图
#     plt.figure()
#     plt.title("Non-White Region Histogram")
#     plt.xlabel("Pixel Value")
#     plt.ylabel("Frequency")
#     plt.plot(l_hist, color="b", label="l")
#     plt.plot(a_hist, color="g", label="a")
#     plt.plot(b_hist, color="r", label="b")
#     plt.legend()
#     plt.xlim([0, 256])
# #    plt.show()

#     return (l.mean(), l.std(), a.mean(), a.std(), b.mean(), b.std())


def _scale_array(arr, clip=True):
    if clip:
        return np.clip(arr, 0, 255)
    else:
        scale_range = (max(arr.min(), 0), min(arr.max(), 255))
        return _min_max_scale(arr, scale_range)


def _min_max_scale(arr, new_range=(0, 255)):
    mn = arr.min()
    mx = arr.max()
    return (new_range[1] - new_range[0]) * (arr - mn) / (mx - mn + 1e-8) + new_range[0]

def show_hist(x, y, z, mask, title):
    # 绘制颜色迁移后的非白色部分的RGB三通道直方图
    l_hist = cv2.calcHist([x.astype("uint8")], [0], mask, [256], [0, 256])
    a_hist = cv2.calcHist([y.astype("uint8")], [0], mask, [256], [0, 256])
    b_hist = cv2.calcHist([z.astype("uint8")], [0], mask, [256], [0, 256])

    plt.figure()
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Frequency")
    plt.plot(l_hist, color="b", label="l")
    plt.plot(a_hist, color="g", label="a")
    plt.plot(b_hist, color="r", label="b")
    plt.legend()
    plt.xlim([0, 256])
    plt.show()