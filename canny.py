import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image_path = '1.png'
image = cv2.imread(image_path)

# 检查图像是否被成功读取
if image is None:
    print(f"Error: Unable to read the image at {image_path}")
else:
    # 定义蓝色的范围
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    # 转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 提取蓝色部分
    mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
    blue_extracted = cv2.bitwise_and(image, image, mask=mask)

    # 转换为灰度图以进行边缘检测
    blue_gray = cv2.cvtColor(blue_extracted, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测
    edges = cv2.Canny(blue_gray, 50, 150)

    # 膨胀操作使线条更粗
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=3)

    # 创建一个全黑的图像用于绘制增强后的蓝色曲线
    enhanced_curve = np.zeros_like(image)
    enhanced_curve[dilated_edges != 0] = [255, 0, 0]  # 用蓝色绘制增强后的曲线

    # 将增强后的蓝色曲线叠加回原图像
    combined_image = cv2.addWeighted(image, 1, enhanced_curve, 1, 0)

    # 显示结果
    plt.figure(figsize=(120, 80))

    # # 原始图像
    # plt.subplot(1, 3, 1)
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.title('Original Image')

    # # 提取的蓝色部分
    # plt.subplot(1, 3, 2)
    # plt.imshow(cv2.cvtColor(blue_extracted, cv2.COLOR_BGR2RGB))
    # plt.title('Blue Extracted')

    # 增强后的曲线叠加回原图像
    # plt.subplot(1, 3, 3)
    plt.xticks([]), plt.yticks([])
    plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    plt.title('Enhanced Curve Combined')
    plt.savefig('enhanced_curve_1.png')
    # plt.show()
