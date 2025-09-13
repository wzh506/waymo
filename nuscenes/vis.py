import numpy as np
import cv2

file = "/home/zhaohui1.wang/github/datasets/nuscenes/samples/LIDAR_TOP/n015-2018-07-24-11-22-45+0800__LIDAR_TOP__1532402927647951.pcd.bin"

# [x, y, z, intensity, ring index]
pc = np.frombuffer(open(file, "rb").read(), dtype=np.float32)
pc = pc.reshape(-1, 5)[:, :4]

x, y, z, intensity = pc.T

# 设置图像的尺寸1024x1024
image_size = 500

# 数据归一化
# 点的坐标范围大概是100
pc_range = 100
x = x / pc_range    # [-1,1]
y = y / pc_range

# 缩放到图像大小，并平移到图像中心
half_image_size = image_size / 2
x = x * half_image_size + half_image_size
y = y * half_image_size + half_image_size

# opencv的图像，可以用numpy进行创建
image = np.zeros((image_size, image_size, 3), np.uint8)

for ix, iy, iz in zip(x, y, z):
    ix = int(ix)
    iy = int(iy)
    
    # 判断是否在图像范围内
    if ix >= 0 and ix < image_size and iy >= 0 and iy < image_size:
        image[iy, ix] = 255, 255, 255 #有物体就设为？

cv2.imwrite("pointcloud.png", image)
# cv2.imshow("image", image)
# cv2.waitKey(0)
