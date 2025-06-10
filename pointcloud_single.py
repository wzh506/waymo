import os
import tensorflow.compat.v1 as tf
import numpy as np
from mayavi import mlab
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import frame_utils

# 配置GPU显存自适应分配
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
# training_segment-11392401368700458296_1086_429_1106_429_with_camera_labels_150913982702468600_append1.jpg
def load_and_visualize_first_frame():
    # 文件路径
    FILENAME = '/media/wzh/datasets/openlane/waymo/segment-11392401368700458296_1086_429_1106_429_with_camera_labels.tfrecord'
    
    # 创建Mayavi画布
    fig = mlab.figure(size=(1920, 1080), bgcolor=(0.1, 0.1, 0.1))
    
    # 读取第一个帧
    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    first_data = next(iter(dataset.take(1)))
    
    # 解析帧数据
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(first_data.numpy()))
    
    # 提取点云数据
    (range_images, camera_projections, range_image_top_pose) = \
        frame_utils.parse_range_image_and_camera_projection(frame)
    
    points, _ = frame_utils.convert_range_image_to_point_cloud(
        frame,
        range_images,
        camera_projections,
        range_image_top_pose)
    
    # 合并所有雷达点云
    all_points = np.concatenate(points, axis=0)
    
    # 提取坐标并可视化
    x = all_points[:, 0]
    y = all_points[:, 1]
    z = all_points[:, 2]
    
    # 创建点云可视化
    pts = mlab.points3d(
        x, y, z, z,  # 使用z值作为颜色映射
        mode="point",
        colormap="spectral",  # 使用光谱色图
        scale_factor=0.05,     # 点大小
        figure=fig
    )
    
    # 调整视角
    mlab.view(azimuth=45, elevation=60, distance=50)
    
    # 添加颜色条
    mlab.colorbar(pts, title='Height (m)', orientation='vertical')
    
    # 显示
    mlab.show()

if __name__ == "__main__":
    load_and_visualize_first_frame()