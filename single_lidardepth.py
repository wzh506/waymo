
import os
import imp
import tensorflow as tf
import math
import numpy as np
import itertools
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# TODO: Change this to your own setting
os.environ['PYTHONPATH']='/env/python:~/github/waymo-open-dataset'
m=imp.find_module('waymo_open_dataset', ['.'])
imp.load_module('waymo_open_dataset', m[0], m[1], m[2])

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
# tf.enable_eager_execution()

import torch
import torch.nn.functional as F
import cv2

from tqdm import tqdm

def image_show(data, name, layout, cmap=None):
  """Show an image."""
  plt.subplot(*layout)
  plt.imshow(tf.image.decode_jpeg(data), cmap=cmap)
  plt.title(name)
  plt.grid(False)
  plt.axis('off')

def parse_range_image_and_camera_projection(frame):
  """Parse range images and camera projections given a frame.

  Args:
     frame: open dataset frame proto
  Returns:
     range_images: A dict of {laser_name,
       [range_image_first_return, range_image_second_return]}.
     camera_projections: A dict of {laser_name,
       [camera_projection_from_first_return,
        camera_projection_from_second_return]}.
    range_image_top_pose: range image pixel pose for top lidar.
  """
  range_images = {}
  camera_projections = {}
  range_image_top_pose = None
  for laser in frame.lasers:
    if len(laser.ri_return1.range_image_compressed) > 0:
      # use tf.io.decode_compressed() if TF 2.0
      range_image_str_tensor = tf.compat.v1.decode_compressed(
          laser.ri_return1.range_image_compressed, 'ZLIB')
      ri = open_dataset.MatrixFloat()
      ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
      range_images[laser.name] = [ri]

      if laser.name == open_dataset.LaserName.TOP:
        # use tf.io.decode_compressed() if TF 2.0
        range_image_top_pose_str_tensor = tf.compat.v1.decode_compressed(
            laser.ri_return1.range_image_pose_compressed, 'ZLIB')
        range_image_top_pose = open_dataset.MatrixFloat()
        range_image_top_pose.ParseFromString(
            bytearray(range_image_top_pose_str_tensor.numpy()))

      # use tf.io.decode_compressed() if TF 2.0
      camera_projection_str_tensor = tf.compat.v1.decode_compressed(
          laser.ri_return1.camera_projection_compressed, 'ZLIB')
      cp = open_dataset.MatrixInt32()
      cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
      camera_projections[laser.name] = [cp]
    if len(laser.ri_return2.range_image_compressed) > 0:
      # use tf.io.decode_compressed() if TF 2.0
      range_image_str_tensor = tf.compat.v1.decode_compressed(
          laser.ri_return2.range_image_compressed, 'ZLIB')
      ri = open_dataset.MatrixFloat()
      ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
      range_images[laser.name].append(ri)

      # use tf.io.decode_compressed() if TF 2.0
      camera_projection_str_tensor = tf.compat.v1.decode_compressed(
          laser.ri_return2.camera_projection_compressed, 'ZLIB')
      cp = open_dataset.MatrixInt32()
      cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
      camera_projections[laser.name].append(cp)
  return range_images, camera_projections, range_image_top_pose 
# read one frame
def plot_range_image_helper(data, name, layout, vmin = 0, vmax=1, cmap='gray'):
  """Plots range image.

  Args:
    data: range image data
    name: the image title
    layout: plt layout
    vmin: minimum value of the passed data
    vmax: maximum value of the passed data
    cmap: color map
  """
  plt.subplot(*layout)
  plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
  plt.title(name)
  plt.grid(False)
  plt.axis('off')

def get_range_image(range_images,laser_name, return_index):
  """Returns range image given a laser name and its return index."""
  return range_images[laser_name][return_index]

def show_range_image(range_image, layout_index_start = 1):
  """Shows range image.

  Args:
    range_image: the range image data from a given lidar of type MatrixFloat.
    layout_index_start: layout offset
  """
  range_image_tensor = tf.convert_to_tensor(range_image.data)
  range_image_tensor = tf.reshape(range_image_tensor, range_image.shape.dims)
  lidar_image_mask = tf.greater_equal(range_image_tensor, 0)
  range_image_tensor = tf.where(lidar_image_mask, range_image_tensor,
                                tf.ones_like(range_image_tensor) * 1e10)
  range_image_range = range_image_tensor[...,0] 
  range_image_intensity = range_image_tensor[...,1]
  range_image_elongation = range_image_tensor[...,2]
  plot_range_image_helper(range_image_range.numpy(), 'range',
                   [8, 1, layout_index_start], vmax=75, cmap='gray')
  plot_range_image_helper(range_image_intensity.numpy(), 'intensity',
                   [8, 1, layout_index_start + 1], vmax=1.5, cmap='gray')
  plot_range_image_helper(range_image_elongation.numpy(), 'elongation',
                   [8, 1, layout_index_start + 2], vmax=1.5, cmap='gray')
  

def convert_range_image_to_point_cloud(frame,
                                       range_images,
                                       camera_projections,
                                       range_image_top_pose,
                                       ri_index = 0):
  """Convert range images to point cloud.

  Args:
    frame: open dataset frame
     range_images: A dict of {laser_name,
       [range_image_first_return, range_image_second_return]}.
     camera_projections: A dict of {laser_name,
       [camera_projection_from_first_return,
        camera_projection_from_second_return]}.
    range_image_top_pose: range image pixel pose for top lidar.
    ri_index: 0 for the first return, 1 for the second return.
  Returns:
    points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
    cp_points: {[N, 6]} list of camera projections of length 5
      (number of lidars).
  """
  calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
  lasers = sorted(frame.lasers, key=lambda laser: laser.name)
  points = [] 
  cp_points = []
  
  frame_pose = tf.convert_to_tensor(
      np.reshape(np.array(frame.pose.transform), [4, 4])) # 4,4 ego pose，传感器坐标系转换到全局坐标系进行时间对齐
  # [H, W, 6]
  range_image_top_pose_tensor = tf.reshape(
      tf.convert_to_tensor(range_image_top_pose.data),
      range_image_top_pose.shape.dims)
  # [H, W, 3, 3]
  range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
      range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
      range_image_top_pose_tensor[..., 2])
  range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
  range_image_top_pose_tensor = transform_utils.get_transform(
      range_image_top_pose_tensor_rotation,
      range_image_top_pose_tensor_translation) #这个应该是雷达的传感器外参
  for c in calibrations:
    range_image = range_images[c.name][ri_index]
    if len(c.beam_inclinations) == 0:
      beam_inclinations = range_image_utils.compute_inclination(
          tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
          height=range_image.shape.dims[0])
    else:
      beam_inclinations = tf.constant(c.beam_inclinations)

    beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
    extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

    range_image_tensor = tf.reshape(
        tf.convert_to_tensor(range_image.data), range_image.shape.dims)
    pixel_pose_local = None
    frame_pose_local = None
    if c.name == open_dataset.LaserName.TOP:
      pixel_pose_local = range_image_top_pose_tensor
      pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
      frame_pose_local = tf.expand_dims(frame_pose, axis=0)
    range_image_mask = range_image_tensor[..., 0] > 0
    range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
        tf.expand_dims(range_image_tensor[..., 0], axis=0),
        tf.expand_dims(extrinsic, axis=0),
        tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
        pixel_pose=pixel_pose_local,
        frame_pose=frame_pose_local)

    range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
    points_tensor = tf.gather_nd(range_image_cartesian,
                                 tf.where(range_image_mask))

    cp = camera_projections[c.name][0]
    cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
    cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))
    points.append(points_tensor.numpy())
    cp_points.append(cp_points_tensor.numpy())

  return points, cp_points


def rgba(r):
  """Generates a color based on range.

  Args:
    r: the range value of a given point.
  Returns:
    The color for a given range
  """
  c = plt.get_cmap('jet')((r % 20.0) / 20.0)
  c = list(c)
  c[-1] = 0.5  # alpha
  return c

def plot_image(camera_image):
  """Plot a cmaera image."""
  plt.figure(figsize=(20, 12))
  plt.imshow(tf.image.decode_jpeg(camera_image.image))
  plt.grid("off")
  # 去除其他的坐标轴
  plt.axis('off')
  plt.gca().set_axis_off()
  plt.subplots_adjust(top=1, bottom=0, right=1, left=0, 
                        hspace=0, wspace=0)
  plt.margins(0, 0)

def plot_points_on_image(projected_points, camera_image, rgba_func,
                         point_size=5.0):
  """Plots points on a camera image.

  Args:
    projected_points: [N, 3] numpy array. The inner dims are
      [camera_x, camera_y, range].
    camera_image: jpeg encoded camera image.
    rgba_func: a function that generates a color from a range value.
    point_size: the point size.

  """
  plot_image(camera_image)

  xs = []
  ys = []
  colors = []

  for point in projected_points:
    xs.append(point[0])  # width, col
    ys.append(point[1])  # height, row
    colors.append(rgba_func(point[2]))

  plt.scatter(xs, ys, c=colors, s=point_size, edgecolors="none")

    # 强制关闭坐标轴显示
  plt.axis('off')
  plt.gca().set_axis_off()
  plt.gca().xaxis.set_major_locator(plt.NullLocator())
  plt.gca().yaxis.set_major_locator(plt.NullLocator())


def sample_ida_augmentation(is_train,org_size,re_size):
    """Generate ida augmentation values based on ida_config."""
    H, W = org_size
    fH, fW = re_size 
    if is_train:
        # resize = np.random.uniform(*(0.386,0.55))
        resize = 0.5 #这个是固定大小没得说,
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int(
            (1 - np.random.uniform(*(0.0,0.0))) *
            newH) - fH  #这里也可以稍微random,给一点点即可
        # crop_w = int(np.random.uniform(0, max(0, newW - fW)))#这里是一个滑动窗口，具体从哪里出发不确定 #不要随机，固定吧
        crop_w = int(max(0, newW - fW) / 2) #这里是一个滑动窗口，具体从哪里出发不确定
        # crop_w = int(max(0, newW - fW))
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        # if self.ida_aug_conf['rand_flip'] and np.random.choice([0, 1]):
        #     flip = True
        rotate_ida = np.random.uniform(*(-5.4, 5.4))
    else:
        resize = max(fH / H, fW / W)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        crop_h = int(
            (1 - np.mean((0,0))) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        flip = False
        rotate_ida = 0
    return resize, resize_dims, crop, flip, rotate_ida


def depth_transform(cam_depth, resize=0.4867678423494496, resize_dims=(256, 384), crop=(18, 182, 722, 438), flip=False, rotate=0):
    """Transform depth based on ida augmentation configuration.
    otate=2.9495885670585267,crop表示左上角的部分不要了
    Args:
        cam_depth (np array): Nx3, 3: x,y,d.
        resize (float): Resize factor.
        resize_dims (list): Final dimension.
        crop (list): x1, y1, x2, y2
        flip (bool): Whether to flip.
        rotate (float): Rotation value.

    Returns:
        np array: [h/down_ratio, w/down_ratio, d]
    """
# 原图片大小是
# 参考值： (0.4867678423494496, (256, 704), (18, 182, 722, 438), False, 2.9495885670585267)
    H, W = resize_dims
    cam_depth[:, :2] = cam_depth[:, :2] * resize
    cam_depth[:, 0] -= crop[0]
    cam_depth[:, 1] -= crop[1]
    if flip:
        cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

    cam_depth[:, 0] -= W / 2.0
    cam_depth[:, 1] -= H / 2.0

    h = rotate / 180 * np.pi
    rot_matrix = [
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ]
    cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

    cam_depth[:, 0] += W / 2.0
    cam_depth[:, 1] += H / 2.0

    depth_coords = cam_depth[:, :2].astype(np.int16)

    depth_map = np.zeros(resize_dims) #看来没有电云的位置应该是直接mask掉
    valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                  & (depth_coords[:, 0] < resize_dims[1])
                  & (depth_coords[:, 1] >= 0)
                  & (depth_coords[:, 0] >= 0))
    depth_map[depth_coords[valid_mask, 1],
              depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

    return torch.Tensor(depth_map)#高度稀疏，很多地方都是0（是0）




  
  
  
  


#主函数开始
# FILENAME = 'tutorial/frames'
# TODO: Change this to your own setting
def single_depth(filepath,savepath,vis):
    # filepath = '/media/wzh/datasets/openlane/waymo/segment-11392401368700458296_1086_429_1106_429_with_camera_labels.tfrecord'
    FILENAME = filepath

    dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
    for data in tqdm(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        # for index, image in enumerate(frame.images):
        #   image_show(image.image, open_dataset.CameraName.Name.Name(image.name), [3, 3, index+1])
        # plt.show()
        


        (range_images, camera_projections, range_image_top_pose) = parse_range_image_and_camera_projection(frame)

        # print(frame.context)


        # visualize the frame photo
        if vis == True:
            plt.figure(figsize=(25, 20))
            for index, image in enumerate(frame.images):
                image_show(image.image, open_dataset.CameraName.Name.Name(image.name), [3, 3, index+1])
        # plt.show()


        # visualize the range info
        if vis == True:
            plt.figure(figsize=(64, 20))


        
        frame.lasers.sort(key=lambda laser: laser.name)
        if vis ==True:
            show_range_image(get_range_image(range_images,open_dataset.LaserName.TOP, 0), 1)
            # plt.show()

            show_range_image(get_range_image(range_images,open_dataset.LaserName.TOP, 1), 4)
        # plt.show()
        # cp waymo-open-dataset/bazel-genfiles/waymo_open_dataset/label_pb2.py waymo-open-dataset/waymo_open_dataset/label_pb2.py


        points, cp_points = convert_range_image_to_point_cloud(frame,
                                                            range_images,
                                                            camera_projections,
                                                            range_image_top_pose)
        points_ri2, cp_points_ri2 = convert_range_image_to_point_cloud(
            frame,
            range_images,
            camera_projections,
            range_image_top_pose,
            ri_index=1)

        # 3d points in vehicle frame.
        points_all = np.concatenate(points, axis=0)
        points_all_ri2 = np.concatenate(points_ri2, axis=0)
        # camera projection corresponding to each point.
        cp_points_all = np.concatenate(cp_points, axis=0)
        cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)

        # examine point cloud

        # print(points_all.shape)
        # print(cp_points_all.shape)
        # print(points_all[0:2])
        # for i in range(5):
        #     print(points[i].shape)
        #     print(cp_points[i].shape)

        # print(points_all_ri2.shape)
        # print(cp_points_all_ri2.shape)
        # print(points_all_ri2[0:2])
        # for i in range(5):
        #     print(points_ri2[i].shape)
        #     print(cp_points_ri2[i].shape)

        # from IPython.display import Image, display

        # import cv2

        # display(Image('tutorial/3d_point_cloud.png'))

        # cv2.imshow('tutorial/3d_point_cloud.png')

        # visualize camera projection

        images = sorted(frame.images, key=lambda i:i.name)
        cp_points_all_concat = np.concatenate([cp_points_all, points_all], axis=-1)
        cp_points_all_concat_tensor = tf.constant(cp_points_all_concat)

        # The distance between lidar points and vehicle frame origin.
        points_all_tensor = tf.norm(points_all, axis=-1, keepdims=True)
        cp_points_all_tensor = tf.constant(cp_points_all, dtype=tf.int32)

        mask = tf.equal(cp_points_all_tensor[..., 0], images[0].name)

        cp_points_all_tensor = tf.cast(tf.gather_nd(
            cp_points_all_tensor, tf.where(mask)), dtype=tf.float32)
        points_all_tensor = tf.gather_nd(points_all_tensor, tf.where(mask))

        projected_points_all_from_raw_data = tf.concat(
            [cp_points_all_tensor[..., 1:3], points_all_tensor], axis=-1).numpy()


        if vis == True:
            plot_points_on_image(projected_points_all_from_raw_data,
                                images[0], rgba, point_size=5.0)
            plt.show()

        depths = projected_points_all_from_raw_data[:, 2]
        coloring = depths
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 0.0) #取在相机平面前方的点
        mask = np.logical_and(mask, projected_points_all_from_raw_data[:, 0] > 1)
        mask = np.logical_and(mask, projected_points_all_from_raw_data[:, 0] < tf.image.decode_jpeg(images[1].image).shape[1] - 1)
        mask = np.logical_and(mask, projected_points_all_from_raw_data[:, 1] > 1)
        mask = np.logical_and(mask, projected_points_all_from_raw_data[:, 1] < tf.image.decode_jpeg(images[0].image).shape[0] - 1)
        points = projected_points_all_from_raw_data[mask,:]
        coloring = coloring[mask]
        # 和bev depth中map_pointcloud_to_image中最后的输出对齐了
        # projected_points_all_from_raw_data, coloring
        # 为什么这里还有这么多点

        #接下来是depth transform
        # 这里的参数一定要慎重
        #先计算缩放比例

        org_size = [1280, 1920]
        re_size = [512, 832] #降采样8倍后就是最终的结果大小 64, 104

        resize, resize_dims, crop, flip, rotate_ida = sample_ida_augmentation(True,org_size,re_size)


        # tf.image.decode_jpeg(images[0].image).shape 原图片的大小这么看，TensorShape([1280, 1920, 3])
        gt_depths = depth_transform(points,resize, re_size, crop, flip)#也是torch.Size([256, 704])#，这个就已经能用了，修改为【360，480】
        downsample_factor = 8 #降采样8倍,这里还要降采样8倍,需要修改为90，120
        # dbound=[2.0,58.0,0.5] #这个得去看bev的配置
        dbound=[2.0,102,0.5]
        # depth_channels=112,dbound[1]=2xdepth_channels-4
        depth_channels = 200# 增大这个值，远处深度图的数据会更多
        gt_depths = gt_depths.unsqueeze(0).unsqueeze(0) # torch.Size([1, 1, 256, 704])
        # torch.Size([1, 6, 256, 704])
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // downsample_factor,
            downsample_factor,
            W // downsample_factor,
            downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, downsample_factor * downsample_factor)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths) # 为true用1e5替换(非常大的深度），为false用原来的值
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values#8448,256,取最小值所有深度，
        gt_depths = gt_depths.view(B * N, H // downsample_factor,
                                    W // downsample_factor) #相当于每256个选个最小的
        #沿着采样的维度取最小值，因为采样的深度图可能存在多个深度值，取最小值
        gt_depths = (gt_depths -
                        (dbound[0] - dbound[2])) / dbound[2] #感知
        gt_depths = torch.where(
            (gt_depths < depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))#有些深度值可能超出范围，这里就不要了,到这里就可以了
        gt_depths_tmp = F.one_hot(gt_depths.long(),#转换为整数,深度是多少那一维度就有值，太夸张了
                                num_classes=depth_channels + 1).view( #0的就不要吧？
                                    -1, depth_channels + 1)[:, 1:] #depth channanel中只有一个是1

        output=gt_depths[0]
        # maximum = np.max(output)
        # minimum = np.min(output)
        # depth_map = output/maximum #用最大值做归一化,但是不需要
        depth_colored = cv2.applyColorMap(cv2.normalize(output.cpu().numpy(), None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),cv2.COLORMAP_MAGMA)
        # print("save to", ops.join(save_dir, new_name.replace("/", "_")))
        # fig3.savefig(ops.join(save_dir, new_name.replace("/", "_")))
        # save_path='/home/wzh/study/github/bev/BEVDepth'
        segment_name='segment-'+str(frame.context.name)+'_with_camera_labels'
        save_dir = os.path.join(savepath, segment_name)
        os.makedirs(save_dir, exist_ok=True)
        save_dir2=os.path.join(save_dir, str(frame.timestamp_micros)+'00'+'.png')
        # save_dir= os.path.join(savepath, "depth2.png")
        # cv2.imwrite('depth.png', depth_colored)
        cv2.imwrite(save_dir2, depth_colored)
        output_numpy = output.detach().cpu().numpy()
        max_value = np.max(output_numpy)
        if max_value == 0:
            max_value = 1.0  # 如果全为0则保持原值
        # Step 3: 归一化
        normalized_output = output_numpy / max_value
        save_dir3=os.path.join(save_dir, str(frame.timestamp_micros)+'00'+'.npy')
        # Step 4: 保存为npy文件
        np.save(save_dir3, normalized_output)
    # cv2.imwrite(ops.join(save_dir2, 'depth.png'), depth_colored)
    print(f'depth of {save_dir} generated!')
#效果很差，需要改参数

if __name__ == '__main__':
    tag='validation'
    save_path='/media/wzh/datasets/openlane/depth'#这个是我电脑上已经有的文件夹
    load_path='/media/wzh/datasets/openlane/waymo'
    loadpaths=os.path.join(load_path, tag)
    savepath=os.path.join(save_path, tag)
    num_workers = 1 #是否使用并行处理，不然会很慢
    vis =False
    
    
    def get_tfrecord_files(filepath):
        """获取目录下所有 .tfrecord 文件的完整路径"""
        all_files = os.listdir(filepath)
        tfrecord_files = [
            os.path.join(filepath, f) 
            for f in all_files 
            if f.endswith('.tfrecord') and os.path.isfile(os.path.join(filepath, f))
        ]
        return tfrecord_files

    tfrecord_files = get_tfrecord_files(loadpaths)
    print(f"找到 {len(tfrecord_files)} 个 TFRecord 文件：")
    for file in tfrecord_files:
        print(file)
    
    task_args = [(file, save_path, vis) for file in tfrecord_files]
    
    
    
    if num_workers > 1:
        import multiprocessing

        def process_single_file(args):
            """包装函数用于并行处理"""
            filepath, savepath, vis = args  # 新增文件索引参数
            
            # 动态分配 GPU（轮询模式）
            gpu_id = os.getpid()
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            # 显存管理初始化
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpus[0], True)
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*12)]  # 为每个 GPU 保留 12GB
                    )
                except RuntimeError as e:
                    print(e)
            
            try:
                single_depth(filepath, savepath=savepath, vis=vis)
                return True
            except Exception as e:
                print(f"处理文件 {filepath} 失败: {str(e)}")
                return False
            finally:
                # 显存强制释放
                tf.keras.backend.clear_session()
                import gc
                gc.collect()
                
                
        # import multiprocessing
        
        # 生成简化版任务参数
        task_args = [(file, save_path, vis) for file in tfrecord_files]
        
        with multiprocessing.Pool(processes=num_workers) as pool:
            # 使用 imap_unordered 提高效率
            results = list(tqdm(
                pool.imap_unordered(process_single_file, task_args),
                total=len(task_args),
                desc="处理进度"
            ))
            success_count = sum(results)
            print(f"处理完成: 成功 {success_count}/{len(tfrecord_files)} 个文件")
    
    else:
        for filepath in tfrecord_files:
            print(f'开始处理文件: {filepath}')
            single_depth(filepath,savepath=save_path,vis=vis)
        print(print(f"处理完成: 成功 {len(tfrecord_files)} 个文件"))


# export 


