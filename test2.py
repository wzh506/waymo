import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset
FILENAME = '/media/wzh/datasets/openlane/waymo/archived_files_training_training_0000/segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord'
dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
for data in dataset:
	frame = open_dataset.Frame()
	frame.ParseFromString(bytearray(data.numpy()))
	break

(range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_camera_image(camera_image, camera_labels, layout, cmap=None):
    ax = plt.subplot(*layout)
	
	 # Draw the camera labels.
    for camera_labels in frame.camera_labels:
	 	# Ignore camera labels that do not correspond to this camera.
        if camera_labels.name != camera_image.name:
            continue
        for label in camera_labels.labels:
	    	# Draw the object bounding box.
            ax.add_patch(patches.Rectangle(xy=(label.box.center_x - 0.5 * label.box.length,
	           							   label.box.center_y - 0.5 * label.box.width),
	       							       width=label.box.length,
	       								   height=label.box.width,
	       							       linewidth=1,
	       								   edgecolor='red',
	       								   facecolor='none'))

	 # Show the camera image.
    plt.imshow(tf.image.decode_jpeg(camera_image.image), cmap=cmap)
    plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
    plt.grid(False)
    plt.axis('off')

		


plt.figure(figsize=(25, 20))

for index, image in enumerate(frame.images):
	show_camera_image(image, frame.camera_labels, [3, 3, index+1])

plt.figure(figsize=(64, 20))
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

def get_range_image(laser_name, return_index):
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
frame.lasers.sort(key=lambda laser: laser.name)
show_range_image(get_range_image(open_dataset.LaserName.TOP, 0), 1)
show_range_image(get_range_image(open_dataset.LaserName.TOP, 1), 4)
points, cp_points = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose)
points_ri2, cp_points_ri2 = frame_utils.convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=1)

# 3d points in vehicle frame.
points_all = np.concatenate(points, axis=0)
points_all_ri2 = np.concatenate(points_ri2, axis=0)
# camera projection corresponding to each point.
cp_points_all = np.concatenate(cp_points, axis=0)
cp_points_all_ri2 = np.concatenate(cp_points_ri2, axis=0)

'''
print(points_all.shape)
print(cp_points_all.shape)
print(points_all[0:2])
for i in range(5):
	print(points[i].shape)
	print(cp_points[i].shape)

print(points_all_ri2.shape)
print(cp_points_all_ri2.shape)
print(points_all_ri2[0:2])
for i in range(5):
	print(points_ri2[i].shape)
	print(cp_points_ri2[i].shape)
'''
from IPython.display import Image, display
# display(Image('/content/waymo-od/tutorial/3d_point_cloud.png'))
print('over')




