# 作者：酱油与清洛 https://www.bilibili.com/read/cv6933475 出处：bilibili
# 修改：Leo

import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import time

tf.enable_eager_execution()

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from mayavi import mlab

# 配置GPU显存自适应分配，防止占用所有显存
# set GPU memory to adaptively be allocated avoid all the memory being occupied
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print('info: following GPU is setted as memory_growth:')
            print(gpu)
    except RuntimeError as e:
        print(e)
else:
    print('info: there have no GPUs in this computer')

@mlab.animate(delay=100)
def updateAnimation():
	framenumber = 0
	FILENAME = '/media/wzh/datasets/openlane/waymo/archived_files_training_training_0000/segment-10226164909075980558_180_000_200_000_with_camera_labels.tfrecord'
	dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
	MAXPOINT = 200000
	for data in dataset:
		framenumber += 1
		print('Frame Number : {}'.format(framenumber))
		frame = open_dataset.Frame()
		frame.ParseFromString(bytearray(data.numpy()))

		(range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

		points, cp_points = frame_utils.convert_range_image_to_point_cloud(
			frame,
			range_images,
			camera_projections,
			range_image_top_pose)

		x = [0]*MAXPOINT
		y = [0]*MAXPOINT
		z = [0]*MAXPOINT
		for point in points:
			# print('len(point)=',len(point))
			for i in range(len(point)):
				x[i] = point[i][0]
				y[i] = point[i][1]
				z[i] = point[i][2]
			fig.mlab_source.set(x=x, y=y, z=z, mode="point")
		yield

MAXPOINT = 200000
fig = mlab.points3d(
	np.zeros(MAXPOINT), 
	np.zeros(MAXPOINT), 
	np.zeros(MAXPOINT),
	mode="point")
updateAnimation()
mlab.show() 
