import numpy as np

file = "/home/zhaohui1.wang/github/datasets/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin"

pc = np.frombuffer(open(file, "rb").read(), dtype=np.float32)
pc = pc.reshape(-1, 5)[:, :4]

x, y, z, intensity = pc.T

arr = np.zeros(x.shape[0] + y.shape[0] + z.shape[0] + intensity.shape[0], dtype=np.float32)
arr[::4] = x
arr[1::4] = y
arr[2::4] = z
arr[3::4] = intensity
arr.astype('float32').tofile('kitti.bin')
