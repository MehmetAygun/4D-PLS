import numpy as np
import random,colorsys


def random_colors(N, bright=True, seed=0):
    brightness = 1.0 if bright else 0.7
    hsv = [(0.15 + i / float(N), 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.seed(seed)
    random.shuffle(colors)
    return colors


def gaussian_colors(N):
    hsv = [(0, i / float(N), 1) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    return colors


def parse_calibration(filename):
    """ read calibration file with given filename
    Returns
    -------
    dict
        Calibration matrices as 4x4 numpy arrays.
    """
    calib = {}

    calib_file = open(filename)
    for line in calib_file:
        key, content = line.strip().split(":")
        values = [float(v) for v in content.strip().split()]

        pose = np.zeros((4, 4))
        pose[0, 0:4] = values[0:4]
        pose[1, 0:4] = values[4:8]
        pose[2, 0:4] = values[8:12]
        pose[3, 3] = 1.0

        calib[key] = pose

    calib_file.close()

    return calib


def parse_poses(filename, calibration):
	""" read poses file with per-scan poses from given filename
		Returns
		-------
		list
			list of poses as 4x4 numpy arrays.
	"""
	file = open(filename)

	poses = []

	Tr = calibration["Tr"]
	Tr_inv = np.linalg.inv(Tr)

	for line in file:
		values = [float(v) for v in line.strip().split()]

		pose = np.zeros((4, 4))
		pose[0, 0:4] = values[0:4]
		pose[1, 0:4] = values[4:8]
		pose[2, 0:4] = values[8:12]
		pose[3, 3] = 1.0

		poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr)))

	return poses