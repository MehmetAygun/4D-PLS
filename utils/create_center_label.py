import numpy as np
import yaml
from helper import parse_calibration, parse_poses, random_colors, gaussian_colors
from scipy.stats import multivariate_normal
from os import listdir
from os.path import exists, join, isdir
import sys

sequences = ['{:02d}'.format(i) for i in range(11)]
path = '../data/SemanticKitti/'
covariance = np.diag(np.array([1, 1, 1]))
center_point = np.zeros((1, 3))

for seq in sequences:
    velo_path = join(path, 'sequences', seq, 'velodyne')
    frames = np.sort([vf[:-4] for vf in listdir(velo_path) if vf.endswith('.bin')])
    seq_path = join(path, 'sequences', seq)
    calib_file = join(path, 'sequences', seq, 'calib.txt')
    pose_file = join(path, 'sequences', seq, 'poses.txt')

    calibration = parse_calibration(calib_file)
    poses_f64 = parse_poses(pose_file, calibration)
    poses = ([pose.astype(np.float32) for pose in poses_f64])
    print('Processing sequence:' + seq)
    for idx, frame in enumerate(frames):
        velo_file = join(seq_path, 'velodyne', frame + '.bin')
        label_file = join(seq_path, 'labels', frame + '.label')
        save_path = join(seq_path, 'labels', frame + '.center')
        frame_labels = np.fromfile(label_file, dtype=np.int32)
        ins_labels = frame_labels & 0xFFFF0000
        pose = poses[idx]
        frame_points = np.fromfile(velo_file, dtype=np.float32)
        points = frame_points.reshape((-1, 4))
        hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
        new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)
        sem_ins_labels = np.unique(ins_labels)
        center_labels = np.zeros((new_points.shape[0], 4))

        for _, semins in enumerate(sem_ins_labels):
            valid_ind = np.argwhere(ins_labels == semins)[:, 0]
            if semins == 0 or valid_ind.shape[0] < 5:  # background classes and small groups
                continue

            x_min = np.min(new_points[valid_ind, 0])
            x_max = np.max(new_points[valid_ind, 0])
            y_min = np.min(new_points[valid_ind, 1])
            y_max = np.max(new_points[valid_ind, 1])
            z_min = np.min(new_points[valid_ind, 2])
            z_max = np.max(new_points[valid_ind, 2])

            center_point[0][0] = (x_min + x_max) / 2
            center_point[0][1] = (y_min + y_max) / 2
            center_point[0][2] = (z_min + z_max) / 2

            gaussians = multivariate_normal.pdf(new_points[valid_ind, 0:3], mean=center_point[0, :3], cov=covariance)
            gaussians = (gaussians - min(gaussians)) / (max(gaussians) - min(gaussians))
            center_labels[valid_ind, 0] = gaussians  # first loc score for centerness
            center_labels[valid_ind, 1:4] = center_point[0, :3] - new_points[valid_ind,
                                                                  0:3]  # last 3 for offset to center

        np.save(save_path, center_labels)
