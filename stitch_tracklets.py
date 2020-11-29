import numpy as np
import torch
import yaml
import os
from utils.tracking_utils import *
from utils.kalman_filter import KalmanBoxTracker
from scipy.optimize import linear_sum_assignment
import sys
import argparse
import time

def associate_instances(previous_instances, current_instances, overlaps,  pose, association_weights):
    pose = pose.cpu().float()
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    p_n = len(previous_instances.keys())
    c_n = len(current_instances.keys())

    association_costs = torch.zeros(p_n, c_n)
    prev_ids = []
    current_ids = []

    current_instances_prev = {}
    for i, (k, v) in enumerate(previous_instances.items()):
        #v['kalman_bbox'][0:3] += pose[:3, 3]
        #v['kalman_bbox'][0:3] = torch.matmul(v['kalman_bbox'][0:3],pose[:3, :3])
        #v['bbox'][0:3] = v['kalman_bbox'][0:3] - v['kalman_bbox'][4:]/2
        #v['bbox'][3:] = v['kalman_bbox'][0:3] + v['kalman_bbox'][4:] / 2
        pass

    for i, (k, v) in enumerate(previous_instances.items()):
        prev_ids.append(k)
        for j, (k1, v1) in enumerate(current_instances.items()):
            if v1['class'] ==  v['class'] and k1 not in overlaps:
                #cost_3d = 1 - IoU(v1['bbox'], v['bbox'])
                #if k1 in current_instances_prev:
                #    cost_3d = min (cost_3d, 1 - IoU(current_instances_prev[k1]['bbox'], v['bbox']))
                #if cost_3d > 0.75:
                #    cost_3d = 1e8
                #if v1['bbox_proj'] is not None:
                #    cost_2d = 1 - IoU(v1['bbox_proj'], v['bbox_proj'])
                #    if k1 in current_instances_prev:
                #        cost_2d = min(cost_2d, 1 - IoU(current_instances_prev[k1]['bbox_proj'], v['bbox_proj']))

                #    if cost_2d > 0.75:
                #        cost_2d = 1e8
                #else:
                #    cost_2d = 0

                cost_center = euclidean_dist(v1['kalman_bbox'], v['kalman_bbox'])
                if k1 in current_instances_prev:
                    cost_center = min(cost_center, euclidean_dist(current_instances_prev[k1]['kalman_bbox'],v['kalman_bbox']))
                if cost_center > 5:
                    cost_center = 1e8

                #feature_cost = 1 - cos(v1['mean'], v['mean'])
                #if k1 in current_instances_prev:
                #    feature_cost = min(feature_cost, 1 - cos(current_instances_prev[k1]['mean'], v['mean']))
                #if feature_cost > 0.5:
                #    feature_cost = 1e8
                costs = torch.tensor([0, 0, cost_center, 0])

                for idx, a_w in enumerate(association_weights):
                    association_costs[i, j] += a_w * costs[idx]
            else:
                association_costs[i, j] = 1e8

            if i == 0:
                current_ids.append(k1)

    idxes_1, idxes_2 = linear_sum_assignment(association_costs.cpu().detach())

    associations = []

    for i1, i2 in zip(idxes_1, idxes_2):
        # max_cost = torch.sum((previous_instances[prev_ids[i1]]['var'][0,-3:]/2)**2)
        if association_costs[i1][i2] < 1e8:
            associations.append((prev_ids[i1], current_ids[i2]))

    return association_costs, associations

def associate_instances_overlapping_frames(previous_ins_label, current_ins_label):

    previous_instance_ids, c_p = np.unique(previous_ins_label, return_counts=True)
    current_instance_ids, c_c = np.unique(current_ins_label, return_counts=True)

    previous_instance_ids = [x for i,x in enumerate(previous_instance_ids) if c_p[i] > 25] #
    current_instance_ids = [x for i, x in enumerate(current_instance_ids) if c_c[i] > 50] #

    p_n = len(previous_instance_ids) -1
    c_n = len(current_instance_ids) -1

    prev_ids = []
    current_ids = []

    association_costs = torch.zeros(p_n, c_n)
    for i, p_id in enumerate(previous_instance_ids[1:]):
        prev_ids.append(p_id)
        for j, c_id in enumerate(current_instance_ids[1:]):
            intersection = np.sum( (previous_ins_label==p_id) & (current_ins_label == c_id) )

            union =  np.sum(previous_ins_label==p_id) + np.sum(current_ins_label == c_id) - intersection
            iou = intersection/union
            cost = 1 - iou
            association_costs[i, j] = cost if cost < 0.50 else 1e8
            if i == 0:
                current_ids.append(c_id)

    idxes_1, idxes_2 = linear_sum_assignment(association_costs.cpu().detach())
    associations = []
    association_costs_matched = []
    for i1, i2 in zip(idxes_1, idxes_2):
        if association_costs[i1][i2] < 1e8:
            associations.append((prev_ids[i1], current_ids[i2]))
            association_costs_matched.append(association_costs[i1][i2])

    return association_costs_matched,  associations

def main(FLAGS):
    data_cfg = 'data/SemanticKitti/semantic-kitti.yaml'
    DATA = yaml.safe_load(open(data_cfg, 'r'))
    split = 'valid'
    dataset = 'data/SemanticKitti'

    prediction_dir =  FLAGS.predictions
    if split == 'valid':
        prediction_path = '{}/val_probs'.format(prediction_dir)
    else:
        prediction_path = '{}/probs'.format(prediction_dir)
    n_test_frames = FLAGS.n_test_frames

    association_weights = [FLAGS.iou_3d, FLAGS.iou_2d, FLAGS.center, FLAGS.feature]

    association_names = ['3d', '2d', 'cen', 'fet']
    assoc_saving = [asc_type for idx, asc_type in enumerate(association_names) if
                    association_weights[idx] > 0]
    assoc_saving.append(str(n_test_frames))
    assoc_saving = '_'.join(assoc_saving)

    save_path = '{}/stitch'.format(prediction_dir)+assoc_saving
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(data_cfg, 'r') as stream:
        doc = yaml.safe_load(stream)
        learning_map_doc = doc['learning_map']
        inv_learning_map_doc = doc['learning_map_inv']

    inv_learning_map = np.zeros((np.max([k for k in inv_learning_map_doc.keys()]) + 1), dtype=np.int32)
    for k, v in inv_learning_map_doc.items():
        inv_learning_map[k] = v

    # get number of interest classes, and the label mappings
    # class
    class_remap = DATA["learning_map"]
    class_inv_remap = DATA["learning_map_inv"]
    class_ignore = DATA["learning_ignore"]
    nr_classes = len(class_inv_remap)
    class_strings = DATA["labels"]

    # make lookup table for mapping
    # class
    maxkey = max(class_remap.keys())

    # +100 hack making lut bigger just in case there are unknown labels
    class_lut = np.zeros((maxkey + 100), dtype=np.int32)
    class_lut[list(class_remap.keys())] = list(class_remap.values())

    # class
    ignore_class = [cl for cl, ignored in class_ignore.items() if ignored]

    print("Ignoring classes: ", ignore_class)

    # get test set
    test_sequences = DATA["split"][split]

    # get label paths

    poses = []

    test_sequences = FLAGS.sequences

    for sequence in test_sequences:
        calib = parse_calibration(os.path.join(dataset, "sequences", '{0:02d}'.format(sequence), "calib.txt"))
        poses_f64 = parse_poses(os.path.join(dataset, "sequences", '{0:02d}'.format(sequence), "poses.txt"), calib)
        poses.append([pose.astype(np.float32) for pose in poses_f64])


    for poses_seq, sequence in zip(poses, test_sequences):
        point_names = []
        point_paths = os.path.join(dataset, "sequences", '{0:02d}'.format(sequence), "velodyne")
        # populate the label names
        seq_point_names = sorted(
            [os.path.join(point_paths, fn) for fn in os.listdir(point_paths) if fn.endswith(".bin")])

        point_names.extend(seq_point_names)

        prev_instances = {}
        overlap_history = {}

        if not os.path.exists(os.path.join(save_path, 'sequences', '{0:02d}'.format(sequence))):
            os.makedirs(os.path.join(save_path, 'sequences', '{0:02d}'.format(sequence)))

        if not os.path.exists(os.path.join(save_path, 'sequences', '{0:02d}'.format(sequence), 'predictions')):
            os.makedirs(os.path.join(save_path, 'sequences', '{0:02d}'.format(sequence), 'predictions'))

        for idx, point_file in zip(range(len(point_names)), point_names):
            times = []
            times.append(time.time())
            pose = poses_seq[idx]

            #load current frame
            sem_path = os.path.join(prediction_path, '{0:02d}_{1:07d}.npy'.format(sequence,idx))
            ins_path = os.path.join(prediction_path, '{0:02d}_{1:07d}_i.npy'.format(sequence,idx))
            fet_path = os.path.join(prediction_path, '{0:02d}_{1:07d}_f.npy'.format(sequence, idx))

            label_sem_class = np.load(sem_path)
            label_inst = np.load(ins_path)
            frame_points = np.fromfile(point_file, dtype=np.float32)
            points = frame_points.reshape((-1, 4))
            hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
            new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)
            points = new_points[:, :3]

            things = (label_sem_class < 9) & (label_sem_class > 0)
            ins_ids = np.unique(label_inst * things)

            if os.path.exists(fet_path):
                features = np.load(fet_path, allow_pickle=True).tolist()
            else:
                features = {}
                for ins_id in ins_ids:
                    features[ins_id] = torch.from_numpy(np.zeros((1,1)))

            projections = do_range_projection(points)
            points = torch.from_numpy(points)
            new_instances = {}

            label_inst = torch.from_numpy(label_inst.astype(np.int32))

            # get instances from current frames to track
            for ins_id in ins_ids:
                if ins_id == 0:
                    continue
                if int(ins_id) not in features:
                    ids = np.where(label_inst == ins_id)
                    label_inst[ids] = 0
                    continue

                mean = features[int(ins_id)]
                ids = np.where(label_inst == ins_id)
                if ids[0].shape[0] < 25:
                    label_inst[ids] = 0
                    continue

                (values, counts) = np.unique(label_sem_class[ids], return_counts=True)
                inst_class = values[np.argmax(counts)]

                new_ids = remove_outliers(points[ids])
                new_ids = ids[0][new_ids]

                bbox, kalman_bbox = get_bbox_from_points(points[ids])
                tracker = KalmanBoxTracker(kalman_bbox, ins_id)
                center = get_median_center_from_points(points[ids])
                bbox_proj = get_2d_bbox(projections[:, new_ids])
                new_instances[ins_id] = {'life': 5, 'bbox': bbox, 'bbox_proj': bbox_proj, 'center' : center, 'n_point':ids[0].shape[0],
                                         'tracker': tracker, 'kalman_bbox': kalman_bbox, 'mean':mean, 'class' : inst_class}
            new_instances_prev = {}
            overlaps = {}
            overlap_scores = {}
            # if multi frame prediction
            times.append(time.time()) # loading time
            if idx > 0:
                for i in range(1, n_test_frames):
                    if idx - i < 0:
                        continue
                    # load previous frames which are predicted with current frame in a multi-frame fashion
                    prev_inst_path = os.path.join(prediction_path, '{0:02d}_{1:07d}_{2}_i.npy'.format(sequence, idx-i, idx))
                    prev_sem_path = os.path.join(prediction_path,
                                                  '{0:02d}_{1:07d}.npy'.format(sequence, idx - i))

                    fet_path = os.path.join(prediction_path, '{0:02d}_{1:07d}_{2}_f.npy'.format(sequence, idx - i, idx))

                    if not os.path.exists(prev_inst_path):
                        continue
                    prev_inst = np.load(prev_inst_path)
                    prev_sem = np.load(prev_sem_path)

                    if os.path.exists(fet_path):
                        features = np.load(fet_path, allow_pickle=True).tolist()
                    else:
                        features = {}
                        for ins_id in ins_ids:
                            features[ins_id] = torch.from_numpy(np.zeros((1, 1)))

                    prev_inst_orig_path = os.path.join(prediction_path,
                                                  '{0:02d}_{1:07d}_i.npy'.format(sequence, idx - i))
                    prev_inst_orig = np.load(prev_inst_orig_path)

                    things = (prev_sem < 9) & (prev_sem > 0)

                    # associate instances from previous frame pred_n and current prediction which contain pred_n, pred_n+1
                    association_costs, associations = associate_instances_overlapping_frames(prev_inst_orig* things, prev_inst* things)

                    for cost, (id1, id2) in zip(association_costs, associations):
                        if id2 not in overlaps:
                            overlap_scores[id2] = cost
                        elif overlap_scores[id2] > cost:
                            continue
                        elif overlap_scores[id2] < cost:
                            overlap_scores[id2] = cost
                        if id1 in overlap_history: #get track id of instance from previous frame
                            id1 = overlap_history[id1]
                        overlaps[id2] = id1

                    prev_point_path = os.path.join(dataset, "sequences", '{0:02d}'.format(int(sequence)), "velodyne", '{0:06d}.bin'.format(idx-i))
                    #pose = poses[0][idx-i]
                    frame_points = np.fromfile(prev_point_path, dtype=np.float32)
                    points = frame_points.reshape((-1, 4))
                    hpoints = np.hstack((points[:, :3], np.ones_like(points[:, :1])))
                    new_points = np.sum(np.expand_dims(hpoints, 2) * pose.T, axis=1)
                    points = new_points[:, :3]

                    points = torch.from_numpy(points)
                    projections = do_range_projection(points.cpu().detach().numpy())
                    prev_inst = torch.from_numpy(prev_inst.astype(np.int32))

                    # add instances for assocaition which are not overlapped
                    ins_ids = np.unique(prev_inst * things)
                    for ins_id in ins_ids:
                        if ins_id == 0:
                            continue
                        if int(ins_id) not in features:
                            ids = np.where(prev_inst == ins_id)
                            prev_inst[ids] = 0
                            continue

                        ids = np.where(prev_inst == ins_id)
                        if ids[0].shape[0] < 25:
                            prev_inst[ids] = 0
                            continue

                        mean = features[int(ins_id)]
                        new_ids = remove_outliers(points[ids])
                        new_ids = ids[0][new_ids]

                        (values, counts) = np.unique(prev_sem[ids], return_counts=True)
                        inst_class = values[np.argmax(counts)]

                        bbox, kalman_bbox = get_bbox_from_points(points[ids])
                        center = get_median_center_from_points(points[ids])
                        bbox_proj = get_2d_bbox(projections[:, new_ids])
                        tracker = KalmanBoxTracker(kalman_bbox, ins_id)
                        new_instances_prev[ins_id] = {'life': 5, 'bbox': bbox, 'bbox_proj': bbox_proj,
                                                 'tracker': tracker, 'kalman_bbox': kalman_bbox, 'mean': mean,
                                                      'center':center, 'class' : inst_class}

            times.append(time.time())  # overlap associate times
            #if len(prev_instances.keys()) > 0:
            #    association_costs, associations = associate_instances(prev_instances, new_instances, overlaps,
            #                                                            torch.from_numpy(pose), association_weights)
            associations = []
            times.append(time.time()) # assoc time from prev
            # if there was instances from previous frames
            if len(prev_instances.keys()) > 0:
                #firstly associate overlapping instances
                for (new_id, prev_id) in overlaps.items():
                    ins_points = torch.where((label_inst == new_id))
                    if not new_id in new_instances or prev_id not in prev_instances:
                        continue
                    overlap_history[new_id] = prev_id#add tracking id
                    label_inst[ins_points[0]] = prev_id
                    prev_instances[prev_id]['bbox_proj'] = new_instances[new_id]['bbox_proj']
                    prev_instances[prev_id]['mean'] = new_instances[new_id]['mean']
                    prev_instances[prev_id]['center'] = new_instances[new_id]['center']

                    prev_instances[prev_id]['life'] += 1
                    prev_instances[prev_id]['tracker'].update(new_instances[new_id]['kalman_bbox'], prev_id)
                    prev_instances[prev_id]['kalman_bbox'] = torch.from_numpy(prev_instances[prev_id]['tracker'].get_state()).float()
                    prev_instances[prev_id]['bbox'] = kalman_box_to_eight_point(prev_instances[prev_id]['kalman_bbox'])

                    del new_instances[new_id]

                for prev_id, new_id in associations:
                    if new_id in overlaps:
                        continue
                    # associate  instances which are not overlapped
                    ins_points = torch.where((label_inst == new_id))
                    label_inst[ins_points[0]] = prev_id
                    overlap_history[new_id] = prev_id
                    prev_instances[prev_id]['bbox_proj'] = new_instances[new_id]['bbox_proj']
                    prev_instances[prev_id]['mean'] = new_instances[new_id]['mean']
                    prev_instances[prev_id]['center'] = new_instances[new_id]['center']

                    prev_instances[prev_id]['life'] += 1
                    prev_instances[prev_id]['tracker'].update(new_instances[new_id]['kalman_bbox'], prev_id)
                    prev_instances[prev_id]['kalman_bbox'] = torch.from_numpy(prev_instances[prev_id]['tracker'].get_state()).float()
                    prev_instances[prev_id]['bbox'] = kalman_box_to_eight_point(prev_instances[prev_id]['kalman_bbox'])

                    del new_instances[new_id]

            for ins_id, instance in new_instances.items():  # add new instances to history
                ids = np.where(label_inst == ins_id)
                if ids[0].shape[0] < 50:
                    continue
                prev_instances[ins_id] = instance

            # kill instances which are not tracked for a  while
            dont_track_ids = []
            for ins_id in prev_instances.keys():
                if prev_instances[ins_id]['life'] == 0:
                    dont_track_ids.append(ins_id)
                prev_instances[ins_id]['life'] -= 1

            for ins_id in dont_track_ids:
                del prev_instances[ins_id]

            times.append(time.time()) # updating ids

            ins_preds = label_inst.cpu().numpy()

            #clean instances which have too few points
            for ins_id in np.unique(ins_preds):
                if ins_id == 0:
                    continue
                valid_ind = np.argwhere(ins_preds == ins_id)[:, 0]
                ins_preds[valid_ind] = ins_id+20
                if valid_ind.shape[0] < 25:
                    ins_preds[valid_ind] = 0

            for sem_id in np.unique(label_sem_class):
                if sem_id < 1 or sem_id > 8:
                    valid_ind = np.argwhere((label_sem_class == sem_id) & (ins_preds == 0))[:, 0]
                    ins_preds[valid_ind] = sem_id

            #write instances to label file which is binary
            ins_preds = ins_preds.astype(np.int32)
            new_preds = np.left_shift(ins_preds, 16)

            sem_pred = label_sem_class.astype(np.int32)
            inv_sem_labels = inv_learning_map[sem_pred]
            new_preds = np.bitwise_or(new_preds,inv_sem_labels)

            new_preds.tofile('{}/{}/{:02d}/predictions/{:06d}.label'.format(save_path, 'sequences', sequence, idx))
            times.append(time.time()) #writing time
            #print ('load, overlap, assoc, update, write')
            #for i in range(1, len(times)):
            #    print (times[i]-times[i-1])
            print("{}/{} ".format(idx, len(point_names)), end="\r", flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./stitch_tracklets.py")
    parser.add_argument(
        '--n_test_frames',
        '-n',
        type=int,
        default=1
    )

    parser.add_argument(
        '--3d', '-3d',
        dest='iou_3d',
        type=float,
        default=0
    )

    parser.add_argument(
        '--2d', '-2d',
        dest='iou_2d',
        type=float,
        default=0
    )

    parser.add_argument(
        '--center', '-c',
        dest='center',
        type=float,
        default=0
    )

    parser.add_argument(
        '--feature', '-f',
        dest='feature',
        type=float,
        default=0
    )

    parser.add_argument(
        '--sequences', '-s',
        dest='sequences',
        type=str,
        default='8'
    )

    parser.add_argument(
        '--predictions', '-p',
        dest='predictions',
        type=str,
        required=True
    )

    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.sequences = [int(x) for x in FLAGS.sequences.split(',')]

    main(FLAGS)
