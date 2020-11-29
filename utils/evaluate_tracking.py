#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
#https://github.com/PRBonn/semantic-kitti-ap
import argparse
import os
import yaml
import numpy as np
import time
import json

from eval_mopt import MOPTEval
from eval_np import Panoptic4DEval
from eval_mot import trackingEvaluation, stat

# possible splits
splits = ["train", "valid", "test"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./evaluate_panoptic.py")
    parser.add_argument(
        '--dataset',
        '-d',
        type=str,
        required=True,
        help='Dataset dir. No Default',
    )
    parser.add_argument(
        '--predictions',
        '-p',
        type=str,
        required=None,
        help='Prediction dir. Same organization as dataset, but predictions in'
             'each sequences "prediction" directory. No Default. If no option is set'
             ' we look for the labels in the same directory as dataset')
    parser.add_argument(
        '--split',
        '-s',
        type=str,
        required=False,
        choices=["train", "valid", "test"],
        default="valid",
        help='Split to evaluate on. One of ' + str(splits) + '. Defaults to %(default)s',
    )
    parser.add_argument(
        '--data_cfg',
        '-dc',
        type=str,
        required=False,
        default="config/semantic-kitti.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--limit',
        '-l',
        type=int,
        required=False,
        default=None,
        help='Limit to the first "--limit" points of each scan. Useful for'
             ' evaluating single scan from aggregated pointcloud.'
             ' Defaults to %(default)s',
    )
    parser.add_argument(
        '--min_inst_points',
        type=int,
        required=False,
        default=50,
        help='Lower bound for the number of points to be considered instance',
    )
    parser.add_argument(
        '--output',
        type=str,
        required=False,
        default=None,
        help='Output directory for scores.txt and detailed_results.html.',
    )

    start_time = time.time()

    FLAGS, unparsed = parser.parse_known_args()

    # fill in real predictions dir
    if FLAGS.predictions is None:
        FLAGS.predictions = FLAGS.dataset

    # print summary of what we will do
    print("*" * 80)
    print("INTERFACE:")
    print("Data: ", FLAGS.dataset)
    print("Predictions: ", FLAGS.predictions)
    print("Split: ", FLAGS.split)
    print("Config: ", FLAGS.data_cfg)
    print("Limit: ", FLAGS.limit)
    print("Min instance points: ", FLAGS.min_inst_points)
    print("Output directory", FLAGS.output)
    print("*" * 80)

    # assert split
    assert (FLAGS.split in splits)

    # open data config file
    DATA = yaml.safe_load(open(FLAGS.data_cfg, 'r'))

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

    ignore = np.array(ignore_class, dtype=np.int64)

    things = ['car', 'truck', 'bicycle', 'motorcycle', 'other-vehicle', 'person', 'bicyclist', 'motorcyclist']
    stuff = [
        'road', 'sidewalk', 'parking', 'other-ground', 'building', 'vegetation', 'trunk', 'terrain', 'fence', 'pole',
        'traffic-sign'
    ]
    all_classes = things + stuff

    print("Ignoring classes: ", ignore_class)

    # get test set
    test_sequences = DATA["split"][FLAGS.split]

    # get label paths
    label_names = []
    for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        label_paths = os.path.join(FLAGS.dataset, "sequences", sequence, "labels")
        # populate the label names
        seq_label_names = sorted([os.path.join(label_paths, fn) for fn in os.listdir(label_paths) if fn.endswith(".label")])
        label_names.extend(seq_label_names)
    # print(label_names)

    # get predictions paths
    pred_names = []
    for sequence in test_sequences:
        sequence = '{0:02d}'.format(int(sequence))
        pred_paths = os.path.join(FLAGS.predictions, "sequences", sequence, "predictions")
        # populate the label names
        seq_pred_names = sorted([os.path.join(pred_paths, fn) for fn in os.listdir(pred_paths) if fn.endswith(".label")])
        pred_names.extend(seq_pred_names)

    # check that I have the same number of files
    assert (len(label_names) == len(pred_names))

    print("Evaluating sequences: ", end="", flush=True)
    # open each file, get the tensor, and make the iou comparison

    complete = len(label_names)
    count = 0
    percent = 10

    #lists for mot evaluation
    gt_sem  = []
    gt_inst = []
    pred_sem = []
    pred_inst = []

    class_evaluator_mopt = MOPTEval(nr_classes, None, ignore_class, min_points=FLAGS.min_inst_points)
    class_evaluator_pq4d = Panoptic4DEval(nr_classes, None, ignore_class, offset=2 ** 32, min_points=FLAGS.min_inst_points)

    scores = np.zeros((8, 8))  # 8 classes, 8 metric  for clear mot metrics

    for label_file, pred_file in zip(label_names, pred_names):
        count = count + 1
        if 100 * count / complete > percent:
            print("{}% ".format(percent), end="", flush=True)
            percent = percent + 10
        # print("evaluating label ", label_file, "with", pred_file)
        # open label

        label = np.fromfile(label_file, dtype=np.uint32)

        u_label_sem_class = class_lut[label & 0xFFFF]  # remap to xentropy format
        u_label_inst = label >> 16
        if FLAGS.limit is not None:
            u_label_sem_class = u_label_sem_class[:FLAGS.limit]
            u_label_sem_cat = u_label_sem_cat[:FLAGS.limit]
            u_label_inst = u_label_inst[:FLAGS.limit]

        label = np.fromfile(pred_file, dtype=np.uint32)

        u_pred_sem_class = class_lut[label & 0xFFFF]  # remap to xentropy format
        u_pred_inst = label >> 16

        if FLAGS.limit is not None:
            u_pred_sem_class = u_pred_sem_class[:FLAGS.limit]
            u_pred_sem_cat = u_pred_sem_cat[:FLAGS.limit]
            u_pred_inst = u_pred_inst[:FLAGS.limit]

        class_evaluator_mopt.addBatch(u_pred_sem_class, u_pred_inst, u_label_sem_class, u_label_inst)
        class_evaluator_pq4d.addBatch(u_pred_sem_class, u_pred_inst, u_label_sem_class, u_label_inst)

        for cl in ignore:
            # make a mask for this class
            gt_not_in_excl_mask = u_label_sem_class != cl
            # remove all other points
            u_pred_sem_class = u_pred_sem_class[gt_not_in_excl_mask]
            u_label_sem_class = u_label_sem_class[gt_not_in_excl_mask]
            u_pred_inst = u_pred_inst[gt_not_in_excl_mask]
            u_label_inst = u_label_inst[gt_not_in_excl_mask]

        gt_sem.append(u_label_sem_class)
        gt_inst.append(u_label_inst)

        pred_sem.append(u_pred_sem_class)
        pred_inst.append(u_pred_inst)
    print("Loading 100%")

    complete_time = time.time() - start_time

    MOTSA, sMOTSA, MOTSP, PTQ, sPTQ = class_evaluator_mopt.getMOPT()
    MOTSA, sMOTSA, MOTSP  = MOTSA.cpu().numpy(), sMOTSA.cpu().numpy(), MOTSP.cpu().numpy()
    PTQ, sPTQ  = PTQ.cpu().numpy(), sPTQ.cpu().numpy()
    PQ4D, AQ_mean, AQ, iou, iou_mean = class_evaluator_pq4d.getPQ4D()

    print("MOT evaluation starting.")
    valid_classes = []
    for cls in range(1,9):
        e = trackingEvaluation(test_sequences, cls)
        e.groundtruth = {'sem' : gt_sem, 'inst' : gt_inst}
        e.tracker = {'sem' : pred_sem, 'inst' : pred_inst}
        e.n_frames = []
        e.n_frames.append(len(pred_sem))
        e.compute3rdPartyMetrics()
        #dump = open('mot.txt', "w+")
        if e.n_gt > 0:
            valid_classes.append(cls-1)
        scores [cls-1] = [e.MOTA, e.MOTAL, e.MOTP, e.id_switches, e.tp, e.fp, e.fn, e.fragments]
        #print('\nMOTA MOTAL MOTP IDS TP FP FN FRAG')
        e.reset()

    output_classes = {}
    output_avg = {}

    for i in range(len(all_classes)):
        if i < len(things):
            output_classes[all_classes[i]] = {
            'MOTSA': str(MOTSA[i]), 'sMOTSA':str(sMOTSA[i]), 'MOTSP':str(MOTSP[i]), 'PTQ': str(PTQ[i+1]), 'sPTQ': str(sPTQ[i+1]),
            'AQ': str(AQ[i+1]), 'IoU': str(iou[i+1]),
            'MOTA' :str(scores[i][0]),  'MOTAL':str(scores[i][1]),  'MOTP':str(scores[i][2]), 'IDS':str(scores[i][3]),
            'TP':str(scores[i][4]), 'FP':str(scores[i][5]), 'FN':str(scores[i][6]), 'FRAG':str(scores[i][7])
            }
        else:
            output_classes[all_classes[i]] = {
                'PTQ': str(PTQ[i+1]), 'IoU': str(iou[i+1])
            }

    w_path = os.path.join(FLAGS.output, "detailed_results.json")
    with open(w_path, 'w') as outfile:
        json.dump(output_classes, outfile)

    output_avg['IoU'] = str(iou_mean)
    output_avg['AQ'] = str(AQ_mean)
    output_avg['PQ4D'] = str(PQ4D)
    output_avg['MOTSA'] = str(np.mean(MOTSA))
    output_avg['sMOTSA'] = str(np.mean(sMOTSA))
    output_avg['PTQ'] = str(np.mean(PTQ[1:]))
    output_avg['sPTQ'] = str(np.mean(sPTQ[1:]))

    w_path = os.path.join(FLAGS.output, "scores.txt")
    with open(w_path, 'w') as outfile:
        json.dump(output_avg, outfile)

    print('Done !')
