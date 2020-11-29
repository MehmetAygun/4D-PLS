#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
#https://github.com/PRBonn/semantic-kitti-ap
import argparse
import os
import yaml
import sys
import numpy as np
import time
import json

from eval_np import Panoptic4DEval


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

  print("Ignoring classes: ", ignore_class)

  # get test set
  test_sequences = DATA["split"][FLAGS.split]

  # create evaluator
  class_evaluator = Panoptic4DEval(nr_classes, None, ignore_class, offset = 2 ** 32, min_points=FLAGS.min_inst_points)

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
  # print(pred_names)

  # check that I have the same number of files
  assert (len(label_names) == len(pred_names))
  print("Evaluating sequences: ", end="", flush=True)
  # open each file, get the tensor, and make the iou comparison

  complete = len(label_names)

  count = 0
  percent = 10
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

    class_evaluator.addBatch(label_file.split('/')[-3], u_pred_sem_class, u_pred_inst, u_label_sem_class, u_label_inst)

  print("100%")

  complete_time = time.time() - start_time
  PQ4D, AQ_ovr, AQ, AQ_p, AQ_r,  iou, iou_mean, iou_p, iou_r = class_evaluator.getPQ4D()
  things_iou = iou[1:9].mean()
  stuff_iou = iou[9:].mean()
  print ("=== Results ===")
  print ("PQ4D:", PQ4D)
  print("AQ_overall:", AQ_ovr)
  float_formatter = "{:.2f}".format
  np.set_printoptions(formatter={'float_kind': float_formatter})
  print ("AQ:", AQ)
  print ("iou:", iou)
  print("things_iou:", things_iou)
  print("stuff_iou:", stuff_iou)

  print ("iou_mean:", iou_mean)

