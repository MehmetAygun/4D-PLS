#!/usr/bin/env python3

# This file is covered by the LICENSE file in the root of this project.
#https://github.com/PRBonn/semantic-kitti-api

import numpy as np
import math
import time


class PanopticEval:
  """ Panoptic evaluation using numpy
  
  authors: Andres Milioto and Jens Behley

  """

  def __init__(self, n_classes, device=None, ignore=None, offset=100000, min_points=30):
    self.n_classes = n_classes
    assert (device == None)
    self.ignore = np.array(ignore, dtype=np.int64)
    self.include = np.array([n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64)

    print("[PANOPTIC EVAL] IGNORE: ", self.ignore)
    print("[PANOPTIC EVAL] INCLUDE: ", self.include)

    self.reset()
    self.offset = offset  # largest number of instances in a given scan
    self.min_points = min_points  # smallest number of points to consider instances in gt
    self.eps = 1e-15

  def num_classes(self):
    return self.n_classes

  def reset(self):
    # general things
    # iou stuff
    self.px_iou_conf_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
    # panoptic stuff
    self.pan_tp = np.zeros(self.n_classes, dtype=np.int64)
    self.pan_iou = np.zeros(self.n_classes, dtype=np.double)
    self.pan_fp = np.zeros(self.n_classes, dtype=np.int64)
    self.pan_fn = np.zeros(self.n_classes, dtype=np.int64)

  ################################# IoU STUFF ##################################
  def addBatchSemIoU(self, x_sem, y_sem):
    # idxs are labels and predictions
    idxs = np.stack([x_sem, y_sem], axis=0)

    # make confusion matrix (cols = gt, rows = pred)
    np.add.at(self.px_iou_conf_matrix, tuple(idxs), 1)

  def getSemIoUStats(self):
    # clone to avoid modifying the real deal
    conf = self.px_iou_conf_matrix.copy().astype(np.double)
    # remove fp from confusion on the ignore classes predictions
    # points that were predicted of another class, but were ignore
    # (corresponds to zeroing the cols of those classes, since the predictions
    # go on the rows)
    conf[:, self.ignore] = 0

    # get the clean stats
    tp = conf.diagonal()
    fp = conf.sum(axis=1) - tp
    fn = conf.sum(axis=0) - tp
    return tp, fp, fn

  def getSemIoU(self):
    tp, fp, fn = self.getSemIoUStats()
    # print(f"tp={tp}")
    # print(f"fp={fp}")
    # print(f"fn={fn}")
    intersection = tp
    union = tp + fp + fn
    union = np.maximum(union, self.eps)
    iou = intersection.astype(np.double) / union.astype(np.double)
    iou_mean = (intersection[self.include].astype(np.double) / union[self.include].astype(np.double)).mean()
    #prec = tp / (tp+fp)
    #recall = tp / (tp+fn)
    return iou_mean, iou  # returns "iou mean", "iou per class" ALL CLASSES

  def getSemAcc(self):
    tp, fp, fn = self.getSemIoUStats()
    total_tp = tp.sum()
    total = tp[self.include].sum() + fp[self.include].sum()
    total = np.maximum(total, self.eps)
    acc_mean = total_tp.astype(np.double) / total.astype(np.double)

    return acc_mean  # returns "acc mean"

  ################################# IoU STUFF ##################################
  ##############################################################################

  #############################  Panoptic STUFF ################################
  def addBatchPanoptic(self, x_sem_row, x_inst_row, y_sem_row, y_inst_row):
    # make sure instances are not zeros (it messes with my approach)
    x_inst_row = x_inst_row + 1
    y_inst_row = y_inst_row + 1

    # only interested in points that are outside the void area (not in excluded classes)
    for cl in self.ignore:
      # make a mask for this class
      gt_not_in_excl_mask = y_sem_row != cl
      # remove all other points
      x_sem_row = x_sem_row[gt_not_in_excl_mask]
      y_sem_row = y_sem_row[gt_not_in_excl_mask]
      x_inst_row = x_inst_row[gt_not_in_excl_mask]
      y_inst_row = y_inst_row[gt_not_in_excl_mask]

    # first step is to count intersections > 0.5 IoU for each class (except the ignored ones)
    for cl in self.include:
      # print("*"*80)
      # print("CLASS", cl.item())
      # get a class mask
      x_inst_in_cl_mask = x_sem_row == cl
      y_inst_in_cl_mask = y_sem_row == cl

      # get instance points in class (makes outside stuff 0)
      x_inst_in_cl = x_inst_row * x_inst_in_cl_mask.astype(np.int64)
      y_inst_in_cl = y_inst_row * y_inst_in_cl_mask.astype(np.int64)

      # generate the areas for each unique instance prediction
      unique_pred, counts_pred = np.unique(x_inst_in_cl[x_inst_in_cl > 0], return_counts=True)
      id2idx_pred = {id: idx for idx, id in enumerate(unique_pred)}
      matched_pred = np.array([False] * unique_pred.shape[0])
      # print("Unique predictions:", unique_pred)

      # generate the areas for each unique instance gt_np
      unique_gt, counts_gt = np.unique(y_inst_in_cl[y_inst_in_cl > 0], return_counts=True)
      id2idx_gt = {id: idx for idx, id in enumerate(unique_gt)}
      matched_gt = np.array([False] * unique_gt.shape[0])
      # print("Unique ground truth:", unique_gt)

      # generate intersection using offset
      valid_combos = np.logical_and(x_inst_in_cl > 0, y_inst_in_cl > 0)
      offset_combo = x_inst_in_cl[valid_combos] + self.offset * y_inst_in_cl[valid_combos]
      unique_combo, counts_combo = np.unique(offset_combo, return_counts=True)

      # generate an intersection map
      # count the intersections with over 0.5 IoU as TP
      gt_labels = unique_combo // self.offset
      pred_labels = unique_combo % self.offset
      gt_areas = np.array([counts_gt[id2idx_gt[id]] for id in gt_labels])
      pred_areas = np.array([counts_pred[id2idx_pred[id]] for id in pred_labels])
      intersections = counts_combo
      unions = gt_areas + pred_areas - intersections
      ious = intersections.astype(np.float) / unions.astype(np.float)


      tp_indexes = ious > 0.5
      self.pan_tp[cl] += np.sum(tp_indexes)
      self.pan_iou[cl] += np.sum(ious[tp_indexes])

      matched_gt[[id2idx_gt[id] for id in gt_labels[tp_indexes]]] = True
      matched_pred[[id2idx_pred[id] for id in pred_labels[tp_indexes]]] = True

      # count the FN
      self.pan_fn[cl] += np.sum(np.logical_and(counts_gt >= self.min_points, matched_gt == False))

      # count the FP
      self.pan_fp[cl] += np.sum(np.logical_and(counts_pred >= self.min_points, matched_pred == False))

  def getPQ(self):
    # first calculate for all classes
    sq_all = self.pan_iou.astype(np.double) / np.maximum(self.pan_tp.astype(np.double), self.eps)
    rq_all = self.pan_tp.astype(np.double) / np.maximum(
        self.pan_tp.astype(np.double) + 0.5 * self.pan_fp.astype(np.double) + 0.5 * self.pan_fn.astype(np.double),
        self.eps)
    pq_all = sq_all * rq_all

    # then do the REAL mean (no ignored classes)
    SQ = sq_all[self.include].mean()
    RQ = rq_all[self.include].mean()
    PQ = pq_all[self.include].mean()

    return PQ, SQ, RQ, pq_all, sq_all, rq_all

  #############################  Panoptic STUFF ################################
  ##############################################################################

  def addBatch(self, x_sem, x_inst, y_sem, y_inst):  # x=preds, y=targets
    ''' IMPORTANT: Inputs must be batched. Either [N,H,W], or [N, P]
    '''
    # add to IoU calculation (for checking purposes)
    self.addBatchSemIoU(x_sem, y_sem)

    # now do the panoptic stuff
    self.addBatchPanoptic(x_sem, x_inst, y_sem, y_inst)


class Panoptic4DEval:
  """ Panoptic evaluation using numpy

  authors: Andres Milioto, Jens Behley, Aljosa Osep

  """

  def __init__(self, n_classes, device=None, ignore=None, offset=2 ** 32, min_points=30):
    self.n_classes = n_classes
    assert (device == None)
    self.ignore = np.array(ignore, dtype=np.int64)
    self.include = np.array([n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64)
    print("[PANOPTIC4D EVAL] IGNORE: ", self.ignore)
    print("[PANOPTIC4D EVAL] INCLUDE: ", self.include)

    self.reset()
    self.offset = offset  # largest number of instances in a given scan
    self.min_points = min_points  # smallest number of points to consider instances in gt
    self.eps = 1e-15

  def num_classes(self):
    return self.n_classes

  def reset(self):
    # general things
    # iou stuff
    self.px_iou_conf_matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

    self.sequences = []
    self.preds = {}
    self.gts = {}
    self.intersects = {}
    self.intersects_ovr = {}

    # Per-class association quality collect here
    self.pan_aq = np.zeros(self.n_classes, dtype=np.double)
    self.pan_aq_ovr = 0.0
  ################################# IoU STUFF ##################################
  def addBatchSemIoU(self, x_sem, y_sem):
    # idxs are labels and predictions
    idxs = np.stack([x_sem, y_sem], axis=0)

    # make confusion matrix (cols = gt, rows = pred)
    np.add.at(self.px_iou_conf_matrix, tuple(idxs), 1)

  def getSemIoUStats(self):
    # clone to avoid modifying the real deal
    conf = self.px_iou_conf_matrix.copy().astype(np.double)
    # remove fp from confusion on the ignore classes predictions
    # points that were predicted of another class, but were ignore
    # (corresponds to zeroing the cols of those classes, since the predictions
    # go on the rows)
    conf[:, self.ignore] = 0

    # get the clean stats
    tp = conf.diagonal()
    fp = conf.sum(axis=1) - tp
    fn = conf.sum(axis=0) - tp
    return tp, fp, fn

  def getSemIoU(self):
    tp, fp, fn = self.getSemIoUStats()
    # print(f"tp={tp}")
    # print(f"fp={fp}")
    # print(f"fn={fn}")
    intersection = tp
    union = tp + fp + fn

    num_present_classes = np.count_nonzero(union)

    union = np.maximum(union, self.eps)
    iou = intersection.astype(np.double) / union.astype(np.double)
    iou_mean = np.sum(iou) / num_present_classes

    prec = tp / np.maximum(tp+fp, self.eps)
    recall = tp / np.maximum(tp+fn, self.eps)

    return iou_mean, iou, np.sum(prec)/num_present_classes, np.sum(recall)/num_present_classes  # returns "iou mean", "iou per class" ALL CLASSES

  def getSemAcc(self):
    tp, fp, fn = self.getSemIoUStats()
    total_tp = tp.sum()
    total = tp[self.include].sum() + fp[self.include].sum()
    total = np.maximum(total, self.eps)
    acc_mean = total_tp.astype(np.double) / total.astype(np.double)

    return acc_mean  # returns "acc mean"

  ################################# IoU STUFF ##################################
  ##############################################################################

  #############################  Panoptic STUFF ################################

  def update_dict_stat(self, stat_dict, unique_ids, unique_cnts):
    for uniqueid, counts in zip(unique_ids, unique_cnts):
      if uniqueid == 1: continue # 1 -- no instance
      if uniqueid in stat_dict:
        stat_dict[uniqueid] += counts
      else:
        stat_dict[uniqueid] = counts

  def addBatchPanoptic4D(self, seq, x_sem_row, x_inst_row, y_sem_row, y_inst_row):

    #start = time.time()
    if seq not in self.sequences:
      self.sequences.append(seq)
      self.preds[seq] = {}
      self.gts[seq] = [{} for i in range(self.n_classes)]
      self.intersects[seq] = [{} for i in range(self.n_classes)]
      self.intersects_ovr[seq] = [{} for i in range(self.n_classes)]

    # make sure instances are not zeros (it messes with my approach)
    x_inst_row = x_inst_row + 1
    y_inst_row = y_inst_row + 1

    # only interested in points that are outside the void area (not in excluded classes)
    for cl in self.ignore:
      # make a mask for this class
      gt_not_in_excl_mask = y_sem_row != cl
      # remove all other points
      x_sem_row = x_sem_row[gt_not_in_excl_mask]
      y_sem_row = y_sem_row[gt_not_in_excl_mask]
      x_inst_row = x_inst_row[gt_not_in_excl_mask]
      y_inst_row = y_inst_row[gt_not_in_excl_mask]

    for cl in self.include:
      # print("*"*80)
      # print("CLASS", cl.item())

      # Per-class accumulated stats
      cl_preds = self.preds[seq]
      cl_gts = self.gts[seq][cl]
      cl_intersects = self.intersects[seq][cl]

      # get a binary class mask (filter acc. to semantic class!)
      x_inst_in_cl_mask = x_sem_row == cl
      y_inst_in_cl_mask = y_sem_row == cl

      # get instance points in class (mask-out everything but _this_ class)
      x_inst_in_cl = x_inst_row * x_inst_in_cl_mask.astype(np.int64)
      y_inst_in_cl = y_inst_row * y_inst_in_cl_mask.astype(np.int64)

      # generate the areas for each unique instance gt_np (i.e., set2)
      unique_gt, counts_gt = np.unique(y_inst_in_cl[y_inst_in_cl > 0], return_counts=True)
      self.update_dict_stat(cl_gts, unique_gt[counts_gt>self.min_points], counts_gt[counts_gt>self.min_points])

      valid_combos_min_point = np.zeros_like(y_inst_in_cl)  # instances which have more than self.min points
      for valid_id in unique_gt[counts_gt > self.min_points]:
        valid_combos_min_point = np.logical_or(valid_combos_min_point, y_inst_in_cl == valid_id)

      y_inst_in_cl = y_inst_in_cl * valid_combos_min_point
      # generate the areas for each unique instance prediction (i.e., set1)
      unique_pred, counts_pred = np.unique(x_inst_in_cl[x_inst_in_cl > 0], return_counts=True)

      # is there better way to do this?
      self.update_dict_stat(cl_preds, unique_pred, counts_pred)

      valid_combos = np.logical_and(x_inst_row > 0,
                                    y_inst_in_cl > 0)  # Convert to boolean and do logical and, based on semantics

      # generate intersection using offset
      offset_combo = x_inst_row[valid_combos] + self.offset * y_inst_in_cl[valid_combos]
      unique_combo, counts_combo = np.unique(offset_combo, return_counts=True)

      self.update_dict_stat(cl_intersects, unique_combo, counts_combo)


  def getPQ4D(self):
    precs = []
    recalls = []
    num_tubes = [0] * self.n_classes
    for seq in self.sequences:
      for cl in self.include:
        cl_preds = self.preds[seq]
        cl_gts = self.gts[seq][cl]
        cl_intersects = self.intersects[seq][cl]
        outer_sum_iou = 0.0
        num_tubes[cl] += len(cl_gts)
        for gt_id, gt_size in cl_gts.items():
          inner_sum_iou = 0.0
          for pr_id, pr_size in cl_preds.items():
            # TODO: pay attention for zero intersection!
            TPA_key =  pr_id + self.offset * gt_id
            if TPA_key in cl_intersects:
              TPA = cl_intersects[TPA_key]
              Prec = TPA / float(pr_size) # TODO I dont think these can ever be zero, but double check
              Recall = TPA / float(gt_size)
              precs.append(Prec)
              recalls.append(Recall)
              TPA_ovr = self.intersects[seq][cl][TPA_key]
              inner_sum_iou += TPA_ovr * (TPA_ovr / (gt_size + pr_size - TPA_ovr))
              if Prec > 1.0 or Recall >1.0:
                print ('something wrong !!')
          outer_sum_iou += 1.0 / float(gt_size) * float(inner_sum_iou)
        self.pan_aq[cl] += outer_sum_iou # 1.0/float(len(cl_gts)) # Normalize by #tubes
        self.pan_aq_ovr += outer_sum_iou
    # ==========

    #print ('num tubes:', len(list(cl_preds.items())))
    AQ_overall = np.sum(self.pan_aq_ovr)/ np.sum(num_tubes[1:9])
    AQ = self.pan_aq / np.maximum(num_tubes, self.eps)

    iou_mean, iou, iou_p, iou_r = self.getSemIoU()

    AQ_p = np.mean(precs)
    AQ_r = np.mean(recalls)

    PQ4D =  math.sqrt(AQ_overall*iou_mean)
    return PQ4D, AQ_overall, AQ, AQ_p, AQ_r,  iou, iou_mean, iou_p, iou_r


  #############################  Panoptic STUFF ################################
  ##############################################################################

  def addBatch(self, seq, x_sem, x_inst, y_sem, y_inst):  # x=preds, y=targets
    ''' IMPORTANT: Inputs must be batched. Either [N,H,W], or [N, P]
    '''
    # add to IoU calculation (for checking purposes)
    self.addBatchSemIoU(x_sem, y_sem)

    # now do the panoptic stuff
    self.addBatchPanoptic4D(seq, x_sem, x_inst, y_sem, y_inst)


if __name__ == "__main__":
  classes = 3  # ignore, car, truck
  cl_strings = ["ignore", "car", "truck"]
  ignore = [0]  # only ignore ignore class

  sem_gt = np.zeros(20, dtype=np.int32)
  sem_gt[5:10] = 1
  sem_gt[10:] = 2

  inst_gt = np.zeros(20, dtype=np.int32)
  inst_gt[5:10] = 1
  inst_gt[10:] = 1
  inst_gt[15:] = 2

  #we have 3 instance 1 car, 2 truck as gt
  sem_pred = np.zeros(20, dtype=np.int32)
  sem_pred[5:10] = 1
  sem_pred[10:15] = 2
  sem_pred[15:] = 1

  inst_pred = np.zeros(20, dtype=np.int32)
  inst_pred[5:10] = 1
  inst_pred[10:] = 2

  # evaluator
  class_evaluator = Panoptic4DEval(3, None, ignore, offset = 2 ** 32, min_points=1)
  class_evaluator.addBatch(1, sem_pred, inst_pred, sem_gt, inst_gt)
  PQ4D, AQ_ovr, AQ, AQ_p, AQ_r, iou, iou_mean, iou_p, iou_r = class_evaluator.getPQ4D()
  np.testing.assert_equal(PQ4D, np.sqrt(1.0/3))
  np.testing.assert_equal(AQ_ovr, 2.0/3)
  np.testing.assert_equal(AQ, [0, 1.0, 0.5])
  np.testing.assert_equal(AQ_p, 2.0/3)
  np.testing.assert_equal(AQ_r, 1.0)
  np.testing.assert_equal(iou, [0, 0.5, 0.5])
  np.testing.assert_equal(iou_mean, 0.5)
