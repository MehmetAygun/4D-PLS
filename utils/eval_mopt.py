import numpy as np
import torch
from collections import defaultdict

class MOPTEval:

    def __init__(self, n_classes, device=None, ignore=None, offset=2 ** 32, min_points=30):
        self.n_classes = n_classes
        assert (device == None)
        self.ignore = np.array(ignore, dtype=np.int64)
        self.include = np.array([n for n in range(self.n_classes) if n not in self.ignore], dtype=np.int64)

        #print("[MOPT EVAL] IGNORE: ", self.ignore)
        #print("[MOPT EVAL] INCLUDE: ", self.include)

        self.offset = offset  # largest number of instances in a given scan
        self.min_points = min_points  # smallest number of points to consider instances in gt
        self.eps = 1e-15

        self.panoptic_buffer = torch.zeros(4, self.n_classes)
        self.seq_trajectories = defaultdict(list)
        self.class_trajectories = defaultdict(list)
        self.iou_trajectories = defaultdict(list)

        self.num_stuff = 8
        self.count = 0
    def num_classes(self):
        return self.n_classes

    def addBatchMOPT(self, x_sem_row, x_inst_row, y_sem_row, y_inst_row):

        x_inst_row = x_inst_row + 1
        y_inst_row = y_inst_row + 1

        # first step is to count intersections > 0.5 IoU for each class (except the ignored ones)
        msk_gt, cat_gt, track_gt = get_processing_format_gt(y_sem_row, y_inst_row, self.offset, self.ignore)
        msk_pred, cat_pred, track_pred = get_processing_format_pred(x_sem_row, x_inst_row, self.offset, self.ignore)

        iou, tp, fp, fn, self.seq_trajectories, self.class_trajectories, self.iou_trajectories = panoptic_compute(msk_gt, cat_gt,
                                                                                                   track_gt,
                                                                                                   self.seq_trajectories,
                                                                                                   self.class_trajectories,
                                                                                                   self.iou_trajectories,
                                                                                                   msk_pred, cat_pred,
                                                                                                   track_pred,
                                                                                                   self.n_classes,
                                                                                                   self.num_stuff)
        self.panoptic_buffer += torch.stack((iou, tp, fp, fn), dim=0)
        self.count += 1

    #############################  MOPT STUFF ################################
    ##############################################################################

    def addBatch(self, x_sem, x_inst, y_sem, y_inst):  # x=preds, y=targets
        self.addBatchMOPT(x_sem, x_inst, y_sem, y_inst)

    def getMOPT(self):

        MOTSA, sMOTSA, MOTSP, PTQ, sPTQ, IDS = get_MOTSP_metrics(self.panoptic_buffer, self.num_stuff, self.seq_trajectories,
                                                            self.class_trajectories, self.iou_trajectories)

        self.TP = self.panoptic_buffer[1][1:9]
        self.FP = self.panoptic_buffer[2][1:9]
        self.FN = self.panoptic_buffer[3][1:9]
        self.IDS = IDS[1:9]
        print_results({'MOTSA': MOTSA, 'sMOTSA': sMOTSA, 'MOTSP': MOTSP, 'PTQ': PTQ, 'sPTQ': sPTQ}, self.num_stuff)

        return MOTSA, sMOTSA, MOTSP, PTQ, sPTQ

def panoptic_compute(msk_gt, cat_gt, track_gt, seq_trajectories, class_trajectories, iou_trajectories, msk_pred,
                     cat_pred, track_pred, num_classes, _num_stuff):
    cat_gt = torch.from_numpy(cat_gt).long()
    msk_gt = torch.from_numpy(msk_gt).long()
    track_gt = torch.from_numpy(track_gt).long()

    for cat_id, track_id in zip(cat_gt, track_gt):
        if track_id != 0:
            seq_trajectories[int(track_id.numpy())].append(-1)
            iou_trajectories[int(track_id.numpy())].append(-1)
            class_trajectories[int(track_id.numpy())].append(cat_id)

    msk_pred = torch.from_numpy(msk_pred).long()
    cat_pred = torch.from_numpy(cat_pred).long()
    track_pred = torch.from_numpy(track_pred).long()

    iou = msk_pred.new_zeros(num_classes, dtype=torch.double)
    tp = msk_pred.new_zeros(num_classes, dtype=torch.double)
    fp = msk_pred.new_zeros(num_classes, dtype=torch.double)
    fn = msk_pred.new_zeros(num_classes, dtype=torch.double)

    if cat_gt.numel() > 1:
        msk_gt = msk_gt.view(-1)
        msk_pred = msk_pred.view(-1)

        confmat = msk_pred.new_zeros(cat_gt.numel(), cat_pred.numel(), dtype=torch.double)

        confmat.view(-1).index_add_(0, msk_gt * cat_pred.numel() + msk_pred,
                                    confmat.new_ones(msk_gt.numel()))
        num_pred_pixels = confmat.sum(0)

        valid_fp = (confmat[0] / num_pred_pixels) <= 0.5

        # compute IoU without counting void pixels (both in gt and pred)
        _iou = confmat / ((num_pred_pixels - confmat[0]).unsqueeze(0) + confmat.sum(1).unsqueeze(1) - confmat)

        # flag TP matches, i.e. same class and iou > 0.5
        matches = ((cat_gt.unsqueeze(1) == cat_pred.unsqueeze(0)) & (_iou > 0.5))
        # remove potential match of void_gt against void_pred
        matches[0, 0] = 0

        _iou = _iou[matches]
        tp_i, _ = matches.max(1)
        fn_i = ~tp_i
        fn_i[0] = 0  # remove potential fn match due to void against void
        fp_i = ~matches.max(0)[0] & valid_fp
        fp_i[0] = 0  # remove potential fp match due to void against void

        # Compute per instance classes for each tp, fp, fn
        tp_cat = cat_gt[tp_i]
        fn_cat = cat_gt[fn_i]
        fp_cat = cat_pred[fp_i]

        match_ind = torch.nonzero(matches)
        for r in range(match_ind.shape[0]):
            if track_gt[match_ind[r, 0]] != 0 and track_gt[match_ind[r, 0]] >= _num_stuff:
                seq_trajectories[int(track_gt[match_ind[r, 0]].numpy())][-1] = int(
                    track_pred[match_ind[r, 1]].cpu().numpy())
                iou_trajectories[int(track_gt[match_ind[r, 0]].numpy())][-1] = float(_iou[r].cpu().numpy())
        if tp_cat.numel() > 0:
            tp.index_add_(0, tp_cat, tp.new_ones(tp_cat.numel()))
        if fp_cat.numel() > 0:
            fp.index_add_(0, fp_cat, fp.new_ones(fp_cat.numel()))
        if fn_cat.numel() > 0:
            fn.index_add_(0, fn_cat, fn.new_ones(fn_cat.numel()))
        if tp_cat.numel() > 0:
            iou.index_add_(0, tp_cat, _iou)

    return iou, tp, fp, fn, seq_trajectories, class_trajectories, iou_trajectories

def get_MOTSP_metrics(panoptic_buffer, num_stuff, seq_trajectories, class_trajectories, iou_trajectories):
    size = panoptic_buffer[0].shape[0]
    IDS, softIDS = compute_ids(seq_trajectories, class_trajectories, iou_trajectories, panoptic_buffer[0].shape[0])


    #MOTSA = (torch.max(panoptic_buffer[1][1:1+num_stuff] - panoptic_buffer[2][1:1+num_stuff] - IDS[1:1+num_stuff],
    #                   torch.zeros((num_stuff,), dtype=torch.double))) / (
    #                panoptic_buffer[1][1:1+num_stuff] + panoptic_buffer[3][1:1+num_stuff] + 1e-8)

    MOTSA = (panoptic_buffer[1][1:1 + num_stuff] - panoptic_buffer[2][1:1 + num_stuff] - IDS[1:1 + num_stuff]) / \
            (   panoptic_buffer[1][1:1 + num_stuff] + panoptic_buffer[3][1:1 + num_stuff] + 1e-8)


    #sMOTSA = (torch.max(panoptic_buffer[0][1:1+num_stuff] - panoptic_buffer[2][1:1+num_stuff] - IDS[1:1+num_stuff],
    #                    torch.zeros((num_stuff,), dtype=torch.double))) / (
    #                 panoptic_buffer[1][1:1+num_stuff] + panoptic_buffer[3][1:1+num_stuff] + 1e-8)

    sMOTSA = (panoptic_buffer[0][1:1 + num_stuff] - panoptic_buffer[2][1:1 + num_stuff] - IDS[1:1 + num_stuff]) / \
             (panoptic_buffer[1][1:1 + num_stuff] + panoptic_buffer[3][1:1 + num_stuff] + 1e-8)


    MOTSP = (panoptic_buffer[0][1:1+num_stuff]) / (panoptic_buffer[1][1:1+num_stuff] + 1e-8)

    denom = panoptic_buffer[1] + 0.5 * (panoptic_buffer[2] + panoptic_buffer[3])
    denom[denom == 0] = 1.

    PTQ = (panoptic_buffer[0] - IDS) / denom
    sPTQ = (panoptic_buffer[0] - softIDS) / denom

    return MOTSA, sMOTSA, MOTSP, PTQ, sPTQ, IDS


def compute_ids(seq_trajectories, class_trajectories, iou_trajectories, size):
    id_switches = torch.zeros((size,), dtype=torch.double)
    soft_id_switches = torch.zeros((size,), dtype=torch.double)
    id_fragments = 0  # no use for now

    if len(seq_trajectories) != 0:
        for g, cl, iou in zip(seq_trajectories.values(), class_trajectories.values(), iou_trajectories.values()):
            # all frames of this gt trajectory are not assigned to any detections
            if all([this == -1 for this in g]):
                continue
            # compute tracked frames in trajectory
            last_id = g[0]
            # first detection (necessary to be in gt_trajectories) is always tracked
            tracked = 1 if g[0] >= 0 else 0
            for f in range(1, len(g)):
                if last_id != g[f] and last_id != -1 and g[f] != -1:
                    id_switches[cl[f]] += 1
                    soft_id_switches[cl[f]] += iou[f]
                if f < len(g) - 1 and g[f - 1] != g[f] and last_id != -1 and g[f] != -1 and g[f + 1] != -1:
                    id_fragments += 1
                if g[f] != -1:
                    tracked += 1
                    last_id = g[f]
            # handle last frame; tracked state is handled in for loop (g[f]!=-1)
            if len(g) > 1 and g[f - 1] != g[f] and last_id != -1 and g[f] != -1:
                id_fragments += 1
    else:
        print('something is wrong')
    return id_switches, soft_id_switches

def get_processing_format_gt(x_sem, x_inst, offset, ignore_classes):

    msk = np.zeros(x_sem.shape[0], np.int32)
    cat = [255]
    track_id = [0]
    ids = np.unique(x_sem)
    for id_i in ids:
        if id_i in ignore_classes:
            continue
        elif id_i > 8:
            cls_i = id_i
            iss_instance_id = len(cat)
            mask_i = x_sem==id_i
            cat.append(cls_i)
            track_id.append(0)
            msk[mask_i] = iss_instance_id
        else:
            t_ids = np.unique(x_inst[x_sem==id_i])
            for t_i in t_ids:
                cls_i = id_i
                iss_instance_id = len(cat)
                mask_i = (x_inst == t_i) & (x_sem==id_i)
                if np.sum(mask_i) < 50: #min points to tra
                    continue
                cat.append(cls_i)
                track_id.append(t_i + offset+ cls_i)
                msk[mask_i] = iss_instance_id

    return msk, np.array(cat), np.array(track_id)

def get_processing_format_pred(x_sem, x_inst, offset, ignore_classes):

    msk = np.zeros(x_sem.shape[0], np.int32)
    cat = [255]
    track_id = [0]
    ids = np.unique(x_sem)
    for id_i in ids:
        if id_i in ignore_classes:
            continue
        if id_i > 8:
            cls_i = id_i
            iss_instance_id = len(cat)
            mask_i = x_sem==id_i
            cat.append(cls_i)
            track_id.append(0)
            msk[mask_i] = iss_instance_id

    ids = np.unique(x_inst[x_inst>20])
    for id in ids:
        sem_cls, sem_count = np.unique(x_sem[x_inst == id], return_counts=True)
        cls_i = sem_cls[np.argmax(sem_count)]
        iss_instance_id = len(cat)
        mask_i = (x_inst == id) #& (x_sem==id_i)
        if np.sum(mask_i) < 50 : #min points to track
            continue
        cat.append(cls_i)
        track_id.append(id + offset+ cls_i)
        msk[mask_i] = iss_instance_id

    return msk, np.array(cat), np.array(track_id)


def print_results(metrics, num_stuff):
    json_save = {}
    col = '|'
    space = ' '
    line = (space * 2) + 'metric' + (space * 2) + col + (space * 2) + 'stuff' + (space * 2) + col + (
                space * 2) + 'thing' + (space * 2) + col + (space * 2) + 'all' + (space * 2) + col
    print()
    print(line)
    print('-' * 39)
    for metric in metrics:
        stuff = '----'
        thing = str(round(metrics[metric].sum().item()/ (num_stuff)  * 100, 2))
        all_ = thing
        if metric in ['PTQ', 'sPTQ']:
            thing = str(round((metrics[metric][1:1+num_stuff].sum().item()/(num_stuff+ 1e-8))* 100, 2))
            stuff = str(round((metrics[metric][1+num_stuff:].sum().item()/(19-num_stuff+ 1e-8)) * 100,2 ))
            all_ = str(round((metrics[metric][1:].sum().item()/(19+ 1e-8)) * 100,2))
        json_save[metric] = [stuff, thing, all_]
        line = (space * 2) + metric + (space * (8 - len(metric))) + col + (space * 2) + stuff + (
                    space * (7 - len(stuff))) + col + (space * 2) + thing + (space * (7 - len(thing))) + col + (
                           space * 2) + all_ + (space * (5 - len(all_))) + col
        print(line)

    #with open('results.json', 'w') as write_json:
    #    json.dump(json_save, write_json)