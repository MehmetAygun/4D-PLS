import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


# https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py

def weighted_mse_loss(output, target, weights):
    out = (output - target) ** 2
    if out.ndim > 1:
        out = out.sum(dim=1)
    out = out * weights.expand_as(out)
    loss = out.mean(0)
    return loss

def pdf_normal(x, mean, var):

    eps = torch.ones_like(var,requires_grad=True, device=x.device) * 1e-5
    var_eps = var + eps
    var_seq = var_eps.squeeze()
    inv_var = torch.diag(1/var_seq)
    mean_rep = mean.repeat(x.shape[0], 1)
    dif = x - mean_rep
    a = torch.matmul(dif, inv_var)
    b = torch.matmul(a, dif.T)
    p = torch.diag(b) * -0.5
    p_e = torch.exp(p)

    return p_e

def new_pdf_normal(x, mean, var):
    """
    Computes instance belonging probability values
    :param x: embeddings values of all points NxD
    :param mean: instance embedding 1XD
    :param var: instance variance value 1XD
    :return: probability scores for all points Nx1
    """
    eps = torch.ones_like(var, requires_grad=True, device=x.device) * 1e-5
    var_eps = var + eps
    var_seq = var_eps.squeeze()
    inv_var = torch.diag(1 / var_seq)
    mean_rep = mean.repeat(x.shape[0], 1)
    dif = x - mean_rep
    d = torch.pow(dif, 2)
    e = torch.matmul(d, inv_var)
    probs = torch.exp(e * -0.5)
    probs = torch.sum(probs, 1) / torch.sum(var_eps)


    return probs

def instance_half_loss(embeddings, ins_labels):
    """
    Computes l2 loss between mean embeddings of half of instances
    :param embeddings: embeddings of points
    :param ins_labels: instance labels
    :return: l2 loss
    """
    instances = torch.unique(ins_labels)
    loss = torch.tensor(0.0).to(embeddings.device)
    loss.requires_grad = True
    for instance in instances:
        if instance == 0:
            continue
        else:
            ins_idxs = torch.where(ins_labels == instance)
            ins_embeddings = embeddings[ins_idxs]
            n_points = ins_embeddings.shape[0]
            perm = torch.randperm(n_points)
            embedding_half1 = ins_embeddings[perm[0:int(n_points / 2)]]
            embedding_half2 = ins_embeddings[perm[int(n_points / 2):]]
            mean1 = torch.mean(embedding_half1, 0, True)
            mean2 = torch.mean(embedding_half2, 0, True)
            ins_half_loss = torch.nn.MSELoss()(mean1, mean2)
            if torch.isnan(ins_half_loss):
                continue
            loss = loss + ins_half_loss
    return  loss

def iou_instance_loss(centers_p, embeddings, variances, ins_labels, points=None, times=None):
    """
    Computes l2 loss between gt-prob values and predicted prob values for instances
    :param centers_p: objectness scores Nx1
    :param embeddings: embeddings  NxD
    :param variances: variances NxD
    :param ins_labels: instance ids Nx1
    :param points: xyz values Nx3
    :param times: time value normalized between 0-1 Nx1
    :return: instance loss
    """

    instances = torch.unique(ins_labels)
    loss = torch.tensor(0.0).to(embeddings.device)
    loss.requires_grad = True

    if variances.shape[1] - embeddings.shape[1] > 4:
        global_emb, _ = torch.max(embeddings, 0, keepdim=True)
        embeddings = torch.cat((embeddings, global_emb.repeat(embeddings.shape[0],1)),1)

    if variances.shape[1] - embeddings.shape[1] == 3:
        embeddings = torch.cat((embeddings, points[0]), 1)
    if variances.shape[1] - embeddings.shape[1] == 4:
        embeddings = torch.cat((embeddings, points[0], times), 1)

    for instance in instances:
        if instance == 0:
            continue
        else:
            ins_idxs = torch.where(ins_labels == instance)
            ins_centers = centers_p[ins_idxs]
            sorted, indices = torch.sort(ins_centers, 0, descending=True)
            range = torch.sum(sorted > 0.9)
            if range == 0:
                random_center = 0
            else:
                random_center = torch.randint(0, range, (1,))

            idx = ins_idxs[0][indices[random_center]]
            mean = embeddings[idx]  # 1xD
            var = variances[idx]

            labels = (ins_labels == instance) * 1.0

            probs = new_pdf_normal(embeddings, mean, var)

            ratio = torch.sum(ins_labels == 0)/(torch.sum(ins_labels == instance)*1.0+ torch.sum(probs > 0.5))
            weights = ((ins_labels == instance) | (probs >0.5)) * ratio + (ins_labels >= 0) * 1 #new loss
            loss = loss + weighted_mse_loss(probs, labels, weights)

    return loss

def variance_smoothness_loss(variances, ins_labels):
    """
    Computes smoothness loss between variance predictions
    :param variances: variances NxD
    :param ins_labels: instance ids Nx1
    :return: variance loss
    """
    instances = torch.unique(ins_labels)
    loss = torch.tensor(0.0).to(variances.device)
    loss.requires_grad = True
    if instances.size()[0] == 1:
        return torch.tensor(0)
    for instance in instances:
        if instance == 0:
            continue
        ins_idxs = torch.where(ins_labels == instance)
        ins_variances = variances[ins_idxs]
        var = torch.mean(ins_variances, dim=0)
        var_gt = var.repeat(ins_variances.shape[0], 1)
        ins_loss = torch.nn.MSELoss()(ins_variances, var_gt)

        loss = loss + ins_loss
        if torch.isnan(ins_loss):
            print("nan")
    return loss

def variance_l2_loss(variances, ins_labels):
    instances = torch.unique(ins_labels)
    loss = torch.tensor(0.0).to(variances.device)
    loss.requires_grad = True
    if instances.size()[0] == 1:
        return torch.tensor(0)

    ins_idxs = torch.where(ins_labels != 0)
    loss = torch.mean(variances[ins_idxs]**2)

    if torch.isnan(loss):
        print("nan")
    return loss

class FocalLoss(nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, output, target):
        target = target.view(-1, 1)
        target = target.long()
        logpt = F.log_softmax(output, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != output.data.type():
                self.alpha = self.alpha.type_as(output.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)  # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []
        for i in range(C):
            if i != ignore:  # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)]  # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                    for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        neg_abs = - input.abs()
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()  # foreground for class c
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = (Variable(fg) - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    if probas.dim() == 3:
        # assumes output of a sigmoid layer
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels


def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n
