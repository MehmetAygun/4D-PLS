#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Define network architectures
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#

from models.blocks import *
from models.losses import *
import numpy as np
import torch.nn as nn
import torch
from scipy.optimize import linear_sum_assignment
from utils.kalman_filter import KalmanBoxTracker


def p2p_fitting_regularizer(net):
    fitting_loss = 0
    repulsive_loss = 0

    for m in net.modules():

        if isinstance(m, KPConv) and m.deformable:

            ##############
            # Fitting loss
            ##############

            # Get the distance to closest input point and normalize to be independant from layers
            KP_min_d2 = m.min_d2 / (m.KP_extent ** 2)

            # Loss will be the square distance to closest input point. We use L1 because dist is already squared
            fitting_loss += net.l1(KP_min_d2, torch.zeros_like(KP_min_d2))

            ################
            # Repulsive loss
            ################

            # Normalized KP locations
            KP_locs = m.deformed_KP / m.KP_extent

            # Point should not be close to each other
            for i in range(net.K):
                other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()
                distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2))
                rep_loss = torch.sum(torch.clamp_max(distances - net.repulse_extent, max=0.0) ** 2, dim=1)
                repulsive_loss += net.l1(rep_loss, torch.zeros_like(rep_loss)) / net.K

    return net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)


def kalman_box_to_eight_point(kalman_bbox):

    # x, y, z, theta, l, w, h to x1,x2,y1,y2,z1,z2
    x1 = kalman_bbox[0]-kalman_bbox[4]/2
    x2 = kalman_bbox[0]+kalman_bbox[4]/2
    y1 = kalman_bbox[1]-kalman_bbox[5]/2
    y2 = kalman_bbox[1]+kalman_bbox[5]/2
    z1 = kalman_bbox[2]-kalman_bbox[6]/2
    z2 = kalman_bbox[2]+kalman_bbox[6]/2

    return [x1,y1,z1,x2,y2,z2]

def get_bbox_from_points(points):
    """
    Runs the loss on outputs of the model
    :param points: instance points Nx3
    :return: 3D bbox [x1,y1,z1,x2,y2,z2]
    """

    x1 = torch.min(points[:, 0])
    x2 = torch.max(points[:, 0])
    y1 = torch.min(points[:, 1])
    y2 = torch.max(points[:, 1])
    z1 = torch.min(points[:, 2])
    z2 = torch.max(points[:, 2])

    return [x1,y1,z1,x2,y2,z2], np.array([x1 + (x2-x1)/2, y1+ (y2-y1)/2,z1+ (z2-z1)/2, 0, x2-x1,y2-y1,z2-z1]) # x, y, z, theta, l, w, h

def get_2d_bbox(points):

    x1 = np.min(points[0, :])
    x2 = np.max(points[0, :])
    y1 = np.min(points[1, :])
    y2 = np.max(points[1, :])

    return [x1, y1, x2, y2]


def IoU(bbox0, bbox1):
    """
    Runs the intersection over union of two bbox
    :param bbox0: bbox1 list
    :param bbox1: bbox2 list

    :return: IoU
    """

    dim = int(len(bbox0)/2)
    overlap = [max(0, min(bbox0[i+dim], bbox1[i+dim]) - max(bbox0[i], bbox1[i])) for i in range(dim)]
    intersection = 1
    for i in range(dim):
        intersection = intersection * overlap[i]
    area0 = 1
    area1 = 1
    for i in range(dim):
        area0 *= (bbox0[i + dim] - bbox0[i])
        area1 *= (bbox1[i + dim] - bbox1[i])
    union = area0 + area1 - intersection
    if union == 0:
        return 0
    return intersection/union

def do_range_projection(points):
    #https: // github.com / PRBonn / semantic - kitti - api / blob / c4ef8140e21e589e6c795ec548584e13b2925b0f / auxiliary / laserscanvis.py  # L11
    proj_H = 128
    proj_W = 2048
    fov_up = 3.0
    fov_down = -25.0
    fov_up = fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad
    depth = np.linalg.norm(points, 2, axis=1)
    # get scan components
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)
    # get projections in image coords
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= proj_W                              # in [0.0, W]
    proj_y *= proj_H                              # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]


    return np.vstack((proj_x, proj_y))

def euclidean_dist(b1, b2):
    ret_sum = 0
    for i in range(3):
        ret_sum += (b1[i] - b2[i])**2
    return  torch.sqrt(ret_sum)


class KPCNN(nn.Module):
    """
    Class defining KPCNN
    """

    def __init__(self, config):
        super(KPCNN, self).__init__()

        #####################
        # Network opperations
        #####################

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points

        # Save all block operations in a list of modules
        self.block_ops = nn.ModuleList()

        # Loop over consecutive blocks
        block_in_layer = 0
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.block_ops.append(block_decider(block,
                                                r,
                                                in_dim,
                                                out_dim,
                                                layer,
                                                config))

            # Index of block in this layer
            block_in_layer += 1

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2
                block_in_layer = 0

        self.head_mlp = UnaryBlock(out_dim, 1024, False, 0)
        self.head_softmax = UnaryBlock(1024, config.num_classes, False, 0)

        ################
        # Network Losses
        ################

        self.criterion = torch.nn.CrossEntropyLoss()
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        return

    def forward(self, batch, config):

        # Save all block operations in a list of modules
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        for block_op in self.block_ops:
            x = block_op(x, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return x

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, labels)
        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    @staticmethod
    def accuracy(outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        predicted = torch.argmax(outputs.data, dim=1)
        total = labels.size(0)
        correct = (predicted == labels).sum().item()

        return correct / total


class KPFCNN(nn.Module):
    """
    Class defining KPFCNN
    """

    def __init__(self, config, lbl_values, ign_lbls):
        super(KPFCNN, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                     r,
                                                     in_dim,
                                                     out_dim,
                                                     layer,
                                                     config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_var = UnaryBlock(config.first_features_dim, out_dim + config.free_dim, False, 0)
        self.head_softmax = UnaryBlock(config.first_features_dim, self.C, False, 0)
        self.head_center = UnaryBlock(config.first_features_dim, 1, False, 0, False)


        self.pre_train = config.pre_train
        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        # Choose segmentation loss
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.center_loss = 0
        self.instance_loss = torch.tensor(0)
        self.variance_loss = torch.tensor(0)
        self.instance_half_loss = torch.tensor(0)
        self.reg_loss = 0
        self.variance_l2 = torch.tensor(0)
        self.l1 = nn.L1Loss()
        self.sigmoid = nn.Sigmoid()

        return

    def forward(self, batch, config):

        # Get input features
        x = batch.features.clone().detach()

        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)


        # Head of network
        f = self.head_mlp(x, batch)
        c = self.head_center(f, batch)
        c = self.sigmoid(c)
        v = self.head_var(f, batch)
        v = F.relu(v)
        x = self.head_softmax(f, batch)

        return x, c, v, f

    def loss(self, outputs, centers_p, variances, embeddings, labels, ins_labels, centers_gt, points=None, times=None):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.unsqueeze(0)
        centers_p = centers_p.squeeze()
        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)
        weights = (centers_gt[:, 0] > 0) * 99 + (centers_gt[:, 0] >= 0) * 1
        self.center_loss = weighted_mse_loss(centers_p, centers_gt[:, 0], weights)

        if not self.pre_train:
            self.instance_half_loss = instance_half_loss(embeddings, ins_labels)
            self.instance_loss = iou_instance_loss(centers_p, embeddings, variances, ins_labels, points, times)
            self.variance_loss = variance_smoothness_loss(variances, ins_labels)
            self.variance_l2 = variance_l2_loss(variances, ins_labels)
        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        #return self.instance_loss + self.variance_loss
        return self.output_loss + self.reg_loss + self.center_loss + self.instance_loss*0.1+ self.variance_loss*0.01

    def ins_pred(self, predicted, centers_output, var_output, embedding, points=None, times=None):
        """
        Calculate instance probabilities for each point on current frame
        :param predicted: class labels for each point
        :param centers_output: center predictions
        :param var_output : variance predictions
        :param embedding : embeddings for all points
        :param points: xyz location of points
        :return: instance ids for all points
        """
        #predicted = torch.argmax(outputs.data, dim=1)

        if var_output.shape[1] - embedding.shape[1] > 4:
            global_emb, _ = torch.max(embedding, 0, keepdim=True)
            embedding = torch.cat((embedding, global_emb.repeat(embedding.shape[0], 1)), 1)

        if  var_output.shape[1] - embedding.shape[1] == 3:
            embedding = torch.cat((embedding, points[0]), 1)
        if  var_output.shape[1] - embedding.shape[1] == 4:
            embedding = torch.cat((embedding, points[0], times), 1)

        if var_output.shape[1] == 3:
            embedding = points[0]
        if var_output.shape[1] == 4:
            embedding = torch.cat((points[0], times), 1)

        ins_prediction = torch.zeros_like(predicted)

        counter = 0
        ins_id = 1
        while True:
            ins_idxs = torch.where((predicted < 9) & (predicted != 0) & (ins_prediction == 0))
            if len(ins_idxs[0]) == 0:
                break
            ins_centers = centers_output[ins_idxs]
            ins_embeddings = embedding[ins_idxs]
            ins_variances = var_output[ins_idxs]
            if counter == 0:
                sorted, indices = torch.sort(ins_centers, 0, descending=True)  # center score of instance classes
            if sorted[0+counter] < 0.1 or (ins_id ==1 and sorted[0] < 0.7):
                break
            idx = indices[0+counter]
            mean = ins_embeddings[idx]
            var = ins_variances[idx]
            #probs = pdf_normal(ins_embeddings, mean, var)
            probs = new_pdf_normal(ins_embeddings, mean, var)

            ins_points = torch.where(probs >= 0.5)
            if ins_points[0].size()[0] < 2:
                counter +=1
                if counter == sorted.shape[0]:
                    break
                continue
            ids = ins_idxs[0][ins_points[0]]
            ins_prediction[ids] = ins_id
            counter = 0
            ins_id += 1
        return ins_prediction

    def ins_pred_in_time(self, config, predicted, centers_output, var_output, embedding, prev_instances, next_ins_id, points=None, times=None, pose=None):
        """
        Calculate instance probabilities for each point with considering old predictions also
        :param predicted: class labels for each point
        :param centers_output: center predictions
        :param var_output : variance predictions
        :param embedding : embeddings for all points
        :param prev_instances : instances which detected in previous frames
        :param next_ins_id : next avaliable ins id
        :param points: xyz location of points
        :return: instance ids for all points, and new instances and next available ins_id
        """
        new_instances = {}
        ins_prediction = torch.zeros_like(predicted)

        if var_output.shape[1] - embedding.shape[1] > 4:
            global_emb, _ = torch.max(embedding, 0, keepdim=True)
            embedding = torch.cat((embedding, global_emb.repeat(embedding.shape[0], 1)), 1)

        if var_output.shape[1] - embedding.shape[1] == 3:
            embedding = torch.cat((embedding, points[0]), 1)
        if var_output.shape[1] - embedding.shape[1] == 4:
            embedding = torch.cat((embedding, points[0], times), 1)

        pose = torch.from_numpy(pose)
        pose = pose.to(embedding.device)

        counter = 0
        ins_id = next_ins_id

        while True:
            ins_idxs = torch.where((predicted < 9) & (predicted != 0) & (ins_prediction == 0))
            if len(ins_idxs[0]) == 0:
                break
            ins_centers = centers_output[ins_idxs]
            ins_embeddings = embedding[ins_idxs]
            ins_variances = var_output[ins_idxs]
            ins_points = points[0][ins_idxs]
            if counter == 0:
                sorted, indices = torch.sort(ins_centers, 0, descending=True)  # center score of instance classes
            if sorted[0 + counter] < 0.1 or (sorted[0] < 0.7):
                break
            idx = indices[0 + counter]
            mean = ins_embeddings[idx]
            var = ins_variances[idx]

            center = points[0][ins_idxs][idx]
            distances = torch.sum((ins_points - center)**2,1)
            if torch.cuda.device_count() > 1:
                new_device = torch.device("cuda:1")
                probs = new_pdf_normal(ins_embeddings.to(new_device), mean.to(new_device), var.to(new_device))
            else:
                probs = new_pdf_normal(ins_embeddings, mean, var)

            probs[distances>20] = 0
            ins_points = torch.where(probs >= 0.5)
            if ins_points[0].size()[0] < 2:
                counter += 1
                if counter == sorted.shape[0]:
                    break
                continue

            ids = ins_idxs[0][ins_points[0]]
            ins_prediction[ids] = ins_id
            if ins_points[0].size()[0] > 25: #add to instance history
                ins_prediction[ids] = ins_id
                mean = torch.mean(embedding[ids], 0, True)
                bbox, kalman_bbox = get_bbox_from_points(points[0][ids])
                tracker = KalmanBoxTracker(kalman_bbox ,ins_id)
                bbox_proj = None
                #var = torch.mean(var_output[ids], 0, True)
                new_instances[ins_id] = {'mean': mean, 'var': var, 'life' : 5, 'bbox': bbox, 'bbox_proj':bbox_proj, 'tracker': tracker, 'kalman_bbox' : kalman_bbox}

            counter = 0
            ins_id += 1

        #associate instances by hungarian alg. & bbox prediction via kalman filter
        if len(prev_instances.keys()) > 0 :

            #association_costs, associations = self.associate_instances(config, prev_instances, new_instances, pose)
            associations = []
            for prev_id, new_id in associations:
                ins_points = torch.where((ins_prediction == new_id))
                ins_prediction[ins_points[0]] = prev_id
                prev_instances[prev_id]['mean'] = new_instances[new_id]['mean']
                prev_instances[prev_id]['bbox_proj'] = new_instances[new_id]['bbox_proj']

                prev_instances[prev_id]['life'] += 1
                prev_instances[prev_id]['tracker'].update(new_instances[new_id]['kalman_bbox'], prev_id)
                prev_instances[prev_id]['kalman_bbox'] = prev_instances[prev_id]['tracker'].get_state()
                prev_instances[prev_id]['bbox'] = kalman_box_to_eight_point(prev_instances[prev_id]['kalman_bbox'])

                del new_instances[new_id]

        return ins_prediction, new_instances, ins_id

    def associate_instances(self, config, previous_instances, current_instances, pose):
        pose = pose.cpu()
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

        p_n = len(previous_instances.keys())
        c_n = len(current_instances.keys())

        association_costs = torch.zeros(p_n, c_n)
        prev_ids = []
        current_ids = []

        for i, (k, v) in enumerate(previous_instances.items()):
            prev_ids.append(k)
            for j, (k1, v1) in enumerate(current_instances.items()):
                cost_3d = 1 - IoU(v1['bbox'], v['bbox'])
                if cost_3d > 0.75:
                    cost_3d = 1e8
                if v1['bbox_proj'] is not None:
                    cost_2d = 1 - IoU(v1['bbox_proj'], v['bbox_proj'])
                    if cost_2d > 0.5:
                        cost_2d = 1e8
                else:
                    cost_2d = 0

                cost_center = euclidean_dist(v1['kalman_bbox'], v['kalman_bbox'])
                if cost_center > 1:
                    cost_center = 1e8

                feature_cost = 1 - cos(v1['mean'], v['mean'])
                if feature_cost > 0.05:
                    feature_cost = 1e8
                costs = torch.tensor([cost_3d, cost_2d, cost_center, feature_cost])
                for idx, a_w in enumerate(config.association_weights):
                    association_costs[i, j] += a_w * costs[idx]

                if i == 0:
                    current_ids.append(k1)

        idxes_1, idxes_2 = linear_sum_assignment(association_costs.cpu().detach())

        associations = []

        for i1, i2 in zip(idxes_1, idxes_2):
            #max_cost = torch.sum((previous_instances[prev_ids[i1]]['var'][0,-3:]/2)**2)
            if association_costs[i1][i2] < 1e8:
                associations.append((prev_ids[i1], current_ids[i2]))

        return association_costs, associations


    def accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total
