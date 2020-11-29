import numpy as np
import yaml
import os
import glob
import concurrent.futures

MAX_PROCESSOR = 8
executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PROCESSOR)
jobs = []

log_dir = 'Log_2020-10-06_16-51-05_importance_None_str1_bigpug_2_xyz'
config_file = '../data/SemanticKitti/semantic-kitti.yaml'
prediction_path = '../test/{}/val_probs'.format(log_dir)
save_path = '../test/{}/binary'.format(log_dir)
data_dir = '../data/SemanticKitti/'

on_val = True

if not os.path.exists(save_path):
    os.makedirs(save_path)

if not os.path.exists(save_path+'/sequences'):
    os.makedirs(save_path+'/sequences')

save_path = save_path + '/sequences'

def write_pred(prediction_path, save_path, sequence, scene, inv_learning_map):

    sem_preds = np.load('{}/{:02d}_{:07d}.npy'.format(prediction_path, sequence, scene))
    ins_preds = np.load('{}/{:02d}_{:07d}_i.npy'.format(prediction_path, sequence, scene))

    ins_preds = ins_preds.astype(np.int32)

    for idx, semins in enumerate(np.unique(sem_preds)):
        if semins < 1 or semins > 8:
            valid_ind = np.argwhere((sem_preds == semins) & (ins_preds == 0))[:, 0]
            ins_preds[valid_ind] = semins

    for idx, semins in enumerate(np.unique(ins_preds)):

        valid_ind = np.argwhere(ins_preds == semins)[:, 0]
        ins_preds[valid_ind] = 20 + semins
        if valid_ind.shape[0] < 25:
            ins_preds[valid_ind] = 0


    new_preds = np.left_shift(ins_preds, 16)
    inv_sem_labels = inv_learning_map[sem_preds]
    new_preds = np.bitwise_or(new_preds, inv_sem_labels)

    new_preds.tofile('{}/{:02d}/predictions/{:06d}.label'.format(save_path, sequence, scene))
    return True


with open(config_file, 'r') as stream:
    doc = yaml.safe_load(stream)
    learning_map_doc = doc['learning_map']
    inv_learning_map_doc = doc['learning_map_inv']

inv_learning_map = np.zeros((np.max([k for k in inv_learning_map_doc.keys()]) + 1), dtype=np.int32)
for k, v in inv_learning_map_doc.items():
    inv_learning_map[k] = v

sequences = [8]
if not on_val:
    sequences = [11,12,13,14,15,16,17,18,19,20,21]

for sequence in sequences:
    if not os.path.exists('{}/{:02d}'.format(save_path, sequence)):
        os.makedirs('{}/{:02d}'.format(save_path, sequence))
    if not os.path.exists('{}/{:02d}/predictions'.format(save_path, sequence)):
        os.makedirs('{}/{:02d}/predictions'.format(save_path, sequence))

    n_scenes = len(glob.glob('{}/sequences/{:02d}/velodyne/*.bin'.format(data_dir, sequence)))
    print ('Processing scene {}'.format(sequence))
    for scene in range(0, n_scenes):
        if not os.path.exists('{}/{:02d}_{:07d}.npy'.format(prediction_path, sequence, scene)):
            print('Scene : {} is missing'.format(scene))
            print ('{}/{:02d}_{:07d}.npy'.format(prediction_path, sequence, scene))
            continue
        #write_pred(prediction_path, save_path, sequence, scene, inv_learning_map)
        jobs.append(executor.submit(write_pred, prediction_path, save_path, sequence, scene, inv_learning_map))

    completed = 0
    for future in concurrent.futures.as_completed(jobs):
        completed+=1
        ret = future.result()
        print('{}/{}'.format(completed, n_scenes), end="\r")

executor.shutdown(wait=True)
