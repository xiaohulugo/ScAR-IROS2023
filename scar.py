import _init_path
import argparse
import datetime
import glob
import os
import re
import time
import copy
import pickle
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from eval_utils import eval_utils
from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from visual_utils import visualize_utils
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

import gco
from scipy.optimize import linear_sum_assignment
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint to start from')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=30, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--eval_tag', type=str, default='default', help='eval tag for this experiment')
    parser.add_argument('--eval_all', action='store_true', default=False, help='whether to evaluate all checkpoints')
    parser.add_argument('--ckpt_dir', type=str, default=None, help='specify a ckpt directory to be evaluated if needed')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()
    #args.cfg_file = '/home/lxh/Documents/Code/Detection3D/ScAR/tools/cfgs/kitti_models/second.yaml' 
    #args.cfg_file = '/home/lxh/Documents/Code/Detection3D/ScAR/tools/cfgs/waymo_models/second.yaml' 
    args.cfg_file = '/home/lxh/Documents/Code/Detection3D/ScAR/tools/cfgs/nuscenes_models/second.yaml' 
    args.batch_size = 1
    args.workers = 0
    args.eval_all = False

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    #cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    cfg.EXP_GROUP_PATH = ''
    np.random.seed(1024)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg



def scaling_uniform(data_loader, args, logger):    
    # get gt
    output_dir = '/media/lxh/Data/ScAR'

    if False:
        box_list = []
        gt = {}
        dataloader_iter = iter(data_loader)
        tt = len(data_loader)
        for j in range(len(data_loader)):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(data_loader)
                batch = next(dataloader_iter)
                print('new iters')
            gt[batch['frame_id'][0]] = batch['gt_boxes'][0]
            box_list.append(batch['gt_boxes'][0])
        with open(output_dir + '/gt.pickle', 'wb') as handle:
            pickle.dump(gt, handle, protocol=pickle.HIGHEST_PROTOCOL)               
    else:
        with open(output_dir + '/gt.pickle', 'rb') as handle:
            gt = pickle.load(handle)  
    key_sorted = sorted(gt)
    gt_sorted = {key:gt[key] for key in key_sorted}
    gt = gt_sorted
    
    # box_list = np.vstack(box_list)
    # size_mean = (box_list[:,3]*box_list[:,4]*box_list[:,5])**(1.0/3.0)
    # size_gt = np.mean(size_mean)
    # scaling
    # KITTI
    # scale_list = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    # scale_step = 0.1    
    # num_label = 4000
    # size_gt = (3.89*1.62*1.53)**(1/3.0) 

    # Waymo
    sigma = 0.4
    scale_min = 1.0 - sigma
    scale_max = 1.0 + sigma
    scale_list = np.linspace(scale_min, scale_max, 3)

    # data
    size_all = []
    size_data = []
    index_data = []
    for i in range(len(key_sorted)):
        key = key_sorted[i]
        #print(key)
        gt_temp = gt[key]
        if not gt_temp.shape[0]:
            continue        
        size_i = (gt_temp[:,3]*gt_temp[:,4]*gt_temp[:,5])**(1.0/3.0)
        size_all.append(size_i)

        for j in range(len(scale_list)):
            # size
            size_data.append(size_i*scale_list[j])
            # index
            index_temp = 1000000000*j + 1000*i + np.arange(gt_temp.shape[0])
            index_data.append(index_temp)
    size_all = np.concatenate(size_all)
    size_gt = np.mean(size_all)
    size_data = np.concatenate(size_data)
    index_data = np.concatenate(index_data)
    idx_sorted = np.argsort(size_data)
    size_data = size_data[idx_sorted]
    index_data = index_data[idx_sorted]
    size_uniform = np.linspace(size_gt*scale_min, size_gt*scale_max, size_data.shape[0])  
    
    # saving
    for k in range(len(scale_list)):
        dataloader_iter = iter(data_loader)
        for i in range(len(data_loader)):
            if i%100==0:
                print(i) 
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(data_loader)
                batch = next(dataloader_iter)
                print('new iters')
            points = batch['points'][:,1:5]        
            key = batch['frame_id'][0]
            index_key = list(key_sorted).index(key)
            gt_boxes = batch['gt_boxes'][0][:,0:7]   
            #visualize_utils.draw_scenes(points, gt_boxes) 
            if not gt_boxes.shape[0]:           
                np.save(output_dir+'/lidar/'+key+str(scale_list[k])+'.npy', points)
                np.save(output_dir+'/label/'+key+str(scale_list[k])+'.npy', gt_boxes)                              
                continue

            # saving
            points_new = []
            gt_new = []
            mask = np.zeros(points.shape[0])   
            for j in range(gt_boxes.shape[0]):            
                box_j = copy.deepcopy(gt_boxes[j:j+1])                
                mask_j = roiaware_pool3d_utils.points_in_boxes_cpu(points[:,0:3], box_j).squeeze(0)
                pts_j = copy.deepcopy(points[mask_j==1])
                size_j = (box_j[0,3]*box_j[0,4]*box_j[0,5])**(1/3)

                id_j = k*1000000000 + index_key*1000 + j
                size_adv_j = size_uniform[index_data==id_j]
                scale_adv_j = size_adv_j/size_j

                pts_j[:,0:3] -= box_j[0,0:3]
                pts_j[:,0:3] *= scale_adv_j
                pts_j[:,0:3] += box_j[0,0:3]
                box_j[0,3:6] *= scale_adv_j
                if scale_adv_j > 1:
                    mask_j = roiaware_pool3d_utils.points_in_boxes_cpu(points[:,0:3], box_j).squeeze(0)
                mask += mask_j
                points_new.append(pts_j)
                gt_new.append(box_j)
            points_new.append(points[mask==0])
            points_new = np.concatenate(points_new)
            gt_new = np.concatenate(gt_new)
            if False:
                visualize_utils.draw_scenes(points, gt_boxes)
                visualize_utils.draw_scenes(points_new, gt_new)

            np.save(output_dir+'/lidar/'+key+str(scale_list[k])+'.npy', points_new)
            np.save(output_dir+'/label/'+key+str(scale_list[k])+'.npy', gt_new)


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_test = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_test = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / ('log_eval_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_test:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    
    cfg.DATA_CONFIG.DATA_AUGMENTOR = None
    cfg.DATA_CONFIG.SAMPLED_INTERVAL.train = 3
    data_set, data_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=True
    )    
    scaling_uniform(data_loader, args, logger)

if __name__ == '__main__':
    # points = np.load('/media/lxh/Data/ScAR/attack/att_d/0.2/lidar/000001.npy')
    # gt = np.load('/media/lxh/Data/ScAR/attack/att_d/0.2/label/000001.npy')
    # visualize_utils.draw_scenes(points, gt)    
    main()
