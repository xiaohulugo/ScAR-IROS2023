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

from scipy.optimize import linear_sum_assignment
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
from os.path import exists
from scipy.optimize import minimize
from scipy.spatial import distance


hist_gt = None
phi = None

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
    #args.ckpt = '/home/lxh/Documents/Code/Detection3D/ScAR/output/kitti/second/default/ckpt/checkpoint_epoch_20.pth'    
    args.cfg_file = '/home/lxh/Documents/Code/Detection3D/ScAR/tools/cfgs/waymo_models/second.yaml'
    #args.ckpt = '/home/lxh/Documents/Code/Detection3D/ScAR/output/IROS/waymo/second/default/ckpt/checkpoint_epoch_10.pth'
    args.ckpt = '/home/lxh/Documents/Code/Detection3D/ScAR/output/IROS/waymo/second_adv2/default/ckpt/checkpoint_epoch_10.pth'

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


def get_original_data(data_loader, folder_out):
    if not os.path.exists(folder_out+'/label'):
        os.makedirs(folder_out+'/label')
    if not os.path.exists(folder_out+'/lidar'):
        os.makedirs(folder_out+'/lidar')
    
    size_list = []
    data_loader.dataset.scale = 1.0
    dataloader_iter = iter(data_loader)
    for i in range(len(data_loader)):
        if i % 100 == 0:
            print(i)
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(data_loader)
            batch = next(dataloader_iter)
        gt_bboxes = batch['gt_boxes'][0][:,0:7]
        points = batch['points'][:,1:4]  
        frame_id = batch['frame_id'][0]
        if gt_bboxes.shape[0]>0:
            size_list.append(gt_bboxes[:,3:6])

        np.save(folder_out+'/lidar/'+frame_id+'.npy', points)
        np.save(folder_out+'/label/'+frame_id+'.npy', gt_bboxes)     
    size_list = np.vstack(size_list)
    np.save(folder_out+'/size_gt.npy', size_list)     
    a = 0

##################################################
# Model-aware Attack
##################################################
def att_m(model, data_loader, sigma_m, folder_out):
    if not os.path.exists(folder_out+'/label'):
        os.makedirs(folder_out+'/label')
    if not os.path.exists(folder_out+'/lidar'):
        os.makedirs(folder_out+'/lidar')

    model.eval()
    scale_list = np.linspace(1.0-sigma_m,1.0+sigma_m, 5)
    arg_sort = np.argsort(abs(scale_list-1.0))
    scale_list = scale_list[arg_sort]
    num_scales = len(scale_list)

    pred_list = []
    for i in range(0,num_scales):
        file_i = folder_out+'/'+'scale'+str(scale_list[i])+'.pickle'
        if not os.path.exists(file_i):
            data_loader_i = copy.deepcopy(data_loader)
            output_dict = {}
            data_loader_i.dataset.scale = scale_list[i]
            dataloader_iter_i = iter(data_loader_i)
            for j in range(len(data_loader_i)):
                if j % 100 == 0:
                    print(j)
                try:
                    batch = next(dataloader_iter_i)
                except StopIteration:
                    dataloader_iter_i = iter(data_loader_i)
                    batch = next(dataloader_iter_i)
                    print('new iters')
                load_data_to_gpu(batch)

                with torch.no_grad():
                    pred_dicts, ret_dict = model(batch) 
                    pred_dicts[0]['pred_boxes'][:,3:6] /= data_loader_i.dataset.scale 
                gt_bbox = batch['gt_boxes'][0].cpu().numpy()
                pred_bbox = pred_dicts[0]['pred_boxes']
                pred_score = pred_dicts[0]['pred_scores'].view(-1,1)
                pred = torch.cat([pred_bbox, pred_score], dim=1)
                output_dict[batch['frame_id'][0]] = pred.cpu().numpy()

                if False:
                    frame_id = batch['frame_id'][0]
                    visualize_utils.draw_scenes(batch['points'][:,1:4],gt_bbox,output_dict[batch['frame_id'][0]])            
            with open(file_i, 'wb') as handle:
                pickle.dump(output_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(file_i, 'rb') as handle:
            pred = pickle.load(handle)     
        pred_list.append(pred)   
    
    # find minimal sigma for each instance
    data_loader.dataset.scale = 1.0
    dataloader_iter = iter(data_loader)
    scale_all = []
    for i in range(len(data_loader)):
        if i % 100 == 0:
            print(i)
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(data_loader)
            batch = next(dataloader_iter)
        gt_bboxes = batch['gt_boxes'][0][:,0:7]
        if gt_bboxes.shape[0]<1:
            continue
        frame_id = batch['frame_id'][0]
        scale_boxes = np.zeros(gt_bboxes.shape[0])
        for j in range(len(pred_list)):
            pred = pred_list[j][frame_id][:,0:7]
            if pred.shape[0]<1:
                continue            
            if False:
                visualize_utils.draw_scenes(batch['points'][:,1:4] , gt_bboxes, pred)
                            
            overlapping = boxes_iou3d_gpu(torch.tensor(gt_bboxes).cuda(), torch.tensor(pred).cuda())
            overlapping = (overlapping>0.7).cpu().numpy()            
            overlapping = np.sum(overlapping, axis=1)
            mask = (overlapping+scale_boxes)==0
            scale_boxes[mask==1] = scale_list[j]
        scale_boxes[scale_boxes==0] = 1.0
        scale_all.append(scale_boxes)

        # saving
        points = batch['points'][:,1:4]  
        points_new = []
        gt_new = []
        mask = np.zeros(points.shape[0])   
        for j in range(gt_bboxes.shape[0]):            
            box_j = copy.deepcopy(gt_bboxes[j:j+1])                
            mask_j = roiaware_pool3d_utils.points_in_boxes_cpu(points[:,0:3], box_j).squeeze(0)
            pts_j = copy.deepcopy(points[mask_j==1])
            scale_adv_j = scale_boxes[j]
            z_shift = box_j[0,5]/2.0*(1.0-scale_adv_j)

            pts_j[:,0:3] -= box_j[0,0:3]
            pts_j[:,0:3] *= scale_adv_j
            pts_j[:,0:3] += box_j[0,0:3]
            #pts_j[:,2] -= z_shift
            box_j[0,3:6] *= scale_adv_j
            #box_j[0,2] -= z_shift
            if scale_adv_j > 1:
                mask_j = roiaware_pool3d_utils.points_in_boxes_cpu(points[:,0:3], box_j).squeeze(0)
            mask += mask_j
            points_new.append(pts_j)
            gt_new.append(box_j)
        points_new.append(points[mask==0])
        points_new = np.concatenate(points_new)
        gt_new = np.concatenate(gt_new)
        if False:
            visualize_utils.draw_scenes(points, gt_bboxes)
            visualize_utils.draw_scenes(points_new, gt_new, gt_bboxes)

        np.save(folder_out+'/lidar/'+frame_id+'.npy', points_new)
        np.save(folder_out+'/label/'+frame_id+'.npy', gt_new)        
    scale_all = np.concatenate(scale_all)
    print(np.mean(scale_all))
    np.save(folder_out+'/scale.npy', scale_all)        

##################################################
# Distribution-aware Attack
##################################################
def objective(x):
    loss1 = np.std(x)
    loss2 = np.sum(np.abs(x))
    loss = loss2
    return loss

# def objective(x):
#     hist_new = hist_gt + x
#     kl1 = hist_new*np.log((hist_new+0.00001)/(hist_gt+0.000001))
#     kl2 = hist_gt*np.log((hist_gt+0.00001)/(hist_new+0.000001))
#     kl = (np.sum(kl1)+np.sum(kl2))/2.0
#     loss = -kl
#     return loss

def constraint_sum(x):
    return np.sum(hist_gt+x)-1.0

def constraint_kl(x):
    dist1 = (hist_gt+0.000001)/np.sum(hist_gt+0.000001)
    hist_new = hist_gt + x
    dist2 = (hist_new+0.000001)/np.sum(hist_new+0.000001)

    js_dist = distance.jensenshannon(dist1, dist2)
    # kl1 = dist1*np.log(dist1/dist2)
    # kl2 = dist2*np.log(dist2/dist1)
    # kl = ((np.sum(kl1)+np.sum(kl2))/2.0)**(0.5)
    return js_dist-phi


def att_d(data_loader, phi_d, folder_out, size_gt):    
    if not os.path.exists(folder_out+'/label'):
        os.makedirs(folder_out+'/label')
    if not os.path.exists(folder_out+'/lidar'):
        os.makedirs(folder_out+'/lidar')
    
    # get original distribution    
    size_gt = size_gt[:,0]*size_gt[:,1]*size_gt[:,2]
    size_gt_mean = np.mean(size_gt)
    size_gt_std = np.std(size_gt)
    mask = (size_gt>(size_gt_mean*0.5))*(size_gt<(size_gt_mean*1.5))
    size_gt = size_gt[mask]

    global hist_gt
    hist_gt, bin_edges = np.histogram(size_gt, density=True, bins=20)
    hist_gt = hist_gt / hist_gt.sum()
    global phi
    phi = phi_d

    # optimization
    bnds = []
    thr = np.max(hist_gt)
    for i in range(hist_gt.shape[0]):
        v_min = np.max([-thr, -hist_gt[i]])
        v_max = np.min([thr, 1-hist_gt[i]])       
        bnds.append((v_min, v_max))

    # initial guesses
    x0 = np.ones(hist_gt.shape[0])*0.00001
    x0[0] = -(hist_gt.shape[0]-1)*0.00001
    print('Initial SSE Objective: ' + str(objective(x0)))

    con1 = {'type': 'eq', 'fun': constraint_sum}
    con2 = {'type': 'eq', 'fun': constraint_kl}
    cons = ([con1, con2])
    solution = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
    x = solution.x

    # show final objective
    print('Final SSE Objective: ' + str(objective(x)))
    print(constraint_sum(x))
    print(constraint_kl(x))
    kl = constraint_kl(x)+phi

    if True:
        fig = plt.figure(figsize = (10, 4))
        #xx = np.arange(hist_gt.shape[0])
        xx = bin_edges[0:-1]
        w = (bin_edges[1]-bin_edges[0])/3
        plt.bar(xx, hist_gt, color ='red', width = w)
        plt.bar(xx+w, hist_gt+x, color ='blue', width = w)
        plt.xlabel("Size of annotation",  fontsize=12)
        plt.ylabel("Probability of annotations",  fontsize=12)
        plt.legend(["$\mathcal{B}_{b_i}$","$\mathcal{B}_{b_i+\delta_i}$"], loc=1, prop={'size': 18})                            
        plt.title('JS($\mathcal{B}_{b_i}$,$\mathcal{B}_{b_i+\delta_i}$) = '+str(kl)[0:5], fontsize=12)
        #plt.text(-5, 60, 'KL($\mathcal{B}_{b_i}$,$\mathcal{B}_{b_i+\delta_i}$) = '+str(kl)[0:3], fontsize = 12)
        #plt.show()
        plt.savefig(folder_out+'/'+str(phi_d)+'.pdf')
        

    # sampling
    hist_adv = hist_gt + x
    hist_adv = hist_adv/np.sum(hist_adv) 
    num_total = size_gt.shape[0]+100
    size_adv = []
    for i in range(hist_adv.shape[0]):
        p = hist_adv[i]
        num = int(num_total*p)
        if num>0:
            size = np.random.uniform(bin_edges[i], bin_edges[i+1], num)
            size_adv.append(size)
    size_adv = np.concatenate(size_adv)
    size_adv = size_adv[np.random.permutation(size_gt.shape[0])]

    # assign scale
    size_gt = np.sort(size_gt)
    size_adv = np.sort(size_adv)

    data_loader.dataset.scale = 1.0
    dataloader_iter = iter(data_loader)
    scale_all = []
    for i in range(len(data_loader)):
        if i % 100 == 0:
            print(i)
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(data_loader)
            batch = next(dataloader_iter)
        gt_bboxes = batch['gt_boxes'][0][:,0:7]
        size = gt_bboxes[:,3]*gt_bboxes[:,4]*gt_bboxes[:,5]
        frame_id = batch['frame_id'][0]
        if gt_bboxes.shape[0]<1:
            continue

        # saving
        points = batch['points'][:,1:4]  
        points_new = []
        gt_new = []
        mask = np.zeros(points.shape[0])   
        for j in range(gt_bboxes.shape[0]):            
            box_j = copy.deepcopy(gt_bboxes[j:j+1])                
            mask_j = roiaware_pool3d_utils.points_in_boxes_cpu(points[:,0:3], box_j).squeeze(0)
            pts_j = copy.deepcopy(points[mask_j==1])
            size_j = size[j]
            tt = np.where(size_gt==size_j)
            if tt[0].shape[0]<1:
                scale_adv_j = 1.0
            else:
                size_ref_j = size_adv[tt[0][0]]
                scale_adv_j = size_ref_j/size_j
            scale_all.append(scale_adv_j)

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
            visualize_utils.draw_scenes(points, gt_bboxes)
            visualize_utils.draw_scenes(points_new, gt_new, gt_bboxes)

        np.save(folder_out+'/lidar/'+frame_id+'.npy', points_new)
        np.save(folder_out+'/label/'+frame_id+'.npy', gt_new)        
    scale_all = np.array(scale_all)
    print(np.mean(scale_all))
    np.save(folder_out+'/scale.npy', scale_all)     


# def att_d(data_loader, phi_d, folder_out, size_gt):    
#     if not os.path.exists(folder_out+'/label'):
#         os.makedirs(folder_out+'/label')
#     if not os.path.exists(folder_out+'/lidar'):
#         os.makedirs(folder_out+'/lidar')
    
#     # get original distribution    
#     size_gt = size_gt[:,0]*size_gt[:,1]*size_gt[:,2]
#     size_gt_mean = np.mean(size_gt)
#     mask = (size_gt>size_gt_mean*0.6)*(size_gt<size_gt_mean*1.4)
#     size_gt = size_gt[mask]

#     global hist_gt
#     hist_gt, bin_edges = np.histogram(size_gt, density=True, bins=20)
#     hist_gt = hist_gt / hist_gt.sum()
#     global phi
#     phi = phi_d

#     # optimization
#     bnds = []
#     for i in range(hist_gt.shape[0]):
#         v_min = np.max([-phi, -hist_gt[i]])
#         v_max = np.min([phi, 1-hist_gt[i]])
#         bnds.append((v_min, v_max))

#     # initial guesses
#     x0 = np.ones(hist_gt.shape[0])*0.00001
#     x0[0] = -(hist_gt.shape[0]-1)*0.00001
#     print('Initial SSE Objective: ' + str(objective(x0)))

#     con1 = {'type': 'eq', 'fun': constraint_sum}
#     con2 = {'type': 'eq', 'fun': constraint_kl}
#     cons = ([con1])
#     solution = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)
#     x = solution.x

#     # show final objective
#     print('Final SSE Objective: ' + str(objective(x)))
#     print(constraint_sum(x))
#     print(objective(x))
#     kl = -objective(x)

#     if True:
#         fig = plt.figure(figsize = (10, 4))
#         #xx = np.arange(hist_gt.shape[0])
#         xx = bin_edges[0:-1]
#         w = (bin_edges[1]-bin_edges[0])/3
#         plt.bar(xx, hist_gt, color ='red', width = w)
#         plt.bar(xx+w, hist_gt+x, color ='blue', width = w)
#         plt.xlabel("Size of annotation",  fontsize=12)
#         plt.ylabel("Probability of annotations",  fontsize=12)
#         plt.legend(["$\mathcal{B}_{b_i}$","$\mathcal{B}_{b_i+\delta_i}$"], loc=1, prop={'size': 18})                            
#         plt.title('KL($\mathcal{B}_{b_i}$,$\mathcal{B}_{b_i+\delta_i}$) = '+str(kl)[0:4], fontsize=12)
#         #plt.text(-5, 60, 'KL($\mathcal{B}_{b_i}$,$\mathcal{B}_{b_i+\delta_i}$) = '+str(kl)[0:3], fontsize = 12)
#         #plt.show()
#         plt.savefig(folder_out+'/'+str(phi_d)+'.pdf')
        

#     # sampling
#     hist_adv = hist_gt + x
#     hist_adv = hist_adv/np.sum(hist_adv) 
#     num_total = size_gt.shape[0]+100
#     size_adv = []
#     for i in range(hist_adv.shape[0]):
#         p = hist_adv[i]
#         num = int(num_total*p)
#         if num>0:
#             size = np.random.uniform(bin_edges[i], bin_edges[i+1], num)
#             size_adv.append(size)
#     size_adv = np.concatenate(size_adv)
#     size_adv = size_adv[np.random.permutation(size_gt.shape[0])]

#     # assign scale
#     size_gt = np.sort(size_gt)
#     size_adv = np.sort(size_adv)

#     data_loader.dataset.scale = 1.0
#     dataloader_iter = iter(data_loader)
#     scale_all = []
#     for i in range(len(data_loader)):
#         if i % 100 == 0:
#             print(i)
#         try:
#             batch = next(dataloader_iter)
#         except StopIteration:
#             dataloader_iter = iter(data_loader)
#             batch = next(dataloader_iter)
#         gt_bboxes = batch['gt_boxes'][0][:,0:7]
#         size = gt_bboxes[:,3]*gt_bboxes[:,4]*gt_bboxes[:,5]
#         frame_id = batch['frame_id'][0]
#         if gt_bboxes.shape[0]<1:
#             continue

#         # saving
#         points = batch['points'][:,1:4]  
#         points_new = []
#         gt_new = []
#         mask = np.zeros(points.shape[0])   
#         for j in range(gt_bboxes.shape[0]):            
#             box_j = copy.deepcopy(gt_bboxes[j:j+1])                
#             mask_j = roiaware_pool3d_utils.points_in_boxes_cpu(points[:,0:3], box_j).squeeze(0)
#             pts_j = copy.deepcopy(points[mask_j==1])
#             size_j = size[j]
#             tt = np.where(size_gt==size_j)
#             if tt[0].shape[0]<1:
#                 scale_adv_j = 1.0
#             else:
#                 size_ref_j = size_adv[tt[0][0]]
#                 scale_adv_j = size_ref_j/size_j
#             scale_all.append(scale_adv_j)

#             pts_j[:,0:3] -= box_j[0,0:3]
#             pts_j[:,0:3] *= scale_adv_j
#             pts_j[:,0:3] += box_j[0,0:3]
#             box_j[0,3:6] *= scale_adv_j
#             if scale_adv_j > 1:
#                 mask_j = roiaware_pool3d_utils.points_in_boxes_cpu(points[:,0:3], box_j).squeeze(0)
#             mask += mask_j
#             points_new.append(pts_j)
#             gt_new.append(box_j)
#         points_new.append(points[mask==0])
#         points_new = np.concatenate(points_new)
#         gt_new = np.concatenate(gt_new)
#         if False:
#             visualize_utils.draw_scenes(points, gt_bboxes)
#             visualize_utils.draw_scenes(points_new, gt_new, gt_bboxes)

#         np.save(folder_out+'/lidar/'+frame_id+'.npy', points_new)
#         np.save(folder_out+'/label/'+frame_id+'.npy', gt_new)        
#     scale_all = np.array(scale_all)
#     print(np.mean(scale_all))
#     np.save(folder_out+'/scale.npy', scale_all)     



##################################################
# Blind Attack
##################################################

def att_b(data_loader, sigma_b, folder_out):    
    if not os.path.exists(folder_out+'/label'):
        os.makedirs(folder_out+'/label')
    if not os.path.exists(folder_out+'/lidar'):
        os.makedirs(folder_out+'/lidar')
    
    # assign scale
    data_loader.dataset.scale = 1.0
    dataloader_iter = iter(data_loader)
    scale_all = []
    for i in range(len(data_loader)):
        if i % 100 == 0:
            print(i)
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(data_loader)
            batch = next(dataloader_iter)
        gt_bboxes = batch['gt_boxes'][0][:,0:7]
        frame_id = batch['frame_id'][0]
        if gt_bboxes.shape[0]<1:
            continue

        # saving
        points = batch['points'][:,1:4]  
        points_new = []
        gt_new = []
        mask = np.zeros(points.shape[0])   
        for j in range(gt_bboxes.shape[0]):            
            box_j = copy.deepcopy(gt_bboxes[j:j+1])                
            mask_j = roiaware_pool3d_utils.points_in_boxes_cpu(points[:,0:3], box_j).squeeze(0)
            pts_j = copy.deepcopy(points[mask_j==1])
            scale_adv_j = 1+sigma_b
            scale_all.append(scale_adv_j)

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
            visualize_utils.draw_scenes(points, gt_bboxes)
            visualize_utils.draw_scenes(points_new, gt_new, gt_bboxes)

        np.save(folder_out+'/lidar/'+frame_id+'.npy', points_new)
        np.save(folder_out+'/label/'+frame_id+'.npy', gt_new)        
    scale_all = np.array(scale_all)
    print(np.mean(scale_all))
    np.save(folder_out+'/scale.npy', scale_all)     


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

    #get_original_data(data_loader, '/media/lxh/Data/ScAR/attack/no_adv')

    # size_gt = np.load('/media/lxh/Data/ScAR/attack/no_adv/size_gt.npy') 
    # phi_d_list = [0.2, 0.4, 0.6]
    # for phi_d in phi_d_list:
    #     folder_out = '/media/lxh/Data/ScAR/attack/att_d/'+str(phi_d)
    #     if not os.path.exists(folder_out):
    #         os.makedirs(folder_out)
    #     att_d(data_loader, phi_d, folder_out, size_gt)    

    sigma_b_list = [-0.4, 0.4]
    for sigma_b in sigma_b_list:
        folder_out = '/media/lxh/Data/Waymo_V1.2/waymo2cad/att_b/'+str(sigma_b)
        if not os.path.exists(folder_out):
            os.makedirs(folder_out)

        cfg.DATA_CONFIG.DATA_AUGMENTOR = None
        # cfg.DATA_CONFIG.SAMPLED_INTERVAL.train = 10
        # cfg.DATA_CONFIG.SAMPLED_INTERVAL.test = 10
        data_set, data_loader, sampler = build_dataloader(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            batch_size=args.batch_size,
            dist=dist_test, workers=args.workers, logger=logger, training=False
        )
        att_b(data_loader, sigma_b, folder_out)
    
    # #    
    # sigma_m_list = [0.1,0.2,0.3]
    # for sigma_m in sigma_m_list:
    #     folder_out = '/media/lxh/Data/Waymo_V1.2/waymo2cad/att_m/'+str(sigma_m)
    #     if not os.path.exists(folder_out):
    #         os.makedirs(folder_out)

    #     cfg.DATA_CONFIG.DATA_AUGMENTOR = None
    #     # cfg.DATA_CONFIG.SAMPLED_INTERVAL.train = 10
    #     # cfg.DATA_CONFIG.SAMPLED_INTERVAL.test = 10
    #     data_set, data_loader, sampler = build_dataloader(
    #         dataset_cfg=cfg.DATA_CONFIG,
    #         class_names=cfg.CLASS_NAMES,
    #         batch_size=args.batch_size,
    #         dist=dist_test, workers=args.workers, logger=logger, training=False
    #     )
    #     model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=data_set)
    #     model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    #     model.cuda()

    #     att_m(model, data_loader, sigma_m, folder_out)


def test_model():
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
    # cfg.DATA_CONFIG.SAMPLED_INTERVAL.train = 10
    # cfg.DATA_CONFIG.SAMPLED_INTERVAL.test = 10
    data_set, data_loader, sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_test, workers=args.workers, logger=logger, training=False
    )
    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=data_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist_test)
    model.cuda()
    model.eval()

    #
    output_folder = '/media/lxh/Data/Waymo_V1.2/waymo2cad/scar_result'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    dataloader_iter_i = iter(data_loader)
    for j in range(len(data_loader)):
        if j % 100 == 0:
            print(j)
        try:
            batch = next(dataloader_iter_i)
        except StopIteration:
            dataloader_iter_i = iter(data_loader)
            batch = next(dataloader_iter_i)
            print('new iters')
        load_data_to_gpu(batch)

        with torch.no_grad():
            pred_dicts, ret_dict = model(batch) 
        gt_bbox = batch['gt_boxes'][0].cpu().numpy()
        pred_bbox = pred_dicts[0]['pred_boxes']
        pred_score = pred_dicts[0]['pred_scores'].view(-1,1)
        pred = torch.cat([pred_bbox, pred_score], dim=1)
        pred = pred.cpu().numpy()

        # filter out overlapping<0.7
        overlapping = boxes_iou3d_gpu(torch.tensor(pred[:,0:7]).cuda(),torch.tensor(gt_bbox[:,0:7]).cuda())
        overlapping = (overlapping>0.7).cpu().numpy()     
        mask = np.sum(overlapping, axis=1)
        pred = pred[mask>0]
            
        np.save(output_folder+'/'+batch['frame_id'][0]+'.npy', pred)
        if False:
            visualize_utils.draw_scenes(batch['points'][:,1:4],gt_bbox,pred)            

if __name__ == '__main__':     
    main()
    #test_model()
