import importlib
import torch
import torchvision
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
from torch.nn.parallel import DataParallel, DistributedDataParallel
from basicsr.utils.dist_util import get_dist_info
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from importlib import import_module
import basicsr.loss as loss
import numpy as np
import matplotlib.pyplot as plt
import random

import json

def create_video_model(opt):
    module = import_module('basicsr.models.archs.dealthdr_arch')
    model = module.make_model(opt)
    return model

metric_module = importlib.import_module('basicsr.metrics')

class DeAltHDRModel(BaseModel):
    def __init__(self, opt):
        super(DeAltHDRModel, self).__init__(opt)
        self.net_g = create_video_model(opt)
        self.net_g = self.model_to_device(self.net_g)
        self.n_sequence = opt['n_sequence']
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))
            print("load_model", load_path)
        if self.is_train:
            self.init_training_settings()
        self.loss = loss.L1BaseLoss()
        self.vggloss = loss.VGGPerceptualLoss(self.device)
        self.rankloss = loss.L1RankLoss()
        device = self.device
        
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Training mode configuration
        self.training_mode = opt.get('training_mode', 'mixed')
        self.use_dual_encoder = opt.get('use_dual_encoder', True)
        self.exposure_types = ['long', 'short']  # Alternating exposure types

    def init_training_settings(self):
        self.net_g.train()
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        
    def model_to_device(self, net):
        net = net.to(self.device)
        if self.opt['dist']:
            net = DistributedDataParallel(
                net,
                device_ids=[torch.cuda.current_device()],
                find_unused_parameters=False)
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')
        train_opt['optim_g'].pop('type')
        self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                            **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
    
    def get_training_mode_for_batch(self, batch_size):
        """Determine training mode for each sample in the batch"""
        # 30% optical flow, 30% attention, 40% FGMA (mixed)
        optical_flow_count = int(0.3 * batch_size)
        attention_count = int(0.3 * batch_size)
        mixed_count = batch_size - optical_flow_count - attention_count
        
        modes = ['optical_flow'] * optical_flow_count + \
                ['attention'] * attention_count + \
                ['mixed'] * mixed_count
        
        # Shuffle the modes
        random.shuffle(modes)
        return modes
    
    def get_exposure_types_for_batch(self, batch_size):
        """Get exposure types for each sample in the batch"""
        # Alternating between long and short exposure
        types = []
        for i in range(batch_size):
            types.append(self.exposure_types[i % len(self.exposure_types)])
        return types
    
    # method to feed the data to the model.
    def feed_data(self, data):
        lq, gt, _, _ = data
        self.lq = lq.to(self.device).half()
        self.gt = gt.to(self.device)

    def get_sensitivity_for_sample(self, mode):
        """
        Get sensitivity parameter based on training mode.
        Paper uses 16 sampling points: s=0, 6 points in (0,1], 6 points in (1,100), s=15, s=100, s=∞
        """
        if mode == 'optical_flow':
            return 0.0  # Pure optical flow (no attention)
        elif mode == 'attention':
            return float('inf')  # Pure attention (full mask)
        else:  # mixed - use FGMA with random sensitivity
            # Sample from the 16 key points
            points_0_to_1 = [i * 1.0/6 for i in range(1, 7)]  # 6 points in (0, 1]
            points_1_to_100 = [1 + i * 99.0/6 for i in range(1, 7)]  # 6 points in (1, 100)
            sample_points = [0] + points_0_to_1 + points_1_to_100 + [15, 100, float('inf')]
            return random.choice(sample_points)
    
    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        with torch.cuda.amp.autocast():
            loss_dict = OrderedDict()
            loss_dict['l_pix'] = 0

            frame_num = self.lq.shape[1]
            batch_size = self.lq.shape[0]
            
            # Get training modes and exposure types for this batch
            training_modes = self.get_training_mode_for_batch(batch_size)
            exposure_types = self.get_exposure_types_for_batch(batch_size)
            
            k_cache, v_cache = None, None
            
            for j in range(frame_num):
                target_g_images = self.gt[:, j, :, :, :]
                
                # Prepare input frames: T-2, T-1, T, T+1, T+2
                if j >= 2 and j < frame_num - 2:
                    # We have enough frames for T-2,T-1,T,T+1,T+2
                    input_frames = torch.stack([
                        self.lq[:, j-2, :, :, :],  # T-2
                        self.lq[:, j-1, :, :, :],  # T-1
                        self.lq[:, j, :, :, :],    # T
                        self.lq[:, j+1, :, :, :],  # T+1
                        self.lq[:, j+2, :, :, :]   # T+2
                    ], dim=1)  # [B, 5, C, H, W]
                else:
                    # Handle edge cases by padding with current frame
                    frames = []
                    for offset in [-2, -1, 0, 1, 2]:
                        frame_idx = max(0, min(frame_num-1, j + offset))
                        frames.append(self.lq[:, frame_idx, :, :, :])
                    input_frames = torch.stack(frames, dim=1)
                
                # Process batch with mixed training modes
                batch_outputs = []
                for b in range(batch_size):
                    sample_frames = input_frames[b:b+1]  # [1, 5, C, H, W]
                    sample_mode = training_modes[b]
                    sample_exposure = exposure_types[b]
                    sample_sensitivity = self.get_sensitivity_for_sample(sample_mode)
                    
                    # Forward pass with specific training mode and sensitivity
                    out_g, _, _ = self.net_g(
                        sample_frames, k_cache, v_cache,
                        exposure_type=sample_exposure,
                        training_mode=sample_mode,
                        sensitivity=sample_sensitivity
                    )
                    
                    batch_outputs.append(out_g)
                
                # Stack outputs back to batch
                out_g = torch.cat(batch_outputs, dim=0)
                
                # Compute losses (tone-mapped domain as per paper)
                # Apply mu-law tone mapping
                mu = 5000.0
                out_g_tonemapped = torch.log(1 + mu * out_g) / torch.log(torch.tensor(1 + mu))
                target_tonemapped = torch.log(1 + mu * target_g_images) / torch.log(torch.tensor(1 + mu))
                
                l_pix = self.loss(out_g_tonemapped, target_tonemapped)
                vgg_pix = self.vggloss(out_g_tonemapped, target_tonemapped)
                
                loss_dict['l_pix'] = loss_dict['l_pix'] + l_pix + 0.5 * vgg_pix

        # normalize w.r.t. total frames seen.
        loss_dict['l_pix'] /= frame_num
        l_total = loss_dict['l_pix'] + 0 * sum(p.sum() for p in self.net_g.parameters())
        loss_dict['l_pix'] = loss_dict['l_pix']

        self.scaler.scale(l_total).backward()
        self.scaler.unscale_(self.optimizer_g)
        self.scaler.step(self.optimizer_g)
        self.scaler.update()
        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self, sensitivity=15.0):
        """
        Test function with configurable sensitivity for dynamic FLOPs adjustment
        
        Args:
            sensitivity: Sensitivity parameter for FGMA (default 15.0 for balanced mode)
        """
        self.net_g.eval()
        with torch.no_grad():
            self.outputs_list = []
            self.gt_lists = []
            self.lq_lists = []
            frame_num = self.lq.shape[1]
            k_cache, v_cache = None, None
            
            for j in range(frame_num):
                target_g_images = self.gt[:, j, :, :, :]    
                
                # Prepare input frames: T-2, T-1, T, T+1, T+2
                if j >= 2 and j < frame_num - 2:
                    input_frames = torch.stack([
                        self.lq[:, j-2, :, :, :],  # T-2
                        self.lq[:, j-1, :, :, :],  # T-1
                        self.lq[:, j, :, :, :],    # T
                        self.lq[:, j+1, :, :, :],  # T+1
                        self.lq[:, j+2, :, :, :]   # T+2
                    ], dim=1)  # [B, 5, C, H, W]
                else:
                    # Handle edge cases by padding with current frame
                    frames = []
                    for offset in [-2, -1, 0, 1, 2]:
                        frame_idx = max(0, min(frame_num-1, j + offset))
                        frames.append(self.lq[:, frame_idx, :, :, :])
                    input_frames = torch.stack(frames, dim=1)
                
                # Use mixed mode for testing with specified sensitivity
                out_g, k_cache, v_cache = self.net_g(
                    input_frames.float(), 
                    k_cache, 
                    v_cache,
                    exposure_type='long',  # Default to long exposure for testing
                    training_mode='mixed',
                    sensitivity=sensitivity
                )
                
                self.outputs_list.append(out_g)
                self.gt_lists.append(target_g_images)
                self.lq_lists.append(self.lq[:, j,:, :, :])
        self.net_g.train()
    
    def non_cached_test(self):
        # proxy to the actual scores to save time.
        self.net_g.eval()
        with torch.no_grad():
            k_cache, v_cache = None, None
            pred, _, _, _ = self.net_g(self.lq.float(), k_cache, v_cache)
            if isinstance(pred, list):
                pred = pred[-1]
            self.output = pred
        self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        logger = get_root_logger()
        import os
        return self.nondist_validation(dataloader, current_iter, 
                                       tb_logger, save_img, 
                                       rgb2bgr, use_image)
    
    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')
        cnt = 0
        
        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            folder_name, img_name = val_data[len(val_data)-1][0][0].split('.')
            self.feed_data(val_data)
            self.test()

            for temp_i  in range(len(self.outputs_list)):
                sr_img = tensor2img(self.outputs_list[temp_i], rgb2bgr=rgb2bgr)
                gt_img = tensor2img(self.gt_lists[temp_i], rgb2bgr=rgb2bgr)
                lq_img = tensor2img(self.lq_lists[temp_i], rgb2bgr=rgb2bgr)

                if save_img:
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                            folder_name,
                                            f'{img_name}_frame{temp_i}_res.png')
                    
                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                            folder_name,
                                            f'{img_name}_frame{temp_i}_gt.png')
                    
                    save_lq_img_path = osp.join(self.opt['path']['visualization'],
                    folder_name,
                    f'{img_name}_frame{temp_i}_lq.png')
                        
                    imwrite(sr_img, save_img_path)
                    imwrite(gt_img, save_gt_img_path)
                    imwrite(lq_img, save_lq_img_path)

                if with_metrics:
                    # calculate metrics
                    opt_metric = deepcopy(self.opt['val']['metrics'])
                    if use_image:
                        for name, opt_ in opt_metric.items():
                            metric_type = opt_.pop('type')
                            self.metric_results[name] += getattr(
                                metric_module, metric_type)(sr_img, gt_img, **opt_)
                    else:
                        for name, opt_ in opt_metric.items():
                            metric_type = opt_.pop('type')
                            self.metric_results[name] += getattr(
                                metric_module, metric_type)(self.outputs_list[temp_i], self.gt_lists[temp_i], **opt_)

                cnt += 1
                if rank == 0:
                    for _ in range(world_size):
                        pbar.update(1)
                        pbar.set_description(f'Test {img_name}')
        
        if rank == 0:
            pbar.close()
            
        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt

            self._log_validation_metric_values(current_iter,
                                               tb_logger)
        return current_metric

    def _log_validation_metric_values(self, current_iter, tb_logger):
        log_str = f'Validation,\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        # pick the current frame.
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq[:,1,:,:,:].detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt[:,1,:,:,:].detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)


