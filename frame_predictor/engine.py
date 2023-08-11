# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
from contextlib import suppress
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from mmcv.ops.focal_loss import softmax_focal_loss
import timm.loss



def CosineLoss(output1, output2):
    cosine_sim = F.cosine_similarity(output1, output2, dim=2)
    loss = torch.mean((1 - cosine_sim).sum(-1))
    
    return loss

def MSELoss(output1,output2):
    mse_loss = (output1 - output2) ** 2
    mse_loss = mse_loss.sum(-1)
    return torch.mean(mse_loss)


def train_one_epoch(model: torch.nn.Module, 
                    vqvae: torch.nn.Module,
                    text_encoder: torch.nn.Module,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    epoch: int, 
                    loss_scaler, 
                    max_norm: float = 0,
                    set_training_mode=True, 
                    tb_logger=None, 
                    start_idx=0, 
                    amp_autocast=suppress,
                    loss_type = "cosine"):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    torch.cuda.synchronize()

    for samples, targets, text in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # text = text.to(device, non_blocking=True)

        with amp_autocast():
            with torch.no_grad():
                init_tokens,_,(_,_,init_id) = vqvae.encode(samples)
                target_tokens,_,(_,_,target_ind) = vqvae.encode(targets)
                text_feature = text_encoder(text)
            
            input = init_tokens if loss_type != "kl" else init_id
            pred_tokens = model(input, text_feature)
            
            # flat tokens for cal loss
            if loss_type != "kl":
                pred_ind = vqvae.get_ind(pred_tokens)
                B, C, H, W = init_tokens.shape
                flat_tar_token = target_tokens.reshape(B, C, H*W).permute(0,2,1)
                flat_pred_token = pred_tokens.reshape(B, C, H*W).permute(0,2,1)
            else:
                pred_ind = pred_tokens.max(-1).indices

            if loss_type == 'cosine':
                loss = CosineLoss(flat_pred_token, flat_tar_token.detach())
            elif loss_type == 'mse':
                loss = MSELoss(flat_pred_token, flat_tar_token.detach()) + 0.1 * CosineLoss(flat_pred_token, flat_tar_token.detach())
            elif loss_type == 'kl':
                loss = F.cross_entropy(pred_tokens, target_ind)
                # loss = softmax_focal_loss(pred_tokens, target_ind)
            else:
                NotImplementedError
            loss_value = loss.item()

            acc = sum(pred_ind == target_ind) / len(pred_ind)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(acc=acc)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if tb_logger is not None and utils.get_rank() == 0 and start_idx % 50 == 0:
            for k, meter in metric_logger.meters.items():
                tb_logger.add_scalar('train/{}_avg'.format(k), meter.global_avg, start_idx)
                tb_logger.add_scalar('train/{}_val'.format(k), meter.value, start_idx)
        start_idx += 1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train_one_epoch_for_vq_trans(model: torch.nn.Module, 
                    vqvae: torch.nn.Module,
                    data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer,
                    device: torch.device, 
                    epoch: int, 
                    loss_scaler, 
                    max_norm: float = 0,
                    set_training_mode=True, 
                    tb_logger=None, 
                    start_idx=0, 
                    amp_autocast=suppress,
                ):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    torch.cuda.synchronize()

    for samples, targets, _ in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # text = text.to(device, non_blocking=True)

        with amp_autocast():
            with torch.no_grad():
                init_tokens,_,(_,_,init_id) = vqvae.encode(samples)
                target_tokens,_,(_,_,target_ind) = vqvae.encode(targets)
            pred_tokens, quant_loss = model(init_id, target_ind)
            
            # flat tokens for cal loss
            B, N, C = pred_tokens.shape
            pred_tokens = pred_tokens.view(B * N, C)
            pred_ind = pred_tokens.max(-1).indices
            
              #  loss = F.cross_entropy(pred_tokens, target_ind)
            ce_loss = F.cross_entropy(pred_tokens, target_ind)
            loss = ce_loss + quant_loss
            loss_value = loss.item()

            acc = sum(pred_ind == target_ind) / len(pred_ind)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print("current out put is {}".format(pred_tokens))
            print("quant_loss is {}".format(quant_loss.item()))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(ce_loss=ce_loss.item())
        metric_logger.update(quant_loss = quant_loss.item())
        metric_logger.update(acc=acc)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if tb_logger is not None and utils.get_rank() == 0 and start_idx % 50 == 0:
            for k, meter in metric_logger.meters.items():
                tb_logger.add_scalar('train/{}_avg'.format(k), meter.global_avg, start_idx)
                tb_logger.add_scalar('train/{}_val'.format(k), meter.value, start_idx)
        start_idx += 1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def sample(model: torch.nn.Module, 
            vqvae: torch.nn.Module,
            text_encoder: torch.nn.Module,
            dataset,
            idx):

        
        samples, _, text = dataset[idx]
        samples = samples.unsqueeze(0)
        
        vqvae.encode(samples)
        # autoregressive generate
        for _ in range(10):
            pass

