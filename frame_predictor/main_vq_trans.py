# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import argparse
import datetime
import logging
import os
import random
from tabnanny import verbose
import numpy as np
import time
import utils
import json
import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as NativeDDP
from tensorboardX import SummaryWriter
import warnings
from engine import train_one_epoch_for_vq_trans as train_one_epoch
warnings.filterwarnings("ignore", category=UserWarning)

from pathlib import Path

from timm.models import create_model

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
import timm.models
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from BridgeDataset import BridgeDataV1


# import models
import Predictor
import Clip
import VqControl
import VQVAE
def get_args_parser():
    parser = argparse.ArgumentParser('Frame Predictor training script', add_help=False)

    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--model', default='translator_base', type=str, 
                        metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--with-timeline',action='store_true')
    

    parser.add_argument('--text-encoder', default='clip_model', type=str)
    parser.add_argument('--vqvae',default='vq-8', type=str)
    parser.add_argument('--input-size', default=128, type=int, help='images input size')
    parser.add_argument('--loss-type', default="cosine", type=str)
    # sample distance
    parser.add_argument('--start-point', default=3, type=int)
    parser.add_argument('--end-point', default=4, type=int)
    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "fusedadamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    parser.add_argument('--seed', default=0, type=int)
    #resume
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')

    # Dataset parameters
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--port', default=29529, type=int, help='port')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    
    return parser

def main(args):
    utils.init_distributed_mode(args, verbose=True)
    output_dir = Path(args.output_dir)
    tb_logger = None
    if utils.get_rank() == 0:
        tensorboard_path = os.path.join(output_dir, 'events')
        tb_logger = SummaryWriter(tensorboard_path)
    utils.init_log(__name__, log_file=os.path.join(output_dir, 'full_log.txt'))
    logger = logging.getLogger(__name__)
    print = logger.info

    print(args)
    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #build dataset
    dataset = BridgeDataV1(
        input_shape=args.input_size,
        target_range=range(args.start_point,args.end_point),
        with_timeline=args.with_timeline
        )

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
            dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    dataloader = torch.utils.data.DataLoader(
        dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    # build model
    model = create_model(args.model)
    vqvae = create_model("VQVAE", name=args.vqvae)
    model.to(device)
    vqvae.to(device)
    vqvae.eval()
    if utils.get_rank() == 0:
        model.eval()
        i = torch.rand(1, model.input_length).long().to(device)
        flops = FlopCountAnalysis(model, (i, i))
        if args.rank == 0:
            print(flop_count_table(flops))
        model.train()
    torch.distributed.barrier()
    model_without_ddp = model

    print(f'batch size {args.batch_size}, world size {utils.get_world_size()}')
    print(f'ori lr {args.lr}')
    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    print(f'scaled lr {args.lr}')
    optimizer = create_optimizer(args, model_without_ddp)

    amp_autocast = torch.cuda.amp.autocast
    loss_scaler = NativeScaler()
    model = NativeDDP(model, device_ids=[args.gpu], find_unused_parameters=False)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'number of params: {n_parameters}')

    torch.distributed.barrier()

    lr_scheduler, _ = create_scheduler(args, optimizer)
    try:
        checkpoint = utils.load_checkpoint(args.resume)
        print('>>>>>> resume from {}'.format(args.resume))
        model_without_ddp.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
    except:
        print('>>>>>> no resume')

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    start_idx = args.start_epoch * len(dataloader)
    for epoch in range(args.start_epoch, args.epochs):
        
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, 
            vqvae=vqvae,
            data_loader=dataloader,
            optimizer=optimizer, 
            device=device, 
            epoch=epoch, 
            loss_scaler=loss_scaler,
            max_norm=args.clip_grad,
            tb_logger=tb_logger, 
            start_idx=start_idx,
            amp_autocast=amp_autocast,
        )
        start_idx += len(dataloader)

        lr_scheduler.step(epoch)
        
        # if epoch % 5 == 0:


        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            # with (output_dir / "log.txt").open("a") as f:
            #     f.write(json.dumps(log_stats) + "\n")
            with open(os.path.join(output_dir, "log.txt"), 'a') as f:
                f.write(json.dumps(log_stats) + "\n")
        try:
            checkpoint_path = args.resume
            utils.save_checkpoint({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, checkpoint_path)
        except:
            raise RuntimeError

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

