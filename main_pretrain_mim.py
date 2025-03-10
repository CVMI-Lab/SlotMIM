import argparse
import json
import os
import shutil
import time
import datetime

import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from data.datasets import ImageFolder, ImageFolderMask
from data.transforms import CustomDataAugmentation, eval_transform

from models import vision_transformer
from models.slotmim import SlotMIM
from utils.logger import setup_logger
from utils.util import AverageMeter
from utils.ddp import init_distributed_mode
from viz_slots_retrieval import prepare_knn, viz_slots
from eval_knn import extract_features, knn_classifier, ReturnIndexDataset

model_names = sorted(name for name in vision_transformer.__all__ if name.islower() and callable(vision_transformer.__dict__[name]))

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    FALSY_STRINGS = {"off", "false", "0"}
    TRUTHY_STRINGS = {"on", "true", "1"}
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def get_params_groups(model):
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

def clip_gradients(model, clip=0):
    total_norm = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            if clip:
                clip_coef = clip / (param_norm + 1e-6)
                if clip_coef < 1:
                    p.grad.data.mul_(clip_coef)
    return total_norm ** 0.5

def cancel_gradients_slots(epoch, model, freeze_slots):
    if epoch >= freeze_slots:
        return
    for n, p in model.named_parameters():
        if "prototypes_patch" in n or "prototypes_slot" in n:
            p.grad = None

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def get_parser():
    parser = argparse.ArgumentParser('SlotMIM', add_help=False)

    # dataset
    parser.add_argument('--dataset', type=str, default='COCO', choices=['COCO', 'COCOplus', 'ImageNet', 'Cityscapes', 'ImgList'], help='dataset type')
    parser.add_argument('--data-dir', type=str, default='./data', help='dataset director')
    parser.add_argument('--image-size', type=int, default=224, help='image crop size')
    parser.add_argument('--min-scale', type=float, default=0.08, help='minimum crop scale')
    parser.add_argument('--coco-path', type=str, default='datasets/coco/', help='coco dataset path')
    parser.add_argument('--imagenet-path', type=str, default='datasets/imagenet/', help='imagenet dataset path')
    # model
    parser.add_argument('--arch', type=str, default='resnet50', choices=model_names, help='backbone architecture')
    parser.add_argument('--dim-hidden', type=int, default=4096, help='hidden dimension')
    parser.add_argument('--dim-hidden-slot', type=int, default=4096, help='hidden dimension')
    parser.add_argument('--dim-out', type=int, default=256, help='output feature dimension')
    parser.add_argument('--use-bn-in-head', type=bool_flag, default=True, help='use batch norm in head')
    parser.add_argument('--num-prototypes', type=int, default=256, help='number of prototypes')
    parser.add_argument('--teacher-momentum', default=0.99, type=float, help='momentum value for the teacher model')
    parser.add_argument('--warmup-teacher-temp', default=0.07, type=float, help='initial temperature for the teacher model')
    parser.add_argument('--teacher-temp', default=0.07, type=float, help='final temperature of the teacher model')
    parser.add_argument('--warmup-teacher-temp-epochs', default=0, type=int, help='number of epochs for the teacher temperature to reach the final value')
    parser.add_argument('--student-temp', default=0.1, type=float, help='student temperature')
    parser.add_argument('--center-momentum', default=0.9, type=float, help='momentum for the center')
    parser.add_argument('--group-loss-weight', default=0.5, type=float, help='balancing weight of the grouping loss')
    parser.add_argument('--head-type', type=str, default='early_return', help='choose head type')
    parser.add_argument('--drop-path-rate', type=float, default=0.1, help="stochastic depth rate")
    parser.add_argument('--use-sinkhorn', type=bool_flag, default=False, help='use sinkhorn knopp for computing teacher logits')
    parser.add_argument('--onehot-slot', type=bool_flag, default=False, help='use ont-hot pooling on slots')

    # SlotMIM
    parser.add_argument('--pred-ratio', default=0.3, type=float, nargs='+', help="""Ratio of partial prediction.
        If a list of ratio is specified, one of them will be randomly choosed for each patch.""")
    parser.add_argument('--pred-ratio-var', default=0, type=float, nargs='+', help="""Variance of partial prediction
        ratio. Length should be indentical to the length of pred_ratio. 0 for disabling. """)
    parser.add_argument('--pred-shape', default='block', type=str, help="""Shape of partial prediction.""")
    parser.add_argument('--pred-start-epoch', default=0, type=int, help="""Start epoch to perform masked
        image prediction. We typically set this to 50 for swin transformer. (Default: 0)""")
    
    parser.add_argument('--use-cross-patch-loss', type=bool_flag, default=True, help='use cross-patch loss')
    parser.add_argument('--use-masked-patch-loss', type=bool_flag, default=True, help='use masked-patch loss')
    parser.add_argument('--mask-loss-weight', type=float, default=1, help='weight of the mask loss')
    parser.add_argument('--cross-loss-weight', default=1, type=float, help='weight of the cross-view loss')

    parser.add_argument('--num-prototypes-slot', type=int, default=8192, help='number of prototypes for slot')
    parser.add_argument('--warmup-teacher-slot-temp', default=0.04, type=float, help='initial temperature for the teacher model')
    parser.add_argument('--teacher-slot-temp', default=0.04, type=float, help='final temperature of the teacher model')
    parser.add_argument('--use-slot-dino', type=bool_flag, default=True, help='apply dino loss on slots, otherwise apply contrastive loss')

    # optim.
    parser.add_argument('--batch-size', type=int, default=512, help='total batch size')
    parser.add_argument('--base-lr', type=float, default=1.0,
                        help='base learning when batch size = 256. final lr is determined by linear scale')
    parser.add_argument('--optimizer', type=str, choices=['sgd', 'adamw'], default='sgd', help='optimizer choice')
    parser.add_argument('--warmup-epoch', type=int, default=5, help='warmup epoch')
    parser.add_argument('--warmup-multiplier', type=int, default=100, help='warmup multiplier')
    parser.add_argument('--min-lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--weight-decay-end', type=float, default=1e-5, help='final weight decay')
    parser.add_argument('--fp16', action='store_true', default=True, help='whether or not to turn on automatic mixed precision')
    parser.add_argument('--start-epoch', type=int, default=1, help='used for resume')
    parser.add_argument('--epochs', type=int, default=800, help='number of training epochs')
    parser.add_argument('--freeze_slots', default=0, type=int, help="""Number of epochs
        during which we keep the slots fixed.""")
    parser.add_argument('--clip-grad', type=float, default=0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    
    # misc
    parser.add_argument('--output-dir', type=str, default='./output', help='output director')
    parser.add_argument('--auto-resume', action='store_true', help='auto resume from current.pth')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to the latest checkpoint')
    parser.add_argument('--print-freq', type=int, default=10, help='print frequency')
    parser.add_argument('--save-freq', type=int, default=50, help='save frequency')
    parser.add_argument('--eval-freq', type=int, default=10, help='eval frequency')
    parser.add_argument('--seed', type=int, help='Random seed.')
    parser.add_argument('--num-workers', type=int, default=8, help='num of workers per GPU to use')
    # viz-related
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--dpi', type=int, default=100)

    parser.add_argument('--compile', action='store_true', help='Whether to enable torchcompile')
    parser.set_defaults(compile=False)

    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local-rank", default=0, type=int, help="Please ignore and do not set this argument.")
    return parser 

def build_model(args):
    encoder = vision_transformer.__dict__[args.arch]
    model = SlotMIM(encoder, args).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    params_groups = get_params_groups(model)
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(params_groups)
    else:
        raise NotImplementedError

    return model, optimizer

def save_checkpoint(args, epoch, model, optimizer, scaler=None, current_only=False, logger=None):
    if logger:
        logger.info('==> Saving...')
    model_state = model.state_dict()
    new_model_state = {}
    for key, value in model_state.items():
        new_key = key.replace('module.', '').replace('_orig_mod.', '')
        new_model_state[new_key] = value
    state = {
        'args': args,
        'model': new_model_state,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    if args.fp16:
        state['scaler'] = scaler.state_dict()
    if not current_only:
        file_name = os.path.join(args.output_dir, f'ckpt_epoch_{epoch}.pth')
        torch.save(state, file_name)
        shutil.copyfile(file_name, os.path.join(args.output_dir, 'current.pth'))
    else:
        file_name = os.path.join(args.output_dir, f'current.pth')
        torch.save(state, file_name)

def load_checkpoint(args, model, optimizer, scaler=None, logger=None):
    if os.path.isfile(args.resume):
        if logger:
            logger.info(f"=> loading checkpoint '{args.resume}'")
        checkpoint = torch.load(args.resume, map_location='cpu')

        args.start_epoch = checkpoint['epoch'] + 1
        model.re_init(args)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        if args.fp16 and 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])

        if logger:
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    else:
        if logger:
            logger.info("=> no checkpoint found at '{}'".format(args.resume)) 

def main(args):
    init_distributed_mode(args)
    cudnn.benchmark = True
    args.batch_size = int(args.batch_size / args.world_size)

    logger = setup_logger(output=args.output_dir,
                          distributed_rank=dist.get_rank(), name="slotmim")

    if dist.get_rank() == 0:
        path = os.path.join(args.output_dir, "config.json")
        with open(path, 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.info("Full config saved to {}".format(path))

        # print args
        logger.info(
            "\n".join("%s: %s" % (k, str(v))
                    for k, v in sorted(dict(vars(args)).items())) 
        )
    if args.seed is not None:
        fix_random_seeds(args.seed)

    # prepare data
    train_transform = CustomDataAugmentation(args.image_size, args.min_scale, return_meta=True)
    train_dataset = ImageFolderMask(args.dataset, args.data_dir, transform=train_transform,
                                    patch_size=int(args.arch[-2:]),
                                    pred_ratio=args.pred_ratio,
                                    pred_ratio_var=args.pred_ratio_var,
                                    pred_aspect_ratio=(0.3, 1/0.3),
                                    pred_shape=args.pred_shape,
                                    pred_start_epoch=args.pred_start_epoch)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), 
        num_workers=args.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    eval_dataset = ImageFolder('COCOval', args.coco_path, eval_transform)

    args.num_instances = len(train_loader.dataset)
    if dist.get_rank() == 0:
        logger.info(f"length of training dataset: {args.num_instances}")

    # create model
    if dist.get_rank() == 0:
        logger.info("=> creating model '{}'".format(args.arch))
    model, optimizer = build_model(args)
    model_without_ddp = model.module

    if dist.get_rank() == 0:
        logger.info(model)

    lr_schedule = cosine_scheduler(
        args.base_lr * (args.batch_size * args.world_size) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(train_loader),
        warmup_epochs=args.warmup_epoch,
    )
    wd_schedule = cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(train_loader),
    )

    # define scaler
    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # optionally resume from a checkpoint
    if args.auto_resume:
        resume_file = os.path.join(args.output_dir, "current.pth")
        if os.path.exists(resume_file):
            if dist.get_rank() == 0:
                logger.info(f'auto resume from {resume_file}')
            args.resume = resume_file
        else:
            if dist.get_rank() == 0:
                logger.info(f'no checkpoint found in {args.output_dir}, ignoring auto resume')

    if args.resume:
        load_checkpoint(args, model_without_ddp, optimizer, scaler, logger)

    if args.compile:
        model_without_ddp.encoder_q = torch.compile(model_without_ddp.encoder_q)
        model_without_ddp.encoder_k = torch.compile(model_without_ddp.encoder_k)
        model_without_ddp.projector_q = torch.compile(model_without_ddp.projector_q)
        model_without_ddp.projector_k = torch.compile(model_without_ddp.projector_k)
        model_without_ddp.grouping_q = torch.compile(model_without_ddp.grouping_q)
        model_without_ddp.grouping_k = torch.compile(model_without_ddp.grouping_k)
        if hasattr(model_without_ddp, 'predictor_slot'):
            model_without_ddp.predictor_slot = torch.compile(model_without_ddp.predictor_slot)

    for epoch in range(args.start_epoch, args.epochs + 1):
        train_loader.sampler.set_epoch(epoch - 1)
        # train for one epoch
        train(train_loader, model, optimizer, scaler, lr_schedule, wd_schedule, epoch, args, logger)

        if dist.get_rank() == 0:
            current_only = not (epoch % args.save_freq == 0 or epoch == args.epochs)
            save_checkpoint(args, epoch, model_without_ddp, optimizer, scaler, current_only, logger)

        if epoch % args.eval_freq == 0:
            if dist.get_rank() == 0:
                eval_coco(eval_dataset, model_without_ddp, epoch, args)
            eval_imnet(model_without_ddp, epoch, logger, args)


def train(train_loader, model, optimizer, scaler, lr_schedule, wd_schedule, epoch, args, logger):
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    patch_loss_cross_meter = AverageMeter()
    patch_loss_masked_meter = AverageMeter()
    slot_loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    slot_meter = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    train_len = len(train_loader)
    epoch = epoch - 1 # 0-indexed
    for it, (crops, coords, flags, masks) in enumerate(train_loader):
        # update weight decay and learning rate according to their schedule
        global_it = len(train_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[global_it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[global_it]

        crops = [crop.cuda(non_blocking=True) for crop in crops]
        coords = [coord.cuda(non_blocking=True) for coord in coords]
        flags = [flag.cuda(non_blocking=True) for flag in flags]
        masks = [mask.cuda(non_blocking=True) for mask in masks]

        # compute output and loss
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            losses = model(crops, coords, flags, masks, epoch)
            
        # Check for NaN losses across all processes
        nan_losses = []
        for name, loss in losses.items():
            if isinstance(loss, torch.Tensor) and torch.isnan(loss).any():
                nan_losses.append(name)
        
        nan_detected = torch.tensor(len(nan_losses) > 0, device=crops[0].device)
        if nan_detected: # print errors on any gpu
            logger.warning(f"NaN loss detected in {nan_losses} at iteration {it}, skipping update")
        
        dist.all_reduce(nan_detected, op=dist.ReduceOp.MAX)
        if nan_detected:
            optimizer.zero_grad()
            continue
        loss = sum(loss for name, loss in losses.items() if 'loss' in name)
        
        optimizer.zero_grad()
        if args.fp16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            param_norm = clip_gradients(model, args.clip_grad)
            cancel_gradients_slots(epoch, model, args.freeze_slots)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            param_norm = clip_gradients(model, args.clip_grad)
            cancel_gradients_slots(epoch, model, args.freeze_slots)
            optimizer.step()

        # avg loss from batch size
        loss_meter.update(loss.item(), crops[0].size(0))
        if 'patch_loss_cross' in losses:
            patch_loss_cross_meter.update(losses['patch_loss_cross'].item(), crops[0].size(0))
        if 'patch_loss_masked' in losses:
            patch_loss_masked_meter.update(losses['patch_loss_masked'].item(), crops[0].size(0))
        slot_loss_meter.update(losses['distill_slot_loss'].item() \
                               if 'distill_slot_loss' in losses else losses['ctr_slot_loss'].item(), crops[0].size(0))

        norm_meter.update(param_norm)
        slot_meter.update(losses['n_slot'].item(), crops[0].size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if it % args.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            etas = batch_time.avg * (train_len - it)
            mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            if dist.get_rank() == 0:
                loss_str = f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})  ' + \
                           f'patch_loss_cross {patch_loss_cross_meter.val:.4f} ({patch_loss_cross_meter.avg:.4f})  ' + \
                           f'patch_loss_masked {patch_loss_masked_meter.val:.4f} ({patch_loss_masked_meter.avg:.4f})  ' + \
                           f'slot_loss {slot_loss_meter.val:.4f} ({slot_loss_meter.avg:.4f})  '
                logger.info(
                    f'Train: [{epoch + 1}/{args.epochs}][{it}/{train_len}]  '
                    f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}  '
                    f'time {batch_time.val:.4f} ({batch_time.avg:.4f})  ' + loss_str + \
                    f'norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})  ' + \
                    f'n_slot {slot_meter.val:.1f} ({slot_meter.avg:.1f})  ' + \
                    f'mem {mem_mb:.0f}M'
                )


def eval_coco(eval_dataset, model, epoch, args):
    model.eval()
    np.random.seed(42)
    dots, idxs = prepare_knn(model, eval_dataset, args)
    slot_idxs = np.random.randint(0, args.num_prototypes, 64)
    
    if not os.path.exists(os.path.join(args.output_dir, 'visualizations')):
        os.mkdir(os.path.join(args.output_dir, 'visualizations'))
    
    save_path = os.path.join(args.output_dir, 'visualizations', 'viz_slots_ep{}.jpg'.format(epoch))
    viz_slots(eval_dataset, dots, idxs, slot_idxs, save_path, args)

def eval_imnet(model, epoch, logger, args, k=20, temperature=0.07):        
    # ============ preparing data ... ============
    dataset_train = ReturnIndexDataset(os.path.join(args.imagenet_path, "train"), transform=eval_transform)
    dataset_val = ReturnIndexDataset(os.path.join(args.imagenet_path, "val"), transform=eval_transform)
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    state_dict = model.encoder_k.state_dict()
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.', '').replace('_orig_mod.', '')
        if new_key == 'norm.weight':
            new_key = 'fc_norm.weight'
        if new_key == 'norm.bias':
            new_key = 'fc_norm.bias'
        new_state_dict[new_key] = value
    encoder = vision_transformer.__dict__[args.arch](head_type='global_pool', num_classes=0)
    encoder.load_state_dict(new_state_dict)
    encoder = encoder.cuda()
    if args.compile:
        encoder = torch.compile(encoder)
    encoder.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    with torch.cuda.amp.autocast():
        train_features = extract_features(encoder, data_loader_train)
    print("Extracting features for val set...")
    with torch.cuda.amp.autocast():
        test_features = extract_features(encoder, data_loader_val)

    if dist.get_rank() == 0:
        train_features = torch.nn.functional.normalize(train_features, dim=1, p=2)
        test_features = torch.nn.functional.normalize(test_features, dim=1, p=2)

    train_labels = torch.tensor([s[-1] for s in dataset_train.samples]).long()
    test_labels = torch.tensor([s[-1] for s in dataset_val.samples]).long()
    if dist.get_rank() == 0:
        train_features = train_features.cuda()
        test_features = test_features.cuda()
        train_labels = train_labels.cuda()
        test_labels = test_labels.cuda()
        print("Features are ready!\nStart the k-NN classification.")
        top1, top5 = knn_classifier(train_features, train_labels,
            test_features, test_labels, k, temperature)
        logger.info(f"[Epoch {epoch}] {k}-NN classifier result: Top1: {top1}, Top5: {top5}")

        with open(os.path.join(args.output_dir, "knn_log.txt"), mode="a", encoding="utf-8") as f:
            f.write(json.dumps({"epoch": epoch, "k": k, "top1": top1, "top5": top5}) + "\n")
    dist.barrier()

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
