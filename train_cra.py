import argparse
import os
import datetime
import logging
import time
import math
import numpy as np
from collections import OrderedDict

import torch
import torch.nn.functional as F

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_model, build_adversarial_discriminator, build_feature_extractor, build_classifier
from core.solver import adjust_learning_rate
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger
from core.utils.cra_utils import prob_2_entropy, pickle_dump, pickle_load


def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    loss = -soft_label.float()*F.log_softmax(pred, dim=1)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))

def soft_label_cross_entropy_si(pred, soft_label, mask, pixel_weights=None):
    N, C, H, W = pred.shape
    loss = -soft_label.float() * F.log_softmax(pred, dim=1) * mask.to('cuda')
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))

def train(cfg, local_rank, distributed):
    
    if cfg.ENT_NORM: 
        pf = pickle_load(os.path.join(cfg.DATASETS.SOURCE_LABEL_DIR, 'pixel_per_cls.pkl'))
        pix_sum = np.asarray([t for t in pf.values()]).sum()
        rc_norm = {}
        for key in range(cfg.MODEL.NUM_CLASSES):
            rc_norm[key] = cfg.ENT_NORM_FACTOR if((pf[key] / pix_sum) < cfg.ENT_NORM_PCT) else 1

    logger = logging.getLogger("FADA.trainer")
    
    train_data = build_dataset(cfg, mode='train', is_source=True)

    feature_extractor = build_feature_extractor(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    feature_extractor.to(device)
    
    classifier = build_classifier(cfg)
    classifier.to(device)

    model_D = build_adversarial_discriminator(cfg)
    model_D.to(device)

#     logger.info("Loading checkpoint from {}".format(cfg.resume))
#     checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
#     model_weights = checkpoint['feature_extractor'] if distributed else strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
#     feature_extractor.load_state_dict(model_weights)
#     classifier_weights = checkpoint['classifier'] if distributed else strip_prefix_if_present(checkpoint['classifier'], 'module.')
#     classifier.load_state_dict(classifier_weights)

    if local_rank==0:
        print(feature_extractor)
        print(model_D)

    batch_size = cfg.SOLVER.BATCH_SIZE//2
    if distributed:
        pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        
        batch_size = int(cfg.SOLVER.BATCH_SIZE / torch.distributed.get_world_size())//2
        if not cfg.MODEL.FREEZE_BN:
            feature_extractor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(feature_extractor)
        feature_extractor = torch.nn.parallel.DistributedDataParallel(
            feature_extractor, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg1
        )
        pg2 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        classifier = torch.nn.parallel.DistributedDataParallel(
            classifier, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg2
        )
        pg3 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        model_D = torch.nn.parallel.DistributedDataParallel(
            model_D, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True, process_group=pg3
        )
        torch.autograd.set_detect_anomaly(True)
        torch.distributed.barrier()
    
    optimizer_fea = torch.optim.SGD(feature_extractor.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_fea.zero_grad()
    
    optimizer_cls = torch.optim.SGD(classifier.parameters(), lr=cfg.SOLVER.BASE_LR*10, momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    optimizer_cls.zero_grad()
    
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=cfg.SOLVER.BASE_LR_D, betas=(0.9, 0.99))
    optimizer_D.zero_grad()

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = local_rank == 0

    start_epoch = 0
    iteration = 0
    
    if cfg.resume == 'none':
        logger.info("No checkpoint")
    else:
        logger.info("Loading checkpoint from {}".format(cfg.resume))
        checkpoint = torch.load(cfg.resume, map_location=torch.device('cpu'))
        model_weights = checkpoint['feature_extractor'] if distributed else strip_prefix_if_present(checkpoint['feature_extractor'], 'module.')
        feature_extractor.load_state_dict(model_weights)
        classifier_weights = checkpoint['classifier'] if distributed else strip_prefix_if_present(checkpoint['classifier'], 'module.')
        classifier.load_state_dict(classifier_weights, strict=False)
#     if "model_D" in checkpoint:
#         logger.info("Loading model_D from {}".format(cfg.resume))
#         model_D_weights = checkpoint['model_D'] if distributed else strip_prefix_if_present(checkpoint['model_D'], 'module.')
#         model_D.load_state_dict(model_D_weights)
    # if "optimizer_fea" in checkpoint:
    #     logger.info("Loading optimizer_fea from {}".format(cfg.resume))
    #     optimizer_fea.load_state_dict(checkpoint['optimizer_fea'])
    # if "optimizer_cls" in checkpoint:
    #     logger.info("Loading optimizer_cls from {}".format(cfg.resume))
    #     optimizer_cls.load_state_dict(checkpoint['optimizer_cls'])
    # if "optimizer_D" in checkpoint:
    #     logger.info("Loading optimizer_D from {}".format(cfg.resume))
    #     optimizer_D.load_state_dict(checkpoint['optimizer_D'])
    # if "iteration" in checkpoint:
    #     iteration = checkpoint['iteration']

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=(train_sampler is None), num_workers=4, pin_memory=True, sampler=train_sampler, drop_last=True)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)

    max_iters = cfg.SOLVER.MAX_ITER
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    feature_extractor.train()
    classifier.train()
    model_D.train()
    start_training_time = time.time()
    end = time.time()
    
    for i, (input, label, entropy, name) in enumerate(train_loader):
#             torch.distributed.barrier()
        data_time = time.time() - end

        current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, max_iters, power=cfg.SOLVER.LR_POWER)
        current_lr_D = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR_D, iteration, max_iters, power=cfg.SOLVER.LR_POWER)
        for index in range(len(optimizer_fea.param_groups)):
            optimizer_fea.param_groups[index]['lr'] = current_lr
        for index in range(len(optimizer_cls.param_groups)):
            optimizer_cls.param_groups[index]['lr'] = current_lr*10
        for index in range(len(optimizer_D.param_groups)):
            optimizer_D.param_groups[index]['lr'] = current_lr_D
            
#       torch.distributed.barrier()

        # nromalizator for entropy
        if cfg.ENT_NORM:
            for cls in range(cfg.MODEL.NUM_CLASSES):
                if (cls in rc_norm.keys()) and (cls in label):
                    entropy[label==cls] = entropy[label==cls] * rc_norm[cls]

        optimizer_fea.zero_grad()
        optimizer_cls.zero_grad()
        optimizer_D.zero_grad()
        src_input = input.cuda(non_blocking=True)
        tgt_input = input.cuda(non_blocking=True)
        label = label.cuda(non_blocking=True).long()
        p_label = label.detach()
        
        p_label[entropy > cfg.ENT_THRESH] = 255
        p_label[entropy == -1] = 255
            
        src_size = src_input.shape[-2:]
        tgt_size = tgt_input.shape[-2:]
        
        # train seg net
        src_fea = feature_extractor(src_input)
        src_pred = classifier(src_fea, src_size)
        temperature = 1.8
        src_pred = src_pred.div(temperature)
        loss_seg = criterion(src_pred, p_label)
        loss_seg.backward()
        # torch.distributed.barrier()
        
        # generate src_soft_label and mask_src
        src_soft_label = F.softmax(src_pred, dim=1).detach()
        src_soft_label[src_soft_label>0.9] = 0.9
        entropy = entropy.unsqueeze(dim=1)
        src_entropy = entropy.repeat(1, cfg.MODEL.NUM_CLASSES, 1, 1)

        mask_src = np.ones((src_entropy.shape))
        mask_src[src_entropy > cfg.ENT_THRESH] = 0
        mask_src[src_entropy == -1] = 0
        mask_src = torch.from_numpy(mask_src)
        mask_src = mask_src.repeat(1, 2, 1, 1)
        
        # generate tgt_soft_label and mask_tgt
        tgt_fea = feature_extractor(tgt_input)
        tgt_pred = classifier(tgt_fea, tgt_size)
        tgt_pred = tgt_pred.div(temperature)

        tgt_soft_label = F.softmax(tgt_pred, dim=1).detach()
        tgt_soft_label[tgt_soft_label>0.9] = 0.9
        tgt_entropy = src_entropy.detach()
        
        mask_tgt = np.ones((tgt_entropy.shape))
        mask_tgt[tgt_entropy <= cfg.ENT_THRESH] = 0
        mask_tgt[tgt_entropy == -1] = 0
        mask_tgt = torch.from_numpy(mask_tgt)
        mask_tgt = mask_tgt.repeat(1, 2, 1, 1)

        tgt_D_pred = model_D(tgt_fea, tgt_size)
        loss_adv_tgt = 0.001*soft_label_cross_entropy_si(tgt_D_pred, torch.cat((tgt_soft_label, torch.zeros_like(tgt_soft_label)), dim=1), mask_tgt)
        loss_adv_tgt.backward()

        optimizer_fea.step()
        optimizer_cls.step()

        optimizer_D.zero_grad()
        # torch.distributed.barrier()
        
        src_D_pred = model_D(src_fea.detach(), src_size)
        loss_D_src = 0.5*soft_label_cross_entropy_si(src_D_pred, torch.cat((src_soft_label, torch.zeros_like(src_soft_label)), dim=1), mask_src)
        loss_D_src.backward()

        tgt_D_pred = model_D(tgt_fea.detach(), tgt_size)
        loss_D_tgt = 0.5*soft_label_cross_entropy_si(tgt_D_pred, torch.cat((torch.zeros_like(tgt_soft_label), tgt_soft_label), dim=1), mask_tgt)
        loss_D_tgt.backward()

        # torch.distributed.barrier()

        optimizer_D.step()
            
        meters.update(loss_seg=loss_seg.item())
        meters.update(loss_adv_tgt=loss_adv_tgt.item())
        meters.update(loss_D=(loss_D_src.item()+loss_D_tgt.item()))
        meters.update(loss_D_src=loss_D_src.item())
        meters.update(loss_D_tgt=loss_D_tgt.item())

        iteration = iteration + 1

        n = src_input.size(0)

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (cfg.SOLVER.STOP_ITER - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        if iteration % 20 == 0 or iteration == max_iters:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer_fea.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
                
        if (iteration == cfg.SOLVER.MAX_ITER or iteration % cfg.SOLVER.CHECKPOINT_PERIOD==0) and save_to_disk:
            filename = os.path.join(output_dir, "model_iter{:06d}.pth".format(iteration))
            torch.save({'iteration': iteration, 'feature_extractor': feature_extractor.state_dict(), 'classifier':classifier.state_dict(), 'model_D': model_D.state_dict(), 'optimizer_fea': optimizer_fea.state_dict(), 'optimizer_cls': optimizer_cls.state_dict(), 'optimizer_D': optimizer_D.state_dict()}, filename)

        if iteration == cfg.SOLVER.MAX_ITER:
            break
        if iteration == cfg.SOLVER.STOP_ITER:
            break
            
    
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (cfg.SOLVER.MAX_ITER)
        )
    )

    return feature_extractor, classifier          


def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
    parser.add_argument("-cfg",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("FADA", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    fea, cls = train(cfg, args.local_rank, args.distributed)
    

if __name__ == "__main__":
    main()
