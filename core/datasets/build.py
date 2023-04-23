import os
import torch
from . import transform
from . import transform_cra
from .dataset_path_catalog import DatasetCatalog

def build_transform(cfg, mode, is_source):
    if mode=="train":
        w, h = cfg.INPUT.SOURCE_INPUT_SIZE_TRAIN if is_source else cfg.INPUT.TARGET_INPUT_SIZE_TRAIN
        trans_list = [
            transform.ToTensor(),
            transform.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
        ]
        if cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN > 0:
            trans_list = [transform.RandomHorizontalFlip(p=cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN),] + trans_list
        if cfg.INPUT.INPUT_SCALES_TRAIN[0]==cfg.INPUT.INPUT_SCALES_TRAIN[1] and cfg.INPUT.INPUT_SCALES_TRAIN[0]==1:
            trans_list = [transform.Resize((h, w)),] + trans_list
        else:
            trans_list = [
                transform.RandomScale(scale=cfg.INPUT.INPUT_SCALES_TRAIN),
                transform.RandomCrop(size=(h, w), pad_if_needed=True),
            ] + trans_list
        if is_source:
            trans_list = [
                transform.ColorJitter(
                    brightness=cfg.INPUT.BRIGHTNESS,
                    contrast=cfg.INPUT.CONTRAST,
                    saturation=cfg.INPUT.SATURATION,
                    hue=cfg.INPUT.HUE,
                ),
            ] + trans_list
        trans = transform.Compose(trans_list)
    else:
        w, h = cfg.INPUT.INPUT_SIZE_TEST
        trans = transform.Compose([
            transform.Resize((h, w), resize_label=False),
            transform.ToTensor(),
            transform.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
        ])
    return trans

def build_transform_cra(cfg, mode, is_source):
    if mode=="train":
        w, h = cfg.INPUT.SOURCE_INPUT_SIZE_TRAIN if is_source else cfg.INPUT.TARGET_INPUT_SIZE_TRAIN
        trans_list = [
            transform_cra.ToTensor(),
            transform_cra.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
        ]
        if cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN > 0:
            trans_list = [transform_cra.RandomHorizontalFlip(p=cfg.INPUT.HORIZONTAL_FLIP_PROB_TRAIN),] + trans_list
        if cfg.INPUT.INPUT_SCALES_TRAIN[0]==cfg.INPUT.INPUT_SCALES_TRAIN[1] and cfg.INPUT.INPUT_SCALES_TRAIN[0]==1:
            trans_list = [transform_cra.Resize((h, w)),] + trans_list
        else:
            trans_list = [
                transform_cra.RandomScale(scale=cfg.INPUT.INPUT_SCALES_TRAIN),
                transform_cra.RandomCrop(size=(h, w), pad_if_needed=True),
            ] + trans_list
        if is_source:
            trans_list = [
                transform_cra.ColorJitter(
                    brightness=cfg.INPUT.BRIGHTNESS,
                    contrast=cfg.INPUT.CONTRAST,
                    saturation=cfg.INPUT.SATURATION,
                    hue=cfg.INPUT.HUE,
                ),
            ] + trans_list
        trans = transform_cra.Compose(trans_list)
    else:
        w, h = cfg.INPUT.INPUT_SIZE_TEST
        trans = transform_cra.Compose([
            transform_cra.Resize((h, w), resize_label=False),
            transform_cra.ToTensor(),
            transform_cra.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=cfg.INPUT.TO_BGR255)
        ])
    return trans

def build_dataset(cfg, mode='train', is_source=True, epochwise=False):
    assert mode in ['train', 'val', 'test']
    if cfg.DATASETS.FLAG == "cra":
        transform = build_transform_cra(cfg, mode, is_source)
    elif cfg.DATASETS.FLAG == "sd":
        transform = build_transform(cfg, mode, is_source)
    else:
        transform = build_transform(cfg, mode, is_source)
    iters = None
    if mode=='train':
        if not epochwise:
            iters = cfg.SOLVER.MAX_ITER*cfg.SOLVER.BATCH_SIZE
        if is_source:
            dataset = DatasetCatalog.get(cfg.DATASETS.SOURCE_TRAIN, mode, num_classes=cfg.MODEL.NUM_CLASSES, max_iters=iters, transform=transform, label_dir=cfg.DATASETS.SOURCE_LABEL_DIR, flag=cfg.DATASETS.FLAG)
        else:
            dataset = DatasetCatalog.get(cfg.DATASETS.TARGET_TRAIN, mode, num_classes=cfg.MODEL.NUM_CLASSES, max_iters=iters, transform=transform)
    elif mode=='val':
        dataset = DatasetCatalog.get(cfg.DATASETS.TEST, 'val', num_classes=cfg.MODEL.NUM_CLASSES, max_iters=iters, transform=transform)
    elif mode=='test':
        dataset = DatasetCatalog.get(cfg.DATASETS.TEST, cfg.DATASETS.TEST.split('_')[-1], num_classes=cfg.MODEL.NUM_CLASSES, max_iters=iters, transform=transform)
    return dataset
