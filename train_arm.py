#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
# @author  : 郑祥忠
# @license : (C) Copyright,2016-2020,广州海格星航科技
# @contact : dylenzheng@gmail.com
# @file    : train_arm.py
# @time    : 9/14/20 8:37 AM
# @desc    :
'''
import argparse
import os
import logging
import sys
import itertools
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from net.anchor.prior_box import MatchPrior
from net.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from data.arm_voc import ARMDataset
from net.losses.multibox_loss import MultiboxLoss
from net.config import ssd_config
from data.data_preprocessing import TrainAugmentation, TestTransform

from tensorboardX import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')
parser.add_argument("--dataset",default="/home/zhex/data/arm_device_voc",help = "dataset path")
parser.add_argument('--label_file_path', default="/home/zhex/data/arm_device_voc/labels.txt",help='label path')

parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")
parser.add_argument('--net_name', default="ssdlite_mobilenet_v2", help="Model name.")
parser.add_argument('--freeze_base_net', action='store_true',default=False,
                    help="Freeze base net layers.")
parser.add_argument('--freeze_net', action='store_true',default=False,
                    help="Freeze all the layers except the prediction head.")

parser.add_argument('--mb2_width_mult', default=1.0, type=float,
                    help='Width Multiplifier for MobilenetV2')

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--base_net_lr', default=None, type=float,
                    help='initial learning rate for base net.')
parser.add_argument('--extra_layers_lr', default=None, type=float,
                    help='initial learning rate for the layers not in base net and prediction heads.')


# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net',
                    help='Pretrained base model')
parser.add_argument('--pretrained_ssd', help='Pre-trained base model')
parser.add_argument('--resume', default="models/70_2.1875808318456014.pth", type=str,
                    help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="cosine", type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="1000", type=str,
                    help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t_max', default=120, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=200, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=10, type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps', default=10, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--gpus', default="0", type=str,
                    help='Specify the gpu number')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")


def train(loader, net, criterion, optimizer, device, debug_steps=2, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    i = 0
    for images, boxes, labels in loader:
        # if len(labels)<=1:
        #     continue
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()

        optimizer.step()
        i += 1
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i, len(loader)}, " +
                f"Average Loss: {avg_loss:.4f}, " +
                f"Average Regression Loss {avg_reg_loss:.4f}, " +
                f"Average Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        num += 1
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


def load_model(net, optimizer, last_epoch, best_mAP):
    # if args.resume:
    #     ckpt = torch.load(args.resume)
    #     net.load_state_dict(ckpt['model'], False)
    #     last_epoch = ckpt['epoch']
    #     optimizer.load_state_dict(ckpt['optimizer'])
    #     best_mAP = ckpt['best_mAP']
    #     del ckpt
    #     logging.info(f"Resume from the model {args.resume}")

    if args.resume:
        net.load_state_dict(torch.load(args.resume))
        logging.info("Resume from the model {}".format(args.resume))
        print("Resume from the model {}".format(args.resume))

    elif args.base_net:
        net.init_from_base_net(args.base_net)
        logging.info(f"Init from base net {args.base_net}")

    elif args.pretrained_ssd:
        net.init_from_pretrained_ssd(args.pretrained_ssd)
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")


    return net, optimizer, last_epoch, best_mAP


if __name__ == '__main__':
    create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult)
    config = ssd_config

    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)
    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std)

    logging.info("Prepare training dataset.")
    train_dataset = ARMDataset(args.dataset,is_train=True,transform=train_transform,target_transform=target_transform)
    val_dataset = ARMDataset(args.dataset,is_train=False,transform=test_transform,target_transform=target_transform)
    train_loader = DataLoader(train_dataset, args.batch_size,num_workers=args.num_workers,shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset, args.batch_size,num_workers=args.num_workers,shuffle=False,drop_last=True)

    num_class = 2
    logging.info("Build network.")
    net = create_net(num_class)
    min_loss = -10000.0
    last_epoch = -1
    best_mAP = 0

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr

    params = [
        {'params': net.base_net.parameters(), 'lr': base_net_lr},
        {'params': itertools.chain(
            net.source_layer_add_ons.parameters(),
            net.extras.parameters()
        ), 'lr': extra_layers_lr},
        {'params': itertools.chain(
            net.regression_headers.parameters(),
            net.classification_headers.parameters()
        )}
    ]

    # first init and to cuda
    net.init()
    net.to(DEVICE)
    # optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,weight_decay=args.weight_decay)
    optimizer = torch.optim.Adam(params, lr=args.lr,weight_decay=args.weight_decay)
    net, optimizer, last_epoch, best_mAP = load_model(net, optimizer, last_epoch, best_mAP)
    # net.load_state_dict(torch.load(args.resume))
    logging.info("load pre_model successful !!!")
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    if args.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,gamma=args.gamma, last_epoch=last_epoch)

    elif args.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)

    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    last_epoch += 1
    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in tqdm(range(last_epoch, args.num_epochs)):
        train(train_loader, net, criterion, optimizer,device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
        if epoch == 0:
            continue

        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs:
            print("[INFO] starting evaluation...")
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            torch.save(net.state_dict(),"models" + "/" + "{}_{}.pth".format(epoch+70,val_loss))