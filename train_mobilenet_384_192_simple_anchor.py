# -*- coding: utf-8 -*-
"""
@author  : 秦丹峰
@contact : daneven.jim@gmail.com
@time    : 2020/8/4 22:52
@file    : train.py
@desc    : 训练代码
"""
import argparse
import os
import logging
import sys
import itertools
import shutil

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from net.anchor.prior_box import MatchPrior
from net.mobilenet_v2_ssd_lite_w_h_simple_anchor import create_mobilenetv2_ssd_lite
from data.voc_dataset import VOCDataset
from net.losses.multibox_loss import MultiboxLoss
from net.config import mobilenet_ssd_384_192_simple_anchor_config
from data.data_preprocessing import TrainAugmentation_w_h, TestTransform_w_h
from tensorboardX import SummaryWriter
from eval_simple import get_map
import datetime

os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
os.environ["OMP_NUM_THREADS"] = "1"

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')

parser.add_argument("--dataset_type", default="voc", type=str,
                    help='Specify dataset type. Currently support voc and open_images.')

parser.add_argument('--datasets', nargs='+', help='Dataset directory path')
parser.add_argument('--validation_dataset', help='Dataset directory path')
parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")
parser.add_argument('--use_small_sample', action='store_true',
                    help="Whether to use small sample training")

parser.add_argument('--net_name', default="ssdlite_mobilenet_v2_default", help="Model name.")
parser.add_argument('--freeze_base_net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze_net', action='store_true',
                    help="Freeze all the layers except the prediction head.")

parser.add_argument('--mb2_width_mult', default=0.125, type=float,
                    help='Width Multiplifier for MobilenetV2')

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
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
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

# Scheduler
parser.add_argument('--scheduler', default="multi-step", type=str,
                    help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones', default="1000", type=str,
                    help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t_max', default=120, type=float,
                    help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch_size', default=128, type=int,
                    help='Batch size for training')
parser.add_argument('--num_epochs', default=120, type=int,
                    help='the number epochs')
parser.add_argument('--num_workers', default=2, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs', default=5, type=int,
                    help='the number epochs')
parser.add_argument('--debug_steps', default=10, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use_cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--gpus', default="0", type=str,
                    help='Specify the gpu number')

parser.add_argument('--checkpoint_folder', default='saved_model/',
                    help='Directory for saving checkpoint models')


logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Use Cuda.")


def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    for i, data in enumerate(loader):
        images, boxes, labels = data
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
            writer.add_scalar("train/Loss", avg_loss, epoch*len(loader)+i)
            writer.add_scalar("train/Regression_Loss", avg_reg_loss, epoch*len(loader)+i)
            writer.add_scalar("train/Classification_Loss", avg_clf_loss, epoch*len(loader)+i)
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
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num


def load_model(net, optimizer, last_epoch, best_mAP):
    timer.start("Load Model")
    if args.resume:
        chkpt = torch.load(args.resume)
        net.load_state_dict(chkpt['model'], False)
        last_epoch = chkpt['epoch']
        optimizer.load_state_dict(chkpt['optimizer'])
        best_mAP = chkpt['best_mAP']
        del chkpt
        logging.info(f"Resume from the model {args.resume}")

    elif args.base_net:
        net.init_from_base_net(args.base_net)
        logging.info(f"Init from base net {args.base_net}")

    elif args.pretrained_ssd:
        net.init_from_pretrained_ssd(args.pretrained_ssd)
        logging.info(f"Init from pretrained ssd {args.pretrained_ssd}")

    # elif args.fine_tune:
    #     net.fine_tune(args.pretrained_ssd)
    #     logging.info(f"fine tune from detection head {args.pretrained_ssd}")
    logging.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    return net, optimizer, last_epoch, best_mAP


def save_model(epoch,net, optimizer, best_mAP, save_best=False):
    if not os.path.exists(os.path.join(args.checkpoint_folder, args.net_name)):
        os.mkdir(os.path.join(args.checkpoint_folder, args.net_name))
    model_path = os.path.join(args.checkpoint_folder, args.net_name, f"last.pth")

    chkpt = {'epoch': epoch,
             'model': net.state_dict(),
             'optimizer': optimizer.state_dict(),
             'best_mAP': best_mAP
             }

    if save_best:
        net.save(model_path.replace("last", "best"))
        torch.save(chkpt, model_path.replace("last", "best_opt"))
    if epoch % 20 == 0:
        # net.save(model_path.replace("last", str(epoch)))
        torch.save(chkpt, model_path.replace("last", str(epoch)))
    torch.save(chkpt, model_path)
    logging.info(f"Saved model {model_path}")

    del chkpt


if __name__ == '__main__':
    timer = Timer()

    if not os.path.exists("log/{}".format(args.net_name)):
        os.mkdir("log/{}".format(args.net_name))
    # log_path = "log/{}/{}".format(args.net_name,
    #                         datetime.datetime.now().strftime('%Y_%m_%d_%H'))
    log_path = "log/{}".format(args.net_name)
    writer = SummaryWriter(log_path)

    logging.info(args)
    create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult)
    config = mobilenet_ssd_384_192_simple_anchor_config

    train_transform = TrainAugmentation_w_h(config.image_width,config.image_height, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform_w_h(config.image_width,config.image_height, config.image_mean, config.image_std)

    logging.info("Prepare training datasets.")
    datasets = []
    for dataset_path in args.datasets:
        if args.dataset_type == 'voc':
            dataset = VOCDataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform,
                                 use_small_sample=args.use_small_sample)
            label_file = os.path.join(args.checkpoint_folder, "voc-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
        else:
            raise ValueError(f"Dataset type {args.dataset_type} is not supported.")
        datasets.append(dataset)
    logging.info(f"Stored labels into file {label_file}.")
    train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
    logging.info("Prepare Validation datasets.")
    if args.dataset_type == "voc":
        val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
                                 target_transform=target_transform, is_test=True,
                                 use_small_sample=args.use_small_sample)
        logging.info(val_dataset)
    logging.info("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)
    logging.info("Build network.")
    net = create_net(num_classes)
    min_loss = -10000.0
    last_epoch = -1
    best_mAP = 0

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    if args.freeze_base_net:  # backbone的权重不更新
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args.freeze_net:  # 只更新回归和分类分支的权重
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
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

    net.init()
    net.to(DEVICE)

    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    net, optimizer, last_epoch, best_mAP = load_model(net, optimizer, last_epoch, best_mAP)

    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    logging.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    if args.scheduler == 'multi-step':
        logging.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                                     gamma=args.gamma, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logging.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    else:
        logging.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    last_epoch += 1
    logging.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch, args.num_epochs):
        scheduler.step()
        lr = scheduler.get_lr()
        if len(lr) == 1:
            writer.add_scalar("learn_rate/pred_heads_lr", lr[0], epoch)
        elif len(lr) == 2:
            writer.add_scalar("learn_rate/ssd_lr", lr[0], epoch)
            writer.add_scalar("learn_rate/pred_heads_lr", lr[1], epoch)
        else:
            writer.add_scalar("learn_rate/mob_lr", lr[0], epoch)
            writer.add_scalar("learn_rate/ssd_lr", lr[1], epoch)
            writer.add_scalar("learn_rate/pred_heads_lr", lr[2], epoch)

        train(train_loader, net, criterion, optimizer,
              device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)

        if epoch == 0:
            continue

        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs:
            print("[INFO] 开始评估")
            val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )
            writer.add_scalar("validation/Loss", val_loss, epoch)
            writer.add_scalar("validation/Regression_Loss", val_regression_loss, epoch)
            writer.add_scalar("validation/Classification_Loss", val_classification_loss, epoch)

            map = get_map(net.state_dict(), args.validation_dataset, label_file,width_mult=args.mb2_width_mult)
            writer.add_scalar("mAP", map, epoch)

            if map > best_mAP:
                save_best = True
                best_mAP = map
            else:
                save_best = False
            save_model(epoch,net, optimizer, best_mAP, save_best)


