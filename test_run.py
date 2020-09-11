# -*- coding: utf-8 -*-
"""
@author  : 秦丹峰
@contact : daneven.jim@gmail.com
@time    : 2020/8/8 17:36
@file    : test_run.py
@desc    : 
"""
import torch


def test_opt():
    # path = "saved_model/mb2_ssd_lite_small_sample_sgd/last.pth"
    path = "saved_model/mb2_ssd_lite_small_sample_tmax_300/last.pth"
    chkpt = torch.load(path)
    print(chkpt['optimizer'].keys())
    print(len(chkpt['optimizer']["state"][140157027935648]))
    for groups in chkpt['optimizer']['param_groups']:
        print(groups["lr"])

    opt = {"state": {}, "param_groups": chkpt['optimizer']['param_groups']}
    for k, y in chkpt['optimizer']["state"].items():
        opt["state"][k] = {"momentum_buffer": y["momentum_buffer"].to("cpu")}
        # print(y)
        # print(k)
        print(opt["state"][k])
        exit()

def opt():
    from net.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
    import itertools

    create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=1.0)
    net = create_net(1)
    params = [
        {'params': net.base_net.parameters(), 'lr': 0.1},
        {'params': itertools.chain(
            net.source_layer_add_ons.parameters(),
            net.extras.parameters()
        ), 'lr': 0.1},
        {'params': itertools.chain(
            net.regression_headers.parameters(),
            net.classification_headers.parameters()
        )}]
    net.init()
    optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9,
                                weight_decay=5e-4)

    for i in range(10):
        optimizer.step()

    chkpt = {
             'optimizer': optimizer.state_dict()
             }
    torch.save(chkpt, "test_opt.pth")

    ckpt = torch.load("test_opt.pth")
    optimizer.load_state_dict(ckpt['optimizer'])
    for i in range(10):
        optimizer.step()


if __name__ == '__main__':
    # test_opt()

    opt()

