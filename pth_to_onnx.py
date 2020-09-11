import torch
# from net.backbone import shuffleNet_v2,mobilenet_v2,shuffleNet_v1
from torchvision.models.shufflenetv2 import ShuffleNetV2,shufflenet_v2_x0_5
from net.mobilenet_v2_ssd_lite_w_h import *
# from net.shufflenet_v2_ssdlite import *
from net.shufflenet_v2_ssdlite_w_h import *
# pth_dir = '/home/zhanjinhao/codes/车辆检测/ssdlite_mobilenet_v2/pth/shuffleNetV2_torchvision_x05.pth'
# pth_dir = '/home/zhanjinhao/codes/车辆检测/ssdlite_mobilenet_v2/net/mobilenet_v2_0125_ssdlite.pth'
# pth_dir = '/home/zhanjinhao/codes/车辆检测/ssdlite_mobilenet_v2/net/mobilenet_v2_0125_5_branch_ssdlite.pth'
# pth_dir = '/home/zhanjinhao/codes/车辆检测/ssdlite_mobilenet_v2/net/shufflenet_v2_0125_b5_ssdlite_full.pth'
# pth_dir = '/home/zhanjinhao/codes/车辆检测/ssdlite_mobilenet_v2/net/mobilenet_v2_0125_ssdlite_384_192_simple.pth'
# pth_dir = '/home/zhanjinhao/codes/车辆检测/ssdlite_mobilenet_v2/net/shufflenet_5b_224.pth'
pth_dir = '/home/zhanjinhao/codes/车辆检测/ssdlite_mobilenet_v2/net/shufflenet_v2_ssdlite_w_h.pth'
# pth_dir = '/home/zhanjinhao/codes/车辆检测/ssdlite_mobilenet_v2/saved_model/ori_mbn_v2_ssdlite/best.pth'


# net = create_mobilenetv2_ssd_lite(1,1.0)
net = create_shufflenetv2_ssd_lite(1)
# net.load_state_dict(torch.load(pth_dir))
net.load_state_dict(torch.load(pth_dir)['net'])
net.eval()
net.cuda()
output_onnx = '/home/zhanjinhao/codes/车辆检测/ssdlite_mobilenet_v2/onnx/shufflenet_v2_ssdlite_w_h.onnx'
print("==> Exporting model to ONNX format at '{}'".format(output_onnx))
input_names = ["input"]
# output_names = ["classifier"]
output_names = ["scores","boxes"]
inputs = torch.randn(1, 3, 384, 192).cuda()
# inputs = torch.randn(1, 3, 384, 192).cuda()
torch_out = torch.onnx._export(net, inputs, output_onnx, export_params=True, verbose=False,
                               input_names=input_names, output_names=output_names)
print('onnx model converted!')