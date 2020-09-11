import torch
import torch.nn as nn

from torchvision.models import shufflenet_v2_x0_5

if __name__ == '__main__':
    net = shufflenet_v2_x0_5()
    print(net)