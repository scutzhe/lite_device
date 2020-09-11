import torch
from torchvision.models.shufflenetv2 import ShuffleNetV2,shufflenet_v2_x0_5
from torch.nn import init
import torch.nn as nn


def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.constant_(m.bias.data, 0.1)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


if __name__ == '__main__':
    model = shufflenet_v2_x0_5()
    weigth_init(model)
    torch.save({'model': model.state_dict()}, 'shuffleNetV2_torchvision_x05.pth')
    print('model saved')



