import torch
import torch.nn as nn

archtecture_config = [        
        # Tuple (kernel_size, output_size, stride, padding)
        (7, 64, 2, 3),
        "M",  # Maxpool
        
        (3, 192, 1, 1),
        "M",
        
        (1, 128, 1, 0),
        (3, 256, 1, 1),
        (1, 256, 1, 0),
        (3, 512, 1, 1),
        "M",

        # List repeat serveral iterations
        [(1, 256, 1, 0), (3, 512, 1, 1), 4],
        (1, 512, 1, 0),
        (3, 1024, 1, 1),
        "M",

        [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
        (3, 1024, 1, 1),
        (3, 1024, 2, 1),   # the conv-20 uses stride 2 to downsample
        
        (3, 1024, 1, 1),
        (3, 1024, 1, 1)
]


class CNNBlock(nn.Module):
    '''Helper function
       In the original YOLOv1 paper, it does not use the Batch normalization, but we will implement here
       In the original YOLOv1 paper, it uses LeakyRelu, instead of the normal ReLU
    '''
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
    
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)  # turn off the learnable bias due to we will add Batch norm y = r x + B, the B has same effect as bias (https://discuss.pytorch.org/t/any-purpose-to-set-bias-false-in-densenet-torchvision/22067/2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyRelu(0.1)

    def forward(self, x):
        return self.leakyrelu(self.bn(self.conv(x)))


class YOLOv1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(YOLOv1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels,
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

        
    def _create_conv_layers(self, architecture)
