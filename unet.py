import torch.nn as nn
from unet_components import DoubleConv, DownSample, UpSample

class Unet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Unet, self).__init__()
        # The down sampling
        self.down_conv1 = DownSample(in_channels=in_channels, out_channels=64)
        self.down_conv3 = DownSample(in_channels=128, out_channels=256)
        self.down_conv4 = DownSample(in_channels=256, out_channels=512)

        # The bottleneck - just two convolutions
        self.bottleneck = DoubleConv(in_channels=512, out_channels=1024)

        # The Up sampling
        self.up_conv1 = UpSample(in_channels=1024, out_channels=512)
        self.up_conv2 = UpSample(in_channels=512, out_channels=256)
        self.up_conv3 = UpSample(in_channels=256, out_channels=128)
        self.up_conv4 = UpSample(in_channels=128, out_channels=64)

        # Output layer
        self.output_layer = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        down_1, pool_1 = self.down_conv1(x)
        down_2, pool_2 = self.down_conv2(pool_1)
        down_3, pool_3 = self.down_conv3(pool_2)
        down_4, pool_4 = self.down_conv4(pool_3)

        bottleneck = self.bottleneck(pool_4)

        up_1 = self.up_conv1(bottleneck, down_4)
        up_2 = self.up_conv2(up_1, down_3)
        up_3 = self.up_conv3(up_2, down_2)
        up_4 = self.up_conv4(up_3, down_1)

        # Output layer
        out = self.output_layer(up_4)
        out = self.relu(out)

        return out



