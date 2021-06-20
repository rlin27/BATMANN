import math
from collections import OrderedDict
import torch
import torch.nn as nn
from quant.binary_module import *


class Controller(nn.Module):
    """ The CNN as a controller for the MANN architecture. """

    def __init__(self, num_in_channels=3, feature_dim=512, quant=0):
        super(Controller, self).__init__()

        # Define the network in the full-precision version
        if quant == 0:
            # CONV layer
            self.features = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(num_in_channels, 128, 5)),
                ('relu1', nn.ReLU()),
                ('conv2', nn.Conv2d(128, 128, 5)),
                ('relu2', nn.ReLU()),
                ('maxpool1', nn.MaxPool2d(2, stride=2)),
                ('conv3', nn.Conv2d(128, 128, 3)),
                ('relu3', nn.ReLU()),
                ('conv4', nn.Conv2d(128, 128, 3)),
                ('relu4', nn.ReLU()),
                ('maxpool2', nn.MaxPool2d(2, stride=2)),
            ]))
            # Last FC layer
            self.add_module('fc1', nn.Linear(2048, feature_dim))

            # Initialize weights
            self._init_weights()

        # Define the network in the binary version
        if quant == 1:
            # CONV layer
            conv_layer = XNOR_BinarizeConv2d
            self.features = nn.Sequential(OrderedDict([
                ('conv1', first_conv(num_in_channels, 128, 5)),
                ('relu1', nn.ReLU()),
                ('conv2', conv_layer(128, 128, 5)),
                ('relu2', nn.ReLU()),
                ('maxpool1', nn.MaxPool2d(2, stride=2)),
                ('conv3', conv_layer(128, 128, 3)),
                ('relu3', nn.ReLU()),
                ('conv4', conv_layer(128, 128, 3)),
                ('relu4', nn.ReLU()),
                ('maxpool2', nn.MaxPool2d(2, stride=2)),
            ]))

            # Last FC layer
            self.add_module('fc1', last_fc(2048, feature_dim))
            
            # Initialize weights
            self._init_weights()

    def forward(self, x):
        """ Forward pass to generate the feature vectors with required dimension. """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)

        return x

    def _init_weights(self):
        """ Initialize the Controller with weights of Gaussian distribution. """
        torch.manual_seed(806)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data = torch.ones(m.bias.data.size())
