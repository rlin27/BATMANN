import math
from collections import OrderedDict
import torch
import torch.nn as nn
from quant.XNOR_module import *
from quant.RBNN_modules import *


class Controller(nn.Module):
    """ The CNN as a controller for the MANN architecture. """

    def __init__(self, num_in_channels=3, feature_dim=512, quant='No', rotation_update=1, a32=1):
        super(Controller, self).__init__()
        self.rotation_update = rotation_update
        self.a32 = a32

        # Define the network in the full-precision version
        if quant == 'No':
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

        # Define the network in the binary version (XNOR)
        if quant == 'XNOR':
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

        # Define the network in the binary version (XNOR) with the binary last FC
        if quant == 'XNOR_binary_fc':
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
            self.add_module('fc1', binary_last_fc(2048, feature_dim))

            # Initialize weights
            self._init_weights()

        # Define the network in the binary version (RBNN)
        if quant == 'RBNN':
            # CONV layer
            self.features = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(num_in_channels, 128, 5)),
                ('relu1', nn.ReLU()),
                ('conv2', BinarizeConv2d(rotation_update=self.rotation_update, a32=self.a32, in_channels=128, out_channels=128, kernel_size=5)),
                ('relu2', nn.ReLU()),
                ('maxpool1', nn.MaxPool2d(2, stride=2)),
                ('conv3', BinarizeConv2d(rotation_update=self.rotation_update, a32=self.a32, in_channels=128, out_channels=128, kernel_size=3)),
                ('relu3', nn.ReLU()),
                ('conv4', BinarizeConv2d(rotation_update=self.rotation_update, a32=self.a32, in_channels=128, out_channels=128, kernel_size=3)),
                ('relu4', nn.ReLU()),
                ('maxpool2', nn.MaxPool2d(2, stride=2)),
            ]))

            # Last FC layer
            self.add_module('fc1', nn.Linear(2048, feature_dim))

            # Initialize weights
            self._init_weights()

        # Define the network in the binary version (RBNN) with the binary last FC
        if quant == 'RBNN_binary_fc':
            # CONV layer
            self.features = nn.Sequential(OrderedDict([
                ('conv1', nn.Conv2d(num_in_channels, 128, 5)),
                ('relu1', nn.ReLU()),
                ('conv2', BinarizeConv2d(rotation_update=self.rotation_update, a32=self.a32, in_channels=128, out_channels=128, kernel_size=5)),
                ('relu2', nn.ReLU()),
                ('maxpool1', nn.MaxPool2d(2, stride=2)),
                ('conv3', BinarizeConv2d(rotation_update=self.rotation_update, a32=self.a32, in_channels=128, out_channels=128, kernel_size=3)),
                ('relu3', nn.ReLU()),
                ('conv4', BinarizeConv2d(rotation_update=self.rotation_update, a32=self.a32, in_channels=128, out_channels=128, kernel_size=3)),
                ('relu4', nn.ReLU()),
                ('maxpool2', nn.MaxPool2d(2, stride=2)),
            ]))

            # Last FC layer
            self.add_module('fc1', Binarize_last_fc(rotation_update=self.rotation_update, a32=self.a32, in_features=2048, out_features=feature_dim, bias=True))

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
