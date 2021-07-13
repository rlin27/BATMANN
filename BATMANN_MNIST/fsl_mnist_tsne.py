import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, Function
from torch.optim.lr_scheduler import StepLR
import torch.backends.cudnn as cudnn
import numpy as np
import os
import logging.config
import math
import argparse
import random
import time, datetime
import os
import shutil

from utils.mann import *
from utils.mann_approx import *
from model.controller import *
from quant.XNOR_module import *

from data.preprocess_mnist import *
from data.data_loading import *

cudnn.benchmark = True
cudnn.enabled = True

controller = Controller(num_in_channels=1, feature_dim=512, quant='XNOR_binary_fc').cuda()
device_id = []
for i in range(3 // 2):
    device_id.append(i)
controller = nn.DataParallel(controller, device_ids=device_id).cuda()

controller = nn.DataParallel(controller, device_ids=[2,3]).cuda()
ckpt = torch.load(os.path.join("/home/rlin/BATMANN/BATMANN_MNIST/log_0713/exp1/", 'checkpoint.path.tar'))
controller.load_state_dict(ckpt['state_dict'])

mnist_testset = datasets.MNIST(root='./mnist', train=False, download=True, transform=None)
testing_image = mnist_testset.data
testing_image = testing_image.unsqueeze(1)
testing_target = mnist_testset.targets

features = controller(testing_image)
print(features.shape)
