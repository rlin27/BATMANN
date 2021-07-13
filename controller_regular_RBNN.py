import torch
import torch.nn as nn
import numpy as np


class Controller_RBNN(nn.Module):
    """
    This model definition is EXTREMELY redundant, only used for Memotorch evaluation.
    """

    def __init__(self, state_dict='/mnt/nfsdisk/rlin/log_0707/log_ab4/model_best.pth.tar', num_in_channels=1,
                 feature_dim=512):
        super(Controller_RBNN, self).__init__()

        ######################################################
        # load weights and alpha/ from the pretained parameters
        ######################################################
        ckpt = torch.load(state_dict, map_location='cuda:0')
        params = ckpt['state_dict']

        ## conv1
        # parameters
        self.conv1_weight = params['module.features.conv1.weight'].detach().cpu().data
        self.conv1_bias = params['module.features.conv1.bias'].detach().cpu().data

        ## conv2
        # parameters
        self.conv2_weight = params['module.features.conv2.weight'].detach().cpu().data
        self.conv2_bias = params['module.features.conv2.bias'].detach().cpu().data
        # RBNN params
        self.conv2_alpha = params['module.features.conv2.alpha'].detach().data
        self.conv2_rotate = params['module.features.conv2.rotate'].detach().data
        self.conv2_R1 = params['module.features.conv2.R1'].detach().data
        self.conv2_R2 = params['module.features.conv2.R2'].detach().data 

        # conv3
        # parameters
        self.conv3_weight = params['module.features.conv3.weight'].detach().cpu().data
        self.conv3_bias = params['module.features.conv3.bias'].detach().cpu().data
        # RBNN params
        self.conv3_alpha = params['module.features.conv3.alpha'].detach().data
        self.conv3_rotate = params['module.features.conv3.rotate'].detach().data
        self.conv3_R1 = params['module.features.conv3.R1'].detach().data
        self.conv3_R2 = params['module.features.conv3.R2'].detach().data

        # conv4
        # parameters
        self.conv4_weight = params['module.features.conv4.weight'].detach().cpu().data
        self.conv4_bias = params['module.features.conv4.bias'].detach().cpu().data
        # RBNN params
        self.conv4_alpha = params['module.features.conv4.alpha'].detach().data
        self.conv4_rotate = params['module.features.conv4.rotate'].detach().data
        self.conv4_R1 = params['module.features.conv4.R1'].detach().data
        self.conv4_R2 = params['module.features.conv4.R2'].detach().data

        # fc1
        # parameters
        self.fc1_weight = params['module.fc1.weight'].detach().cpu().data
        self.fc1_bias = params['module.fc1.bias'].detach().cpu().data
        # RBNN params
        self.fc1_alpha = params['module.fc1.alpha'].detach().data
        self.fc1_rotate = params['module.fc1.rotate'].detach().data
        self.fc1_R1 = params['module.fc1.R1'].detach().data
        self.fc1_R2 = params['module.fc1.R2'].detach().data

        ######################################################
        # Define different layers
        ######################################################
        self.conv1 = nn.Conv2d(num_in_channels, 128, 5, bias=False)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 128, 5, bias=False)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(128, 128, 3, bias=False)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(128, 128, 3, bias=False)
        self.relu4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(2048, feature_dim, bias=True)

        ######################################################
        # initialize the weights
        ######################################################
        ## conv1
        # get quantized weights for conv1
        self.conv1_weight_q = self.conv1_weight
        # initialize conv1
        self.conv1.weight = nn.Parameter(data=self.conv1_weight_q)
        self.conv1.bias = nn.Parameter(data=self.conv1_bias)

        ## conv2
        # get quantized weights for conv1
        conv2_w = self.conv2_weight
        conv2_w = conv2_w.cuda()
        
        conv2_w1 = conv2_w - conv2_w.mean([1, 2, 3], keepdim=True)
        conv2_w1 = conv2_w1.cuda()
        
        conv2_w2 = conv2_w1 / conv2_w1.std([1, 2, 3], keepdim=True)
        conv2_w2 = conv2_w2.cuda()
        
        conv2_a, conv2_b = get_ab(np.prod(conv2_w.shape[1:]))
        conv2_X = conv2_w2.view(conv2_w.shape[0], conv2_a, conv2_b)
        conv2_X = conv2_X.cuda()
        
        conv2_Rweight = ((self.conv2_R1.t()) @ conv2_X @ (self.conv2_R2)).view_as(conv2_w)
        conv2_Rweight = conv2_Rweight.cuda()
        
        conv2_delta = conv2_Rweight.detach() - conv2_w2
        conv2_delta = conv2_delta.cuda()
        
        conv2_w3 = conv2_w2 + torch.abs(torch.sin(self.conv2_rotate)) * conv2_delta
        conv2_w3 = conv2_w3.cuda()
        
        conv2_bw = torch.sign(conv2_w3)
        self.conv2_weight_q = conv2_bw
        # initialize conv1
        self.conv2.weight = nn.Parameter(data=self.conv2_weight_q)
        self.conv2.bias = nn.Parameter(data=self.conv2_bias)

        ## conv3
        # get quantized weights for conv1
        conv3_w = self.conv3_weight
        conv3_w = conv3_w.cuda()
        
        conv3_w1 = conv3_w - conv3_w.mean([1, 2, 3], keepdim=True)
        conv3_w1 = conv3_w1.cuda()
        
        conv3_w2 = conv3_w1 / conv3_w1.std([1, 2, 3], keepdim=True)
        conv3_w2 = conv3_w2.cuda()
        
        conv3_a, conv3_b = get_ab(np.prod(conv3_w.shape[1:]))
        conv3_X = conv3_w2.view(conv3_w.shape[0], conv3_a, conv3_b)
        conv3_X = conv3_X.cuda()
        
        conv3_Rweight = ((self.conv3_R1.t()) @ conv3_X @ (self.conv3_R2)).view_as(conv3_w)
        conv3_Rweight = conv3_Rweight.cuda()
        
        conv3_delta = conv3_Rweight.detach() - conv3_w2
        conv3_delta = conv3_delta.cuda()
        
        conv3_w3 = conv3_w2 + torch.abs(torch.sin(self.conv3_rotate)) * conv3_delta
        conv3_w3 = conv3_w3.cuda()
        
        conv3_bw = torch.sign(conv3_w3)
        self.conv3_weight_q = conv3_bw
        # initialize conv1
        self.conv3.weight = nn.Parameter(data=self.conv3_weight_q)
        self.conv3.bias = nn.Parameter(data=self.conv3_bias)

        ## conv4
        # get quantized weights for conv1
        conv4_w = self.conv4_weight
        conv4_w = conv4_w.cuda()
        
        conv4_w1 = conv4_w - conv4_w.mean([1, 2, 3], keepdim=True)
        conv4_w1 = conv4_w1.cuda()
        
        conv4_w2 = conv4_w1 / conv4_w1.std([1, 2, 3], keepdim=True)
        conv4_w2 = conv4_w2.cuda()
        
        conv4_a, conv4_b = get_ab(np.prod(conv4_w.shape[1:]))
        conv4_X = conv4_w2.view(conv4_w.shape[0], conv4_a, conv4_b)
        conv4_X = conv4_X.cuda()
        
        conv4_Rweight = ((self.conv4_R1.t()) @ conv4_X @ (self.conv4_R2)).view_as(conv4_w)
        conv4_Rweight = conv4_Rweight.cuda()
        
        conv4_delta = conv4_Rweight.detach() - conv4_w2
        conv4_delta = conv4_delta.cuda()
        
        conv4_w3 = conv4_w2 + torch.abs(torch.sin(self.conv4_rotate)) * conv4_delta
        conv4_w3 = conv4_w3.cuda()
        
        conv4_bw = torch.sign(conv4_w3)
        self.conv4_weight_q = conv4_bw
        # initialize conv1
        self.conv4.weight = nn.Parameter(data=self.conv4_weight_q)
        self.conv4.bias = nn.Parameter(data=self.conv4_bias)

        ## fc1
        # get quantized weights for conv1
        fc1_w = self.fc1_weight
        fc1_w = fc1_w.cuda()
        
        fc1_w1 = fc1_w - fc1_w.mean([1], keepdim=True)
        fc1_w1 = fc1_w1.cuda()
        
        fc1_w2 = fc1_w1 / fc1_w1.std([1], keepdim=True)
        fc1_w2 = fc1_w2.cuda()
        
        fc1_a, fc1_b = get_ab(np.prod(fc1_w.shape[1:]))
        fc1_X = fc1_w2.view(fc1_w.shape[0], fc1_a, fc1_b)
        fc1_X = fc1_X.cuda()
        
        fc1_Rweight = ((self.fc1_R1.t()) @ fc1_X @ (self.fc1_R2)).view_as(fc1_w)
        fc1_Rweight = fc1_Rweight.cuda()
        
        fc1_delta = fc1_Rweight.detach() - fc1_w2
        fc1_delta = fc1_delta.cuda()
        
        fc1_w3 = fc1_w2 + torch.abs(torch.sin(self.fc1_rotate)) * fc1_delta
        fc1_w3 = fc1_w3.cuda()
        
        fc1_bw = torch.sign(fc1_w3)
        self.fc1_weight_q = fc1_bw
        # initialize conv1
        self.fc1.weight = nn.Parameter(data=self.fc1_weight_q)
        self.fc1.bias = nn.Parameter(data=self.fc1_bias)

    def forward(self, x):
        # pass through conv1
        x = self.conv1(x)

        # pass through relu1
        x = self.relu1(x)

        # pass through conv2
        x0 = x
        x1 = x0 - x0.mean([1, 2, 3], keepdim=True)
        x2 = x1 / x1.std([1, 2, 3], keepdim=True)
        # bx = torch.sign(x2)
        bx = x2
        x = self.conv2(bx)
        x = x * self.conv2_alpha

        # pass through relu2
        x = self.relu2(x)

        # pass through maxpool1
        x = self.maxpool1(x)

        # pass through conv3
        x0 = x
        x1 = x0 - x0.mean([1, 2, 3], keepdim=True)
        x2 = x1 / x1.std([1, 2, 3], keepdim=True)
        # bx = torch.sign(x2)
        bx = x2
        x = self.conv3(bx)
        x = x * self.conv3_alpha

        # pass through relu3
        x = self.relu3(x)

        # pass through conv4
        x0 = x
        x1 = x0 - x0.mean([1, 2, 3], keepdim=True)
        x2 = x1 / x1.std([1, 2, 3], keepdim=True)
        # bx = torch.sign(x2)
        bx = x2
        x = self.conv4(bx)
        x = x * self.conv4_alpha

        # pass through relu4
        x = self.relu4(x)

        # pass through maxpooling2
        x = self.maxpool2(x)

        # vectorize the input
        x = x.view(x.size(0), -1)
        x0 = x
        x1 = x0 - x0.mean([1], keepdim=True)
        x2 = x1 / x1.std([1], keepdim=True)
        # bx = torch.sign(x2)
        bx = x2
        x = self.fc1(bx)
        x = x * self.fc1_alpha

        # bipolar output
        x = torch.sign(x)

        return x


def get_ab(N):
    sqrt = int(np.sqrt(N))
    for i in range(sqrt, 0, -1):
        if N % i == 0:
            return i, N // i


if __name__ == '__main__':
    controller = Controller_RBNN(state_dict='/mnt/nfsdisk/rlin/log_0707/log_ab4/model_best.pth.tar', num_in_channels=1,
                                 feature_dim=512)
    x = torch.rand([10, 1, 32, 32])
    features = controller(x)
    print(features.shape)
    print(features)
