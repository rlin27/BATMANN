import torch
import torch.nn as nn


class Controller_XNOR(nn.Module):
    """
    This model definition is EXTREMELY redundant, only used for Memotorch evaluation.
    """

    def __init__(self, state_dict='/mnt/nfsdisk/rlin/log_0707/log_ab3/model_best.pth.tar', num_in_channels=1,
                 feature_dim=512):
        super(Controller_XNOR, self).__init__()

        # load weights and alpha from the pretrained parameters
        ckpt = torch.load(state_dict, map_location='cuda:0')
        params = ckpt['state_dict']
        self.conv1_weight = params['module.features.conv1.weight'].detach().cpu().data
        self.conv2_weight = params['module.features.conv2.weight'].detach().cpu().data
        self.conv2_alpha = params['module.features.conv2.alpha'].detach().data
        self.conv3_weight = params['module.features.conv3.weight'].detach().cpu().data
        self.conv3_alpha = params['module.features.conv3.alpha'].detach().data
        self.conv4_weight = params['module.features.conv4.weight'].detach().cpu().data
        self.conv4_alpha = params['module.features.conv4.alpha'].detach().data
        self.fc1_weight = params['module.fc1.weight'].detach().cpu().data
        self.fc1_bias = params['module.fc1.bias'].detach().cpu().data
        self.fc1_alpha = params['module.fc1.alpha'].detach().cpu().data

        # define different layers
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

        # get quantized weight for conv1
        conv1_restore_w = self.conv1_weight
        conv1_max = conv1_restore_w.data.max()
        conv1_weight_q = conv1_restore_w.div(conv1_max).mul(127).round().div(127).mul(conv1_max)
        conv1_weight_q = (conv1_weight_q - conv1_restore_w).detach() + conv1_restore_w
        self.conv1_weight_q = conv1_weight_q
        # initialize conv1
        self.conv1.weight = nn.Parameter(data=self.conv1_weight_q)

        # get quantized weight for conv2
        conv2_w = self.conv2_weight
        conv2_w1 = conv2_w - conv2_w.mean([1, 2, 3], keepdim=True)
        conv2_w2 = conv2_w1 / conv2_w1.std([1, 2, 3], keepdim=True)
        self.conv2_weight_q = torch.sign(conv2_w2)
        # initialize conv2
        self.conv2.weight = nn.Parameter(data=self.conv2_weight_q)

        # get quantized weight for conv3
        conv3_w = self.conv3_weight
        conv3_w1 = conv3_w - conv3_w.mean([1, 2, 3], keepdim=True)
        conv3_w2 = conv3_w1 / conv3_w1.std([1, 2, 3], keepdim=True)
        self.conv3_weight_q = torch.sign(conv3_w2)
        # initialize conv3
        self.conv3.weight = nn.Parameter(data=self.conv3_weight_q)

        # get quantized weight for conv4
        conv4_w = self.conv4_weight
        conv4_w1 = conv4_w - conv4_w.mean([1, 2, 3], keepdim=True)
        conv4_w2 = conv4_w1 / conv4_w1.std([1, 2, 3], keepdim=True)
        self.conv4_weight_q = torch.sign(conv4_w2)
        # initialize conv4
        self.conv4.weight = nn.Parameter(data=self.conv4_weight_q)

        # get quantized weight for fc1
        fc1_w = self.fc1_weight
        fc1_w1 = fc1_w - fc1_w.mean([1], keepdim=True)
        fc1_w2 = fc1_w1 / fc1_w1.std([1], keepdim=True)
        self.fc1_weight_q = torch.sign(fc1_w2)
        # initialize fc1
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
        bx = torch.sign(x2)
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
        bx = torch.sign(x2)
        x = self.conv3(bx)
        x = x * self.conv3_alpha

        # pass through relu3
        x = self.relu3(x)

        # pass through conv4
        x0 = x
        x1 = x0 - x0.mean([1, 2, 3], keepdim=True)
        x2 = x1 / x1.std([1, 2, 3], keepdim=True)
        bx = torch.sign(x2)
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
        bx = torch.sign(x2)
        x = self.fc1(bx)
        x = x * self.fc1_alpha

        return x


if __name__ == '__main__':
    controller = Controller_XNOR(state_dict='/mnt/nfsdisk/rlin/log_0707/log_ab3/model_best.pth.tar', num_in_channels=1,
                                 feature_dim=512)
    x = torch.rand([10, 1, 32, 32])
    features = controller(x)
    print(features.shape)
    print(features)
