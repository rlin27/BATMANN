import torch
import torchvision.datasets as datasets
import os 
from torchvision import transforms

def MNIST2FSL(download_PATH, train_PATH, test_PATH):

    mnist_trainset = datasets.MNIST(root=download_PATH, train=True,
                                    download=True, transform=None)
    mnist_testset = datasets.MNIST(root=download_PATH, train=False,
                                   download=True, transform=None)
    
    toPIL = transforms.ToPILImage()

    # Training dataset
    training_image = mnist_trainset.data
    training_image = training_image.unsqueeze(1)
    training_target = mnist_trainset.targets

    idx_list = []
    for i in range(10):
        idx_temp = [x for x in range(len(training_target)) if training_target[x] == i]
        idx_list.append(idx_temp)

    image_list = [0 for i in range(10)]
    for i in range(10):
        image_list[i] = training_image[idx_list[i], :, :, :]

    for i in range(10):
        temp = image_list[i]
        for j in range(20):
            if not os.path.isdir(os.path.join(train_PATH, 'digit_' + str(i))):
                os.makedirs(os.path.join(train_PATH, 'digit_' + str(i)))
            idx = torch.randperm(len(idx_list[i]))[j]
            single_digit = toPIL(temp[idx, :, :, :])
            single_digit.save(os.path.join(train_PATH, 'digit_' + str(i), str(j+1) + '.png'))

    # Testing dataset
    testing_image = mnist_testset.data
    testing_image = testing_image.unsqueeze(1)
    testing_target = mnist_testset.targets

    idx_list = []
    for i in range(10):
        idx_temp = [x for x in range(len(testing_target)) if testing_target[x] == i]
        idx_list.append(idx_temp)
        
    image_list = [0 for i in range(10)]
    for i in range(10):
        image_list[i] = testing_image[idx_list[i], :, :, :]

    for i in range(10):
        temp = image_list[i]
        for j in range(20):
            if not os.path.isdir(os.path.join(test_PATH, 'digit_' + str(i))):
                os.makedirs(os.path.join(test_PATH, 'digit_' + str(i)))
            idx = torch.randperm(len(idx_list[i]))[j]
            single_digit = toPIL(temp[idx, :, :, :])
            single_digit.save(os.path.join(test_PATH, 'digit_' + str(i), str(j+1) + '.png'))
