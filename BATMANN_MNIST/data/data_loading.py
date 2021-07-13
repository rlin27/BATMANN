import random
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
import numpy as np
import os
from PIL import Image

from data.preprocess_mnist import MNIST2FSL

class FewShotDataset(data.Dataset):

    def __init__(self, task, split='train', transform=None, target_transform=None):
        """
        - task: a class defined in MNIST_Task
        - split: a string, which claims splitting data for training or testing
        - transform: torch operation on the input image
        - target_transform: torch operations on label
        """
        self.transform = transform
        self.target_transform = target_transform
        self.task = task
        self.split = split

        # split the support vectors
        if self.split == 'train':
            self.image_roots = self.task.train_roots
        elif self.split == 'query':
            self.image_roots = self.task.test_roots

        # split the labels
        if self.split == 'train':
            self.labels = self.task.train_labels
        elif self.split == 'query':
            self.labels = self.task.test_labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        """ Warning when the function cannot function well. """
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class MNIST(FewShotDataset):
    """
    Process images and labels according to the given transform (if not None).
    """

    def __init__(self, *args, **kwargs):
        super(MNIST, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        # process the image
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('L')
        if self.transform is not None:
            image = self.transform(image)

        # Process the label
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

class Rotate(object):

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x

def mnist_digit_folders(train_PATH, test_PATH):
    digit_folders_train = [os.path.join(train_PATH, 'digit_' + str(x)) \
                           for x in range(10)]
    digit_folders_test = [os.path.join(test_PATH, 'digit_' + str(x)) \
                          for x in range(10)]
    return digit_folders_train, digit_folders_test


class MNIST_Task(object):

    def __init__(self, digit_folders, num_classes=10, train_num=5, query_num=3):
        """
        After running preprocess_mnist.py, the dataset is reformatted into 10 tensors.

        - digit_folders_train: a string, the PATH of the 10 training tensors.
        - digit_folders_test: a string, the PATH of the 10 testing tensor.
        - num_classes: an int, the default value is 10.
        - train_num: an int, the number of support samples selected in each class.
        - query_num: an int, the number of query samples selected in each class.

        Generate train_roots, test_roots
        """

        self.digit_folders = digit_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.query_num = query_num

        # get all classes
        digit_folders = self.digit_folders
        # generate labels according
        labels = np.array(range(len(digit_folders)))
        # assign a number to each folder
        labels = dict(zip(digit_folders, labels))

        samples = dict()

        self.train_roots = []
        self.test_roots = []
        for c in digit_folders:
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            samples[c] = random.sample(temp, len(temp))
            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:train_num+query_num]

        self.train_labels = [labels[self.get_class(x)] for x in self.train_roots]
        self.test_labels = [labels[self.get_class(x)] for x in self.test_roots]

    def get_class(self, sample):
        return os.path.join(*sample.split('/')[:-1])

class ClassBalancedSampler(Sampler):
    """
    Select #num_per_class (n-shot) samples from each class

    - num_per_class: number of shots
    - num_cl: number of classes
    - num_inst: number of instances per class
    - shuffle: change the order of the samples
    """
    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]
                      for j in range(self.num_cl)]]
        else:
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]

        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)

        return iter(batch)

    def __len__(self):
        return 1

def get_data_loader(task, num_per_class=5, split='train', shuffle=True, rotation=0):
    """
    - task: it consists data and labels
    - num_per_class: number of shots (m-way, n-shot problem)
    - split: a string, {'train', 'query'}}.
    """
    normalize = transforms.Normalize(mean=[0.1307], std=[0.3081])

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
    elif split == 'query':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.query_num, shuffle=shuffle)

    dataset = MNIST(task, split=split, transform=transforms.Compose([Rotate(rotation), transforms.ToTensor(), normalize]))
    loader = DataLoader(dataset, batch_size = num_per_class * task.num_classes, sampler=sampler)

    return loader


if __name__ == '__main__':
    print('========> Use MNIST to generate dataset for FLS ...')
    MNIST2FSL()
    print('Done!')
    print('========> DataLoader Test ...')
    data_path = 'mnist_digit'
    class_num = 10
    train_num=5
    query_num=3
    digit_folders_train, digit_folders_test = mnist_digit_folders(data_path)
    print('Show the folders of training digits:')
    print(digit_folders_train)
    print('Show the folders of testing digits:')
    print(digit_folders_test)
    degrees = random.choice([0, 90, 180, 270])
    task_train = MNIST_Task(digit_folders_train, class_num, train_num, query_num)
    support_dataloader = get_data_loader(task_train, num_per_class=5, split='train', shuffle=False, rotation=degrees)
    query_dataloader = get_data_loader(task_train, num_per_class=3, split='query', shuffle=True, rotation=degrees)
    supports, supports_labels = support_dataloader.__iter__().next()
    queries, queires_labels = query_dataloader.__iter__().next()
    print('The shapes of support set and the corresponding label (training):')
    print(supports.shape, supports_labels.shape)
    print('The shapes of support set and the corresponding label (testing):')
    print(queries.shape, queires_labels.shape)
    print('Dataloader works well!')

