import random
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
from data.dataset import *
import numpy as np
import os


class Rotate(object):

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x, mode="reflect"):
        x = x.rotate(self.angle)
        return x


def omniglot_character_folders(data_path):
    data_folder = data_path  # I may add a args here

    # 1) for family in os.listdir(data_folder): all folders under the data_folder, each folder is a 'family'
    # 2) if os.path.isdir(os.path.join(data_folder, family)): ./data/[family]
    # 3) for character in os.list(os.path.join(data_folder, family)): the folders in
    #  ./data/[family]/, each folder is a 'character' .
    # 4) os.path.join(data_folder, family, character): ./data/[family]/[character]
    # 5) character_folders: a list, each element is a string to record character path
    character_folders = [os.path.join(data_folder, family, character) \
                         for family in os.listdir(data_folder) \
                         if os.path.isdir(os.path.join(data_folder, family)) \
                         for character in os.listdir(os.path.join(data_folder, family))]

    # shuffle the string
    random.seed(806)
    random.shuffle(character_folders)

    # divide the data into training set and testing set
    num_train = 964  # according to Nat Comm
    manntrain_character_folders = character_folders[:num_train]
    mannval_character_folders = character_folders[num_train:]

    # list formed by string
    return manntrain_character_folders, mannval_character_folders


class OmniglotTask(object):

    def __init__(self, character_folders, num_classes, train_num, query_num, val_num):
        # 1) character_folders: a list, each of its element is path to a character
        # 2) num_classes: an int, number of classes
        # 3) train_num: an int, number of samples for each class when training
        # 4) query_num: an int, number of samples for each class when testing
        # 5) val_num: an int, number of samples for each class when doing validation. For training
        #    dataset, val_num > 0; for testing dataset, val_num = 0.

        self.character_folders = character_folders
        self.num_classes = num_classes
        self.train_num = train_num
        self.test_num = query_num
        self.val_num = val_num

        # randomly get #num_classes dir strings.
        class_folders = random.sample(self.character_folders, self.num_classes)
        # generate labels according to the #num_classes.
        labels = np.array(range(len(class_folders)))
        # assgin a number to each folder
        labels = dict(zip(class_folders, labels))

        samples = dict()

        self.train_roots = []
        self.test_roots = []
        self.val_roots = []
        for c in class_folders:  # for training dataset, c: ./data/[family]/[character]
            # 1) for x in os.listdir(c): for all the files under the ./data/family/character directory
            # 2) os.path.join(c, x): ./data/[family]/[character]/[c]
            # 3) temp is a list, each of its element is a string that records ./data/[family]/[character]/[c]
            temp = [os.path.join(c, x) for x in os.listdir(c)]
            # 1) samples is a dictionary
            # 2) len(temp) = num_classes
            # 3) temp: ./data/[family]/[character]/[x]
            # 4) equivalent to randperm?
            samples[c] = random.sample(temp, len(temp))

            self.train_roots += samples[c][:train_num]
            self.test_roots += samples[c][train_num:(train_num + query_num)]
            self.val_roots += samples[c][(train_num + query_num):(train_num + query_num + val_num)]

        # train_roots have train_num * num_classes paths
        # 1) for x in self.train_root: ./data/family/character/y
        # 2) for self.get_class(x): see below
        # 3) labels[get ./data/[family]/[character]] = label
        self.train_labels = [labels['/' + self.get_class(x)] for x in self.train_roots]

        # test_roots have query_num * num_classes paths
        # The same as above
        self.test_labels = [labels['/' + self.get_class(x)] for x in self.test_roots]

        # val_roots have val_num * num_classes paths
        # The same as above
        self.val_labels = [labels['/' + self.get_class(x)] for x in self.val_roots]

    def get_class(self, sample):
        """ get ./data/[family]/[character] """
        return os.path.join(*sample.split('/')[:-1])


class ClassBalancedSampler(Sampler):
    """
    Select #num_per_class samples from each class in the 'num_cl' pools.

    - num_per_class: number of shots.
    - num_cl: number of classes.
    - num_inst: number of instances per class.
    - shuffle: change the order of the samples.
    """

    def __init__(self, num_per_class, num_cl, num_inst, shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        """ Return a single list of indices, assuming that items will be grouped by class. """

        # 1) for j in range(self.num_cl): for each class for i in torch.randperm(self.num_inst): for all the samples in
        # 2) one selected class j torch.randperm(self.num_inst)[:self.num_per_class]: choose fixed number of samples,
        # namely, number of shots each class
        # 3) i+j*self.num_inst: assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i + j * self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class] for j in
                      range(self.num_cl)]]
        else:  # no randperm to change the order of samples in one class
            batch = [[i + j * self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in
                     range(self.num_cl)]

        # [[a1,a2,....,an], ..., [z1, z2,...,zn]] --> [a1, a2,...,an,...,z1,z2,...,zn]
        batch = [item for sublist in batch for item in sublist]

        # batch is a list with each element as a index now
        if self.shuffle:
            random.shuffle(batch)

        return iter(batch)

    def __len__(self):
        return 1


def get_data_loader(task, num_per_class=5, split='train', shuffle=True, rotation=0):
    """
    - task: it consists data and labels, which will be split into train / val set.
    - num_per_class: number of shots (m-way, n-shot problem).
    - split: a str, to split the dataset or not.
    - shuffle: Bool, which decides to change the order of samples idx or not.
    - rotation: data augmentation.
    """

    normalize = transforms.Normalize(mean=[0.92206], std=[0.08426])

    dataset = Omniglot(task, split=split,
                       transform=transforms.Compose([Rotate(rotation), transforms.ToTensor(), normalize]))

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.train_num, shuffle=shuffle)
    elif split == 'val':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.val_num, shuffle=shuffle)
    elif split == 'query':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.test_num, shuffle=shuffle)

    loader = DataLoader(dataset, batch_size=num_per_class * task.num_classes, sampler=sampler)

    return loader
