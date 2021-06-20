from PIL import Image
import torch.utils.data as data


class FewShotDataset(data.Dataset):
    """
    Load image-label paris from a task to pass to Torch DataLoader, and tasks consist
    data and labels, which will be split into train / val set. This is a subclass for
    a particular dataset.
    """

    def __init__(self, task, split='train', transform=None, target_transform=None):
        """
        task: a tuple, (data, label).
        split: a string, which claims splitting data for training/validation or testing.
        transform: torch operations on the input image.
        target_transform: torch operations on the label.
        """
        self.transform = transform
        self.target_transform = target_transform
        self.task = task
        self.split = split

        # Split the support vectors
        if self.split == 'train':
            self.image_roots = self.task.train_roots
        elif self.split == 'query':
            self.image_roots = self.task.test_roots
        elif self.split == 'val':
            self.image_roots = self.task.val_roots

        # Split the corresponding labels
        if self.split == 'train':
            self.labels = self.task.train_labels
        elif self.split == 'query':
            self.labels = self.task.test_labels
        elif self.split == 'val':
            self.labels = self.task.val_labels

    def __len__(self):
        """ Get the number support vectors in the training set """
        return len(self.image_roots)

    def __getitem__(self, idx):
        """ Warning when the function cannot function well. """
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")


class Omniglot(FewShotDataset):
    """ Process images and labels according to the given transform (if not None). """

    def __init__(self, *args, **kwargs):
        super(Omniglot, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        # Process the image
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('L')
        image = image.resize((32, 32), resample=Image.LANCZOS)
        if self.transform is not None:
            image = self.transform(image)

        # Process the label
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label
