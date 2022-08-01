import math
from typing import Any, Callable, Optional, Union

import torch
import pytorch_lightning as pl
from torchvision import transforms, datasets


class MNISTDataModule(pl.LightningDataModule):

    name = "mnist"
    k = 10
    dims = (1, 28, 28)
    unrolled_dim = math.prod(dims)

    def __init__(
        self,
        data_dir: str = None,
        val_split: Union[int, float] = 0.2,
        normalize: bool = False,
        flatten: bool = False,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.val_split = val_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        transform = [transforms.ToTensor()]
        if normalize:
            transform.append(transforms.Normalize(mean=(0.5,), std=(0.5,)))
        if flatten:
            transform.append(transforms.Lambda(lambda x: torch.flatten(x)))
        self.transform = transforms.Compose(transform)

    def prepare_data(self):
        # download
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        mnist_full = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
        if self.val_split in {0, 0.0}:
            self.mnist_train = mnist_full
            self.mnist_val = None
        else:
            if isinstance(self.val_split, float):
                self.mnist_train, self.mnist_val = torch.utils.data.random_split(mnist_full, [int((1 - self.val_split) * len(mnist_full)), int(self.val_split * len(mnist_full))])
            else:
                self.mnist_train, self.mnist_val = torch.utils.data.random_split(mnist_full, [len(mnist_full) - self.val_split, self.val_split])
        self.mnist_test = datasets.MNIST(self.data_dir, train=False, transform=self.transform)
        self.mnist_predict = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

        def produce_X_y(dataset):
            X, y = next(iter(torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False, drop_last=False)))
            y, indices = torch.sort(y)
            X = X[indices]
            return X, y

        self.X_train, self.y_train = produce_X_y(self.mnist_train)
        self.X_test, self.y_test = produce_X_y(self.mnist_test)


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last)

    def val_dataloader(self):
        if not self.mnist_val:
            return None
        return torch.utils.data.DataLoader(self.mnist_val, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_test, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=self.drop_last)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_predict, batch_size=self.batch_size, shuffle=False, drop_last=False)
