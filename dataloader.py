import numpy as np
import torch
import torch.utils.data as data
from os.path import join
from glob2 import glob


class SHL(data.Dataset):
    def __init__(self, x_path, y_path):
        self.x = np.load(x_path)
        self.y = np.load(y_path).astype(np.long)
        assert len(self.x) == len(self.y), "Length of X and Y doesn't match"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        """
        TODO: try feature expansion (downsampling, frequency sampling) - https://arxiv.org/pdf/1603.06995.pdf

        :param item: index
        :return: (x,y) x - 2D array [features, channels]
        """
        return self.x[item], self.y[item]


def shl_loaders(train_split_ratio=0.7, batch_size=32):
    """
    :return: source_train_loader, source_val_loader, target_train_loader, target_val_loader
    """
    source_folders = list(glob('./data/shl-source/*'))
    target_folders = list(glob('./data/shl-3users-target/*'))
    source_place = 'Hips'
    target_place = 'Torso'

    source_datasets = [
        SHL(join(f, source_place + "_Motion.npy"), join(f, source_place + "_Motion_labels.npy"))
        for f in source_folders
    ]
    target_datasets = [
        SHL(join(f, target_place + "_Motion.npy"), join(f, target_place + "_Motion_labels.npy"))
        for f in target_folders
    ]

    target_train_loader, target_val_loader = to_dataloaders(*tt_split(torch.utils.data.ConcatDataset(target_datasets),
                                                                      train_split_ratio),
                                                            batch_size=batch_size)

    source_train_loader, source_val_loader = to_dataloaders(*tt_split(torch.utils.data.ConcatDataset(source_datasets),
                                                                      train_split_ratio),
                                                            batch_size=batch_size)
    return source_train_loader, source_val_loader, target_train_loader, target_val_loader


def to_dataloaders(train_dataset, val_dataset, batch_size):
    train_dl = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=0)

    val_dl = torch.utils.data.DataLoader(dataset=val_dataset,
                                         batch_size=512,
                                         shuffle=False,
                                         num_workers=0)
    return train_dl, val_dl


def tt_split(dataset, train_split_ratio):
    train_size = int(train_split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    torch.random.manual_seed(42)
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_set, test_set


# Little test case
if __name__ == '__main__':
    source = SHL('./data/shl/220617/Hand_Motion.npy', './data/shl/220617/Hand_Motion_labels.npy')
    target = SHL('./data/shl/220617/Hips_Motion.npy', './data/shl/220617/Hips_Motion_labels.npy')

    a = source[0]
    b = source[1]
    print(a[0].shape)
    assert a[0].shape == b[0].shape

    # Try dataloader
    source_dataloader = torch.utils.data.DataLoader(dataset=source,
                                                    batch_size=16,
                                                    shuffle=True,
                                                    num_workers=0)

    target_dataloader = torch.utils.data.DataLoader(dataset=target,
                                                    batch_size=16,
                                                    shuffle=True,
                                                    num_workers=0)

    a, aa = next(iter(source_dataloader))
    print(a.shape, aa.shape)  # torch.Size([16, 9, 500]) torch.Size([16, 1])
    b, bb = next(iter(target_dataloader))
    assert a.shape == b.shape
    assert aa.shape == bb.shape
