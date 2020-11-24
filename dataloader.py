import numpy as np
import torch
import torch.utils.data as data


class SHL(data.Dataset):
    def __init__(self, x_path, y_path):
        self.x = np.load(x_path)
        self.y = np.load(y_path)
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
