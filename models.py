import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import SHL


class Encoder(nn.Module):

    @staticmethod
    def conv(in_ch, out_ch=128, ks=15, stride=10):
        conv = nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=ks, stride=stride)
        bn = nn.BatchNorm1d(out_ch)
        act = nn.ReLU()
        return nn.Sequential(conv, bn, act)

    def __init__(self):
        """
        Trying architecture as in Saito's "Asymmetric Tri-training for Unsupervised Domain Adaptation"
        """
        super(Encoder, self).__init__()
        self.conv1 = self.conv(9, 128, 15, 10)
        self.conv2 = self.conv(128, 128, 5, 2)
        self.conv3 = self.conv(128, 128, 3, 1)
        self.fc = nn.Sequential(nn.Linear(2688, 512),
                                nn.BatchNorm1d(512),
                                nn.ReLU(),
                                nn.Dropout())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.fc(x.flatten(start_dim=1))


class Classifier(nn.Module):

    def __init__(self, in_shape=512, out_shape=9):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_shape, 128)
        self.fc2 = nn.Linear(128, out_shape)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x


if __name__ == '__main__':
    # Init loader
    source = SHL('./data/shl-3users-target/220617/Hand_Motion.npy',
                 './data/shl-3users-target/220617/Hand_Motion_labels.npy')
    source_dataloader = torch.utils.data.DataLoader(dataset=source,
                                                    batch_size=16,
                                                    shuffle=True,
                                                    num_workers=0)
    a, aa = next(iter(source_dataloader))
    print("Tensor from dataloader:", a.shape)
    # Init model
    net = nn.Conv1d(in_channels=9, out_channels=16, kernel_size=15, stride=7)

    # Try forward batch
    a = a.float()
    res = net(a)
    print("Result of Conv1d:", res.shape)

    pool1d = nn.MaxPool1d(5, 2)
    res = pool1d(res)
    print("Result of MaxPool1d:", res.shape)

    net = Encoder()
    res = net(a)
    print("Result of Encoder:", res.shape)

    clf = Classifier()
    res = clf(res)
    print("Result of Classifier:", res.shape)
