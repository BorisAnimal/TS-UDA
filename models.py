import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import SHL


class Encoder(nn.Module):
    def __init__(self):
        """
        Trying architecture as in Saito's "Asymmetric Tri-training for Unsupervised Domain Adaptation"
        """
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=9, out_channels=64,
                               kernel_size=15, stride=7)  # TODO: abulate hyperparameters
        self.pool1 = nn.MaxPool1d(kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=5, stride=1)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=1)
        # self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(2560, 512)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        # x = self.pool3(x)

        x = self.fc1(x.flatten(start_dim=1))
        x = F.relu(x)

        return x


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
    source = SHL('./data/shl/220617/Hand_Motion.npy', './data/shl/220617/Hand_Motion_labels.npy')
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
