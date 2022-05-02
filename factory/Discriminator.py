import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    def __init__(self, feature_num, down_feature, apply_spectral_norm=False):
        super(Discriminator, self).__init__()
        if apply_spectral_norm:
            self.conv1 = spectral_norm(
                nn.Conv1d(feature_num, down_feature, 5, padding=2)
            )
            self.conv2 = spectral_norm(
                nn.Conv1d(down_feature, int(down_feature / 2), 5, padding=2)
            )
            self.conv3 = spectral_norm(
                nn.Conv1d(int(down_feature / 2), int(down_feature / 4), 5, padding=2)
            )
            self.dense1 = spectral_norm(nn.Linear(int(down_feature / 4) * 128, 1))
        else:
            self.conv1 = nn.Conv1d(feature_num, down_feature, 5, padding=2,)
            self.conv2 = nn.Conv1d(down_feature, int(down_feature / 2), 5, padding=2)
            self.conv3 = nn.Conv1d(
                int(down_feature / 2), int(down_feature / 4), 5, padding=2
            )
            self.dense1 = nn.Linear(int(down_feature / 4) * 128, 1)

        self.leaky_relu = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm1d(down_feature)
        self.bn2 = nn.BatchNorm1d(int(down_feature / 2))
        self.bn3 = nn.BatchNorm1d(int(down_feature / 4))
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)
        x = self.flatten(x)
        x_1 = self.dense1(x)
        return self.sigmoid(x_1)
