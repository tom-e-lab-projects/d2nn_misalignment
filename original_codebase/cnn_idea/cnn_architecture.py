
''' 
for training for the missalignments we want to use cnn to learn any geometric features deformations
to then hopefully learn missalignments

num_samples = 10
M = 64
batch_size = 2
train_set = random_data(num_samples , M, system)
train_loader = torch.utils.data.DataLoader(train_set,
                                                batch_size=batch_size,
                                                shuffle=True, num_workers=0,pin_memory = True)
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_c)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)




'''

class CNN_1(nn.Module):
    def __init__(self):
        super().__init__()

        # -------- Image branch --------
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )

        # Global pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # -------- Regression head --------
        self.fc1 = nn.Linear(128 + 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 8)

    def forward(self, image, centroid):

        x = self.features(image)
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = torch.cat([x, centroid], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        tilt = x[:, :4].view(-1, 2, 2)
        offset = x[:, 4:].view(-1, 2, 2)

        return tilt, offset
'''