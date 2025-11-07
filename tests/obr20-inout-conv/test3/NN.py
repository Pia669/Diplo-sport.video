import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import Dataset


FRAMES_STEP = 5


class NeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(NeuralNet, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, 3, 1, 1)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(64, 128, 3, 2, 1)
        self.bn2 = nn.BatchNorm3d(128)
        self.conv3 = nn.Conv3d(128, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm3d(128)

        self.maxPool3d = nn.MaxPool3d(3, 2, 1)
        self.avgPool3d = nn.AvgPool3d(5, 1, (0, 2, 2))

        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(256)

        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.swapaxes(x, 1, 2)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxPool3d(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.avgPool3d(x)

        batch_size, num_frames, c, h, w = x.size()
        x = x.view(batch_size, -1, h, w)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)

        x = self.avgPool(x)
        x = x.view(batch_size, -1)

        x = self.relu(self.fc1(x))

        return x


if __name__ == '__main__':
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)

    net = NeuralNet(3)
    print('NN')

    test_batch_size = 1
    dataset = Dataset.VideoDataset('test.json')
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=test_batch_size, shuffle=False)
    print("dataset")

    for images, labels in test_loader:
        output = net(images)

        print('output type: {}, expected: {}'.format(output, labels))

        break
