import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import Dataset


FRAMES_STEP = 5


class Block2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class NeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(60, 128, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = Block2D(128, 128, stride=1)
        self.layer2 = Block2D(128, 256, stride=2)
        self.layer3 = Block2D(256, 256, stride=1)
        self.layer4 = Block2D(256, 512, stride=2)
        self.layer5 = Block2D(512, 512, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.size()
        x = x.view(batch_size, -1, h, w)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.avgpool(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)

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
