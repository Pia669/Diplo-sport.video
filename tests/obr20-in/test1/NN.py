import torch
import torch.nn as nn
import torchvision.models as models

import Dataset


class NeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(NeuralNet, self).__init__()

        self.resnet = models.resnet18(pretrained=True)

        self.resnet.conv1 = nn.Conv2d(
            60,
            self.resnet.conv1.out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False
        )

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        batch_size, num_frames, c, h, w = x.size()
        x = torch.reshape(x, (batch_size, -1, h, w))

        x = self.resnet(x)

        x = x.view(batch_size, -1)

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
