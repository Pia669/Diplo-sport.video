import os

import torch
import torch.nn as nn

import Dataset
from Dataset import VideoDataset
from NN import NeuralNet
import myutils


PATH = 'tests/'
MODEL_NAME = 'modelt.pt'
MODEL_PERFORMANCE = 'stats.json'


print('Datasets')
training_dataset = VideoDataset('train.json')
test_dataset = VideoDataset('test.json')

training_batch_size = 4
test_batch_size = 4

training_loader = torch.utils.data.DataLoader(training_dataset, batch_size=training_batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

if os.path.exists(PATH + MODEL_PERFORMANCE):
    perfomance = myutils.read_data(PATH + MODEL_PERFORMANCE)
    train_losses = perfomance['train_losses']
    train_counter = perfomance['train_counter']
    train_accuracy = perfomance['train_accuracy']
    train_confusion_matrix = perfomance['train_confusion_mtx']

    test_accuracy = perfomance['test_accuracy']
    test_losses = perfomance['test_losses']
    test_counter = perfomance['test_counter']
    test_confusion_matrix = perfomance['test_confusion_mtx']
else:
    train_losses = []
    train_counter = []
    train_accuracy = []
    train_confusion_matrix = []

    test_accuracy = []
    test_losses = []
    test_counter = []
    test_confusion_matrix = []

net = NeuralNet(len(Dataset.TYPES_NAMES_TO_NUMBERS.keys()))
print('Model')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('using ', device)
if torch.cuda.is_available():
    net.to(device)

if len(test_accuracy) != 0:
    net.load_state_dict(torch.load(PATH + MODEL_NAME.format(len(test_accuracy)-1), weights_only=True))

loss_function = nn.CrossEntropyLoss()

learning_rate = 0.01
momentum = .9

optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

random_seed = 1
torch.manual_seed(random_seed)


def train(epoch):
    conf_mtx = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    correct_guesses = 0
    for batch_idx, (images, labels) in enumerate(training_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        output = net(images)

        loss = loss_function(output, labels)

        guesses = torch.max(output, 1, keepdim=True)[1]

        correct_guesses += torch.eq(guesses, labels.data.view_as(guesses)).sum()

        loss.backward()
        optimizer.step()

        conf_mtx = change_conf_mtx(conf_mtx, labels, output)

        if (batch_idx * training_batch_size + training_batch_size) % int(len(training_dataset) / 10) == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * training_batch_size, len(training_dataset),
                       100 * batch_idx / len(training_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * training_batch_size) + ((epoch - 1) * len(training_dataset)))

    current_accuracy = float(correct_guesses) / float(len(training_dataset))
    train_accuracy.append(current_accuracy)
    train_confusion_matrix.append(conf_mtx)
    print('Train: Avg. loss: {:.4f}, Accuracy: {:.4f}%\n{}\n{}\n{}'.format(
        sum(train_losses[len(train_losses)-10:])/10, current_accuracy * 100.,
        conf_mtx[0], conf_mtx[1], conf_mtx[2]))


def change_conf_mtx(mtx, labels, ouput):
    for i in range(len(ouput)):
        max_output = ouput[i].argmax()
        mtx[labels[i]][max_output] += 1

    return mtx


def test():
    conf_mtx = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    test_loss = 0
    correct_guesses = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            output = net(images)

            test_loss += loss_function(output, labels).item()

            guesses = torch.max(output, 1, keepdim=True)[1]

            correct_guesses += torch.eq(guesses, labels.data.view_as(guesses)).sum()
            conf_mtx = change_conf_mtx(conf_mtx, labels, output)

        test_loss /= len(test_dataset) / test_batch_size
        test_losses.append(test_loss)

        current_accuracy = float(correct_guesses) / float(len(test_dataset))
        test_accuracy.append(current_accuracy)

        test_confusion_matrix.append(conf_mtx)

        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n{}\n{}\n{}\n'.format(
            test_loss, correct_guesses, len(test_dataset),
            100. * current_accuracy, conf_mtx[0], conf_mtx[1], conf_mtx[2]))


print("\nTraining began")
epoch = len(test_accuracy)
while epoch <= 100:
    train(epoch)
    test()

    test_counter.append(epoch * training_dataset.num_clips)

    if epoch % 5 == 0:
        myutils.save_data(
            {
                'train_losses': train_losses,
                'train_counter': train_counter,
                'train_accuracy': train_accuracy,
                'train_confusion_mtx': train_confusion_matrix,
                'test_accuracy': test_accuracy,
                'test_losses': test_losses,
                'test_counter': test_counter,
                'test_confusion_mtx': test_confusion_matrix
            },
            PATH + MODEL_PERFORMANCE
        )
        torch.save(net.state_dict(), PATH + MODEL_NAME)

    epoch += 1

print('Total epochs: {}'.format(epoch))
print('Max Accuracy is: {}%'.format(round(100 * max(test_accuracy), 2)))
