import matplotlib.pyplot as plt
import myutils


PATH = 'tests/test{}/stats.json'


perfomance = myutils.read_data(PATH)

train_losses = perfomance['train_losses']
train_counter = perfomance['train_counter']
train_accuracy = perfomance['train_accuracy']

test_accuracy = perfomance['test_accuracy']
test_losses = perfomance['test_losses']
test_counter = perfomance['test_counter']

if 'test_confusion_mtx' in perfomance.keys():
    test_confusion_matrix = perfomance['test_confusion_mtx']

print('Total epochs: {}'.format(len(test_accuracy)))
print('Max Accuracy is: {}%'.format(round(100 * max(test_accuracy), 2)))
if 'test_confusion_mtx' in perfomance.keys():
    for row in test_confusion_matrix[-1]:
        print(row)

fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue', zorder=1)
plt.scatter(test_counter, train_accuracy, color='yellow', zorder=2)
plt.scatter(test_counter, test_losses, color='red', zorder=3)
plt.scatter(test_counter, test_accuracy, color='green', marker='+', zorder=4)
plt.legend(['Train Loss', 'Train accuracy', 'Test Loss', 'Accuracy'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.savefig('plot.png')
plt.ylim(0, 2)
fig.show()