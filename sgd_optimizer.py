from __future__ import print_function
import argparse
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(log_interval, model, train_loader, optimizer, epoch, train_acc):
    correct = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        accuracy = 100. * correct / len(data)
        train_acc.append(accuracy)
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, test_loader, test_acc):
    model.eval()
    test_loss = 0
    correct_test = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct_test += pred.eq(target.view_as(pred)).sum().item()
            accuracy = 100. * correct_test / len(data)
            test_acc.append(accuracy)

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct_test, len(test_loader.dataset),
        100. * correct_test / len(test_loader.dataset)))


def SGDTrain(batch_size=64, test_batch_size=1000, epochs=2, lr=0.01, momentum=0.5, seed=1, log_interval=10, save_model=False, ratio=0.5):
    torch.manual_seed(seed)
    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=test_batch_size, shuffle=True, **kwargs)

    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=lr,
                          momentum=momentum)

    train_acc = []
    test_acc = []
    for epoch in range(1, epochs + 1):
        print('loop test')
        train(log_interval, model, train_loader, optimizer, epoch, train_acc)
        test(model, test_loader, test_acc)
    
    if (save_model):
        torch.save(model.state_dict(), "mnist_cnn.pt")

    return train_acc, test_acc, model