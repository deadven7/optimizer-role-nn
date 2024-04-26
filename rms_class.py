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
        self.dropout = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.bn = nn.BatchNorm2d()
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


class RMS_TrainTest:
    def __init__(self, batch_size=64, test_batch_size=1000, epochs=10, lr=0.01, momentum=0, seed=1, log_interval=10, save_model=False, ratio=0.5):
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.lr = lr
        self.momentum = momentum
        self.seed = seed
        self.log_interval = log_interval
        self.save_model = save_model
        self.ratio = ratio
        self.train_loader, self.test_loader = self._prepare_data_loaders()
        self.model = Net()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.lr)
        self.train_loss = []
        self.test_loss = []
        self.train_acc = []
        self.test_acc = []

    def _prepare_data_loaders(self):
        torch.manual_seed(self.seed)
        kwargs = {'num_workers': 1, 'pin_memory': True}

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=self.batch_size, shuffle=True, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=self.test_batch_size, shuffle=True, **kwargs)

        return train_loader, test_loader

    def NN_model(self):
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train(epoch)
            self.test()
            self.train_loss.append(train_loss)

        if self.save_model:
            torch.save(self.model.state_dict(), "mnist_LeNet-5_rms.pt")

    def train(self, epoch):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = 100. * correct / len(data)
            self.train_acc.append(accuracy)
            if batch_idx % self.log_interval == 0:
                print('Epoch Set: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx *
                    len(data), len(self.train_loader.dataset),
                    100. * batch_idx / len(self.train_loader), loss.item()))
        train_loss /= len(self.train_loader.dataset)
        return train_loss

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_loader.dataset)
        accuracy = 100. * correct / len(self.test_loader.dataset)
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset), accuracy))
        return accuracy

    def plot_losses(self):
        plt.plot(range(1, self.epochs+1), self.train_loss, label='Training')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
