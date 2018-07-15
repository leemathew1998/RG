from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random
from resnet import *
from utils import progress_bar



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=0)

def poly(base_lr, epoch,max_iter=100,power=0.9):
    return base_lr*((1-float(epoch+1)/max_iter)**(power))
# def Cyclical_strategy(epoch):
#     if epoch<=50:
#         return epoch*2/50
#     else:
#         return 4 - 2*epoch/50
# def Normal_strategy(epoch):
#     if epoch<=50:
#         return 0.1
#     elif epoch>50 and epoch <=75:
#         return 0.01
#     else:
#         return 0.001
# def Cyclical_strategy_2_2(epoch):
#     if epoch<=25:
#         return 2*epoch/25
#     elif epoch>25 and epoch <=50:
#         return 4.1 - 2*epoch/25
#     elif epoch>50 and epoch <=75:
#         return 2*epoch/25 - 4
#     else:
#         return 8 - 2*epoch/25
# def Cyclical_strategy_4_2(epoch):
#     if epoch<=10 or epoch>20 and epoch <=30 or epoch>40 and epoch <=50 or epoch>60 and epoch <=70 or epoch>80 and epoch <=90:
#         if epoch%10 == 0:
#             return 2
#         else:
#             return epoch%10/5
#     else:
#         if epoch%10 == 0:
#             return 0.2
#         else:
#             return 2 - epoch%10/5
# list = [2/5, 4/5, 6/5, 8/5, 2, 2, 8/5, 6/5, 4/5, 2/5,
#        2/5, 4/5, 6/5, 8/5, 2, 2, 8/5, 6/5, 4/5, 2/5,
#        2/5, 4/5, 6/5, 8/5, 2, 2, 8/5, 6/5, 4/5, 2/5,
#        2/5, 4/5, 6/5, 8/5, 2, 2, 8/5, 6/5, 4/5, 2/5,
#        2/5, 4/5, 6/5, 8/5, 2, 2, 8/5, 6/5, 4/5, 2/5,
#        2/5, 4/5, 6/5, 8/5, 2, 2, 8/5, 6/5, 4/5, 2/5,
#        2/5, 4/5, 6/5, 8/5, 2, 2, 8/5, 6/5, 4/5, 2/5,
#        2/5, 4/5, 6/5, 8/5, 2, 2, 8/5, 6/5, 4/5, 2/5,
#        2/5, 4/5, 6/5, 8/5, 2, 2, 8/5, 6/5, 4/5, 2/5,
#        2/5, 4/5, 6/5, 8/5, 2, 2, 8/5, 6/5, 4/5, 2/5,]
# def Cyclical_strategy_10_2(epoch):
#     return list[epoch]
# list = [0.5, 1, 2, 1, 0.5,
#         0.5, 1, 2, 1, 0.5,
#         0.5, 1, 2, 1, 0.5,
#         0.5, 1, 2, 1, 0.5,
#         0.5, 1, 2, 1, 0.5,
#         0.5, 1, 2, 1, 0.5,
#         0.5, 1, 2, 1, 0.5,
#         0.5, 1, 2, 1, 0.5,
#         0.5, 1, 2, 1, 0.5,
#         0.5, 1, 2, 1, 0.5,
#         0.5, 1, 2, 1, 0.5,
#         0.5, 1, 2, 1, 0.5,
#         0.5, 1, 2, 1, 0.5,
#         0.5, 1, 2, 1, 0.5,
#         0.5, 1, 2, 1, 0.5,
#         0.5, 1, 2, 1, 0.5,
#         0.5, 1, 2, 1, 0.5,
#         0.5, 1, 2, 1, 0.5,
#         0.5, 1, 2, 1, 0.5,
#         0.5, 1, 2, 1, 0.5,]
# def Cyclical_strategy_20_2(epoch):
#     return list[epoch]
# Model
print('==> Building model..')
net = ResNet34()

if device == 'cuda':
    net.cuda()
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.95, weight_decay=0.0001)

M_loss = 0
# Training
def train(epoch):
    global M_loss
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss_ = loss * Cyclical_strategy_20_2(epoch)  # Here represents the Random Gradient
        # if batch_idx%2 == 0:
        #     M_loss = loss * random.random()
        #     M_loss.backward(retain_graph=True)
        # else:
        #     M_loss = M_loss + random.random() * loss
        #     M_loss.backward()
        loss_.backward()
        optimizer.step()
        train_loss += loss_.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        open('./test/Cyclical_strategy_20_2.txt', 'a').write(str(epoch) + '_' + str(acc) + ',')
        best_acc = acc
    print('best_acc:', best_acc)


def adjust_learning_rate(optimizer, epoch, net):
    lr = Cyclical_strategy_20_2(epoch)    # This is normal way to reduce the LR, you can replace it with CLR
    print('current lr: ', lr)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.95, weight_decay=0.0001)
if __name__ == '__main__':
    for epoch in range(0, 100):
        adjust_learning_rate(optimizer,epoch,net)
        train(epoch)
        test(epoch)
        
