# -*- coding: utf-8 -*-
"""
creating a network for training Densenet on HAM10000 dataset

Pretrained = False

test:train = 75:25

Classes = 7
"""


from __future__ import print_function, division
import os
import torch
#from skimage import io, transform
import numpy as np
#import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils,models
import torch.optim as optim
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.autograd import Variable
import warnings
import nonechucks as nc
warnings.filterwarnings("ignore")
device = torch.device("cuda:0")

def load_dataset():
    data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    path = './HAM_dataset_work'
    dataset_total1 = torchvision.datasets.ImageFolder(root=path,transform=data_transform)
    dataset_total = nc.SafeDataset(dataset_total1)
    train_size = int(0.75*len(dataset_total))
    test_size = len(dataset_total) - train_size
    train_dataset,test_dataset = torch.utils.data.random_split(dataset_total,[train_size,test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=64, num_workers=0, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size=64,num_workers=0,shuffle=True)
    return train_loader,test_loader

if __name__ == '__main__':
    train_loader,test_loader = load_dataset()
    print(len(train_loader))
    
    densenet = torchvision.models.densenet169(pretrained=False)
    densenet.to(device)
    for param in densenet.parameters():
        param.requires_grad = True
    densenet.classifier = nn.Linear(1664, 7)
    
    #densenet = torchvision.models.resnet18(pretrained=True)
    #num_ftrs = densenet.fc.in_features
    #densenet.fc = nn.Linear(num_ftrs, 198)

    #densenet = densenet.to(device)
    #for param in densenet.parameters():
    #    param.requires_grad = True
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer_conv = optim.SGD(densenet.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    loss_curve = []   
    for epoch in range(25):  # loop over the dataset multiple times
        print('*----Epoch is----* ',epoch)
        i=0
        running_loss = 0.0
        running_corrects = 0.0
        for data in train_loader:
            #print('in here!')
            # get the inputs
            inputs, labels = data
            inputs,labels = Variable(inputs),Variable(labels)
            inputs,labels = inputs.to(device),labels.to(device)
            # zero the parameter gradients
            optimizer_conv.zero_grad()
    
            # forward + backward + optimize
            outputs = densenet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_conv.step()
    
            # print statistics
            running_loss += loss.item() * inputs.size(0)
            if i % 2000 == 1999: # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        loss_curve.append(running_loss)
        i+=1
            
    print('Finished Training')
    torch.save(densenet.state_dict(),'./models/HAM_dense_ep50.pth')
    #testing on test set to get the accuracy
    correctly_pred = 0
    total = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data in test_loader:
            inputs,labels = data
            inputs,labels = Variable(inputs),Variable(labels)
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = densenet(inputs)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correctly_pred += (predict == labels).sum().item()
            y_true.append(labels)
            y_pred.append(predict)
                
    accuracy = (correctly_pred/total)*100
    
    print('The testing accuracy for HAM is ',accuracy)
    
    #lets get the training accuracy as well
    correctly_pred = 0
    total = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data in train_loader:
            inputs,labels = data
            inputs,labels = Variable(inputs),Variable(labels)
            inputs,labels = inputs.to(device),labels.to(device)
            outputs = densenet(inputs)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correctly_pred += (predict == labels).sum().item()
            y_true.append(labels)
            y_pred.append(predict)
                
    accuracy = (correctly_pred/total)*100
    
    print('The training accuracy for HAM is',accuracy)

    loss_save = np.array(loss_curve)
    np.save('HAM_loss_curve_densenet_50epochs.npy',loss_save)
#    epochs = np.arange(2)
#    plt.plot(epochs,loss_curve)
#    plt.xlabel('epochs')
#    
    #printing confusion matrix
    
#    classes = 198

#    confusion_matrix = torch.zeros(classes, classes)
#    with torch.no_grad():
#        for i, (inputs, classes) in enumerate(test_loader):
#            inputs = Variable(inputs)
#            classes = Variable(classes)
#            outputs = densenet(inputs)
#            _, preds = torch.max(outputs, 1)
#            for t, p in zip(classes.view(-1), preds.view(-1)):
#                    confusion_matrix[t.long(), p.long()] += 1
#    
#    print(confusion_matrix)

