import src.model.model_parameters as par
import torch.nn.functional as F

import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision
import torch


class ClassificationNet(nn.Module):
    def __init__(self, optimizer = None, criterion = None):
        super(ClassificationNet, self).__init__()
        self.optimizer = optimizer
        self.criterion = criterion

        self.lin0 = nn.Linear(512, 512)
        self.lin1 = nn.Linear(512, 128)
        self.lin2 = nn.Linear(128, par.NUM_CLASSES['esc50'])

        self.relu = nn.ReLU()
        self.soft = nn.Softmax()

    def forward(self, x):
        x = self.lin0(x)
        x = self.relu(x)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)

        return self.soft(x)

def train(audio, label, model):
    model.train()
    model.cuda()

    audio, label = audio.to("cuda"), label.to("cuda")
    model.optimizer.zero_grad()
    # (bs, 41, 1, 257, 196)
    pred = np.ndarray(shape=(len(label), par.NUM_CLASSES['esc50']))
    pred = torch.from_numpy(np.asarray(pred, dtype = np.float32))
    pred = pred.to("cuda")
    for i in range(audio.shape[0]):
        sample_out = model.forward(audio[i])
        sample_out = torch.mean(sample_out, dim=0)
        pred[i] = sample_out

    # label = torch.max(label, 1)[1]   #Remove one hot encoding  
    loss = model.criterion(pred, label)
    loss.backward()
    model.optimizer.step()
    
    pred = pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct = pred.eq(label.view_as(pred)).sum().item()/pred.shape[0]
    return loss.item(), correct # loss, accuracy 

def test(audio, label, model):
    model.eval() # chech with batch normalization
    model.cuda() 
    audio, label = audio.to("cuda"), label.to("cuda")

    # (bs, 41, 1, 257, 196)
    pred = np.ndarray(shape=(len(label), par.NUM_CLASSES['esc50']))
    pred = torch.from_numpy(np.asarray(pred, dtype = np.float32))
    pred = pred.to("cuda")
    for i in range(audio.shape[0]):
        sample_out = model.forward(audio[i])
        sample_out = torch.mean(sample_out, dim=0)
        pred[i] = sample_out
    
    #loss calculation
    # label = torch.max(label, 1)[1]
    loss = model.criterion(pred, label)

    # remove one hot encodind
    pred = pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct = pred.eq(label.view_as(pred)).sum().item()/pred.shape[0] # pred.shape[0] batch size

    return loss.item(), correct # loss, accuracy 
