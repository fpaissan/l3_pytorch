import src.model.vision_model as vision_model
import src.model.audio_model as audio_model
import src.model.model_parameters as par
import torch.nn.functional as F

import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision
import torch
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, audioNet, visionNet, optimizer = None, criterion = None, scheduler = None):
        super(Net, self).__init__()
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler

        self.visionNet = visionNet
        self.audioNet = audioNet
        self.lin1 = nn.Linear(1024, 128)
        self.lin2 = nn.Linear(128, 2)

        self.relu = nn.ReLU()

    def forward(self, audioX, visionX):
        x = torch.cat((self.audioNet.forward(audioX.float()), self.visionNet.forward(visionX.float())), 1)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)


def avcNet_generator(optimizer = None, criterion = None):
  audioNet = audio_model.Net()
  visionNet = vision_model.Net()
  avcNet = Net(audioNet, visionNet, optimizer, criterion)
  return avcNet

def train(audio, video, label, model):
    model.train()
    model.cuda()

    audio, video, label = audio.to("cuda"), video.to("cuda"), label.to("cuda")
    model.optimizer.zero_grad()
    output = model.forward(audio, video)

    # remove one hot encodind
    label = torch.max(label, 1)[1]

    loss = model.criterion(output, label)
    loss.backward()
    model.optimizer.step()
    
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct = pred.eq(label.view_as(pred)).sum().item()/pred.shape[0]
    return loss.item(), correct # loss, accuracy 

def test(audio, video, label, model):
    model.eval() # chech with batch normalization
    model.cuda()
    audio, video, label = audio.to("cuda"), video.to("cuda"), label.to("cuda")
    output = model.forward(audio, video)

    #loss calculation
    label = torch.max(label, 1)[1]
    loss = model.criterion(output, label)

    # remove one hot encodind
    pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    correct = pred.eq(label.view_as(pred)).sum().item()/pred.shape[0] # pred.shape[0] batch size

    return loss.item(), correct # loss, accuracy 
