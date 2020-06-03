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


if par.MODEL_TYPE == 'default':
    class Net(nn.Module):
        def __init__(self, audioNet, visionNet):
            super(Net, self).__init__()
            self.visionNet = visionNet
            self.audioNet = audioNet
            self.lin1 = nn.Linear(1024, 128)
            self.lin2 = nn.Linear(128, 2)

            self.relu = nn.ReLU()
            self.soft = nn.Softmax()

        def forward(self, audioX, visionX):
            x = torch.cat((self.audioNet.forward(audioX.float()), self.visionNet.forward(visionX.float())), 1)
            x = self.lin1(x)
            x = self.relu(x)
            x = self.lin2(x)

            return self.soft(x)


def avcNet_generator():
  audioNet = audio_model.Net()
  visionNet = vision_model.Net()
  avcNet = Net(audioNet, visionNet)
  return avcNet

def train(audio, video, label, model, optimizer, criterion):
    model.train()
    model.cuda()

    audio, video, label = torch.from_numpy(audio), torch.from_numpy(video), torch.from_numpy(label) 
    audio, video, label = audio.to("cuda"), video.to("cuda"), label.to("cuda")
    optimizer.zero_grad()
    output = model(audio, video)

    loss = criterion(output, torch.max(label, 1)[1])
    loss.backward()
    optimizer.step()
    
    return loss.item(), 1 # loss, accuracy 
