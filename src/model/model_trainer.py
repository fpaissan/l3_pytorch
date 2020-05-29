import src.model.vision_model as vision_model
import src.model.audio_model as audio_model
from .. import parameters as par

import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision
import torch

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
            x = torch.cat((self.audioNet.forward(audioX.double()), self.visionNet.forward(visionX.double())), 0)
            x = self.lin1(x)
            x = self.relu(x)
            x = self.lin2(x)

            return self.soft(x)

def train(X, y, batch_size=64, lr=1e-4, weigth_decay=1e-5):
    audioNet = audio_model.Net().double()
    visionNet = vision_model.Net().double()

    

    avcNet = Net(audioNet, visionNet).double()
    print("AVC Net output shape: ", avcNet.forward(X[0], X[1]).shape)
