import src.model.audio_model as audio_model
import src.model.model_parameters as par
import torch.nn.functional as F

import torch.optim as optim
import torch.nn as nn
import numpy as np
import torchvision
import torch
import torch.nn.functional as F

class ClassificationNet(nn.Module):
    def __init__(self, audioNet, optimizer, criterion):
        super(Net, self).__init__()
        self.optimizer = optimizer
        self.criterion = criterion

        self.audioNet = audioNet
        self.lin0 = nn.Linear(512, 512)
        self.lin1 = nn.Linear(512, 128)
        self.lin2 = nn.Linear(128, par.NUM_CLASSES['esc50'])

        self.relu = nn.ReLU()
        self.soft = nn.Softmax()

    def forward(self, audioX):
        x = self.audioNet.forward(audioX)

        x = self.lin0(x)
        x = self.relu(x)
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)

        return self.soft(x)