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
            x = torch.cat((self.audioNet.forward(audioX.double()), self.visionNet.forward(visionX.double())), 0)
            x = self.lin1(x)
            x = self.relu(x)
            x = self.lin2(x)

            return self.soft(x)


def avcNet_generator():
  audioNet = audio_model.Net().double()
  visionNet = vision_model.Net().double()
  avcNet = Net(audioNet, visionNet).double()
  return avcNet

def train(audio, video, y, model, optimizer):
    model.train()
    model.cuda()

    audio, video, label = np.asarray(audio, dtype = np.double), np.asarray(video, dtype = np.double), np.asarray(y, dtype = np.double) 
    video = np.moveaxis(video, -1, 1)
    audio, video, label = torch.from_numpy(audio), torch.from_numpy(video), torch.from_numpy(label) 
    audio = audio.unsqueeze(1)
    audio, video, label = audio.to("cuda"), video.to("cuda"), label.to("cuda")

    optimizer.zero_grad()
    output = model(audio, video)
    loss = F.nll_loss(output, label)
    loss.backward()
    optimizer.step()
    print("INFO: loss {:.6f}".format(loss.item()))
    # print("AVC Net output shape: ", model.forward(X[0], X[1]).shape)
