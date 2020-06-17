import src.model.model_parameters as par

import torch.nn as nn
import torch

'''
Model from R. Arandjelovic et al (2017) - Look, Listen and Learn.
'''
class Net(nn.Module):
    def __init__(self, optimizer = None, criterion = None):
        super(Net, self).__init__()
        self.optimizer = optimizer
        self.criterion = criterion

        self.conv1 = nn.Conv2d(1, par.AUDIO_channels[0], kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(par.AUDIO_channels[0], eps=0.001, momentum=0.99)    #Eps and momentum from keras default
        if par.double_convolution:
          self.conv1B = nn.Conv2d(par.AUDIO_channels[0], par.AUDIO_channels[0], kernel_size=3, stride=1, padding=1)
          self.conv1B_bn = nn.BatchNorm2d(par.AUDIO_channels[0], eps=0.001, momentum=0.99)    #Eps and momentum from keras default

        self.conv2 = nn.Conv2d(par.AUDIO_channels[0], par.AUDIO_channels[1], kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(par.AUDIO_channels[1], eps=0.001, momentum=0.99)    #Eps and momentum from keras default
        if par.double_convolution:
          self.conv2B = nn.Conv2d(par.AUDIO_channels[1], par.AUDIO_channels[1], kernel_size=3, stride=1, padding=1)
          self.conv2B_bn = nn.BatchNorm2d(par.AUDIO_channels[1], eps=0.001, momentum=0.99)    #Eps and momentum from keras default

        self.conv3 = nn.Conv2d(par.AUDIO_channels[1], par.AUDIO_channels[2], kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(par.AUDIO_channels[2], eps=0.001, momentum=0.99)    #Eps and momentum from keras default
        if par.double_convolution:
          self.conv3B = nn.Conv2d(par.AUDIO_channels[2], par.AUDIO_channels[2], kernel_size=3, stride=1, padding=1)
          self.conv3B_bn = nn.BatchNorm2d(par.AUDIO_channels[2], eps=0.001, momentum=0.99)    #Eps and momentum from keras default

        self.conv4 = nn.Conv2d(par.AUDIO_channels[2], 512, kernel_size=3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(512, eps=0.001, momentum=0.99)    #Eps and momentum from keras default
        if par.double_convolution:
          self.conv4B = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
          self.conv4B_bn = nn.BatchNorm2d(512, eps=0.001, momentum=0.99)    #Eps and momentum from keras default         
        self.maxpool_4 = nn.MaxPool2d(kernel_size=(32, 24), stride=None)
        
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        # 1st conv block
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.relu(x)

        if par.double_convolution:
          x = self.conv1B(x)
          x = self.conv1B_bn(x)
          x = self.relu(x)

        x = self.maxpool(x)

        # 2nd conv block
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.relu(x)

        if par.double_convolution:
          x = self.conv2B(x)
          x = self.conv2B_bn(x)
          x = self.relu(x)

        x = self.maxpool(x)

        # 3rd conv block
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = self.relu(x)

        if par.double_convolution:
          x = self.conv3B(x)
          x = self.conv3B_bn(x)
          x = self.relu(x)

        x = self.maxpool(x)

        # 4th conv block
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = self.relu(x)

        if par.double_convolution:
          x = self.conv4B(x)
          x = self.conv4B_bn(x)
          x = self.relu(x)

        x = self.maxpool_4(x)
                    
        return torch.flatten(x, 1)