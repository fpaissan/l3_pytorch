import src.model.model_parameters as par

import torch.nn as nn
import torch

#from kapre.time_frequency import Spectrogram

if par.MODEL_TYPE == 'default':
    '''
    Model from R. Arandjelovic et al (2017) - Look, Listen and Learn.
    '''
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # ToDo: Move on preprocessing
            # self.Spect = Spectrogram(n_dft=par.AUDIO_n_dft, n_hop=par.AUDIO_n_hop, power_spectrogram=1.0, return_decibel_spectrogram=False, padding='valid')
            # self.L3_Norm = lambda x: torch.log(torch.max(x, 1e-12) / 5.0)

            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
            self.conv1B = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
            self.conv1_bn = nn.BatchNorm2d(64, eps=0.001, momentum=0.99)    #Eps and momentum from keras default

            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.conv2B = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
            self.conv2_bn = nn.BatchNorm2d(128, eps=0.001, momentum=0.99)    #Eps and momentum from keras default

            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            self.conv3B = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.conv3_bn = nn.BatchNorm2d(256, eps=0.001, momentum=0.99)    #Eps and momentum from keras default

            self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
            self.conv4B = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.conv4_bn = nn.BatchNorm2d(512, eps=0.001, momentum=0.99)    #Eps and momentum from keras default
            self.maxpool_4 = nn.MaxPool2d(kernel_size=(32, 24), stride=None)
            
            self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
            self.relu = nn.ReLU()

        def forward(self, x):
            #x = self.Spect(x)
            #x = self.L3_Norm(x)

            # 1st conv block
            x = self.conv1(x)
            x = self.conv1_bn(x)
            x = self.relu(x)

            x = self.conv1B(x)
            x = self.conv1_bn(x)
            x = self.relu(x)

            x = self.maxpool(x)

            # 2nd conv block
            x = self.conv2(x)
            x = self.conv2_bn(x)
            x = self.relu(x)

            x = self.conv2B(x)
            x = self.conv2_bn(x)
            x = self.relu(x)

            x = self.maxpool(x)

            # 3rd conv block
            x = self.conv3(x)
            x = self.conv3_bn(x)
            x = self.relu(x)

            x = self.conv3B(x)
            x = self.conv3_bn(x)
            x = self.relu(x)

            x = self.maxpool(x)

            # 4th conv block
            x = self.conv4(x)
            x = self.conv4_bn(x)
            x = self.relu(x)

            x = self.conv4B(x)
            x = self.conv4_bn(x)
            x = self.relu(x)

            x = self.maxpool_4(x)
                        
            return torch.flatten(x, 1)