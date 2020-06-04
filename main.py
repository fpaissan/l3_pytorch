import src.model.model_trainer as model_trainer
from src.model.model_trainer import avcNet_generator
import src.model.model_parameters as p
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torchvision
import torch
import os
import argparse
import pickle
import gzip
from tqdm import tqdm
torch.set_default_tensor_type(torch.FloatTensor)
import time
import os
#Net debugging
# X = []
# X.append(torch.tensor(np.ones(shape=(par.AVC_batchSize, par.AUDIO_C, par.AUDIO_H, par.AUDIO_W))))
# X.append(torch.tensor(np.ones(shape=(par.AVC_batchSize, par.VIDEO_C, par.VIDEO_H, par.VIDEO_W))))

# model_trainer.train(X, None)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Moves data from a single folder to train test folder')
    parser.add_argument('--feat-dir',
                        action='store',
                        type=str,
                        help='Path to directory where data files are stored')
    parser.add_argument('--log-dir',
                        action='store',
                        type=str,
                        help='Path to directory where data files are stored',
                        default = "/home/gcerutti/workspace/runs/")
    return parser.parse_args()

if __name__ == "__main__":
  args = parse_arguments()
  id_log = str(int(time.time()))
  # create train_dir and test_dir variables
  train_dir = os.path.join(args.feat_dir, 'train')
  test_dir = os.path.join(args.feat_dir, 'test')

  # initialize optimizer
  model = avcNet_generator() 
  optimizer = optim.Adam(model.parameters(), lr=p.AVC_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=p.AVC_weightdecay, amsgrad=False)
  criterion = nn.CrossEntropyLoss()
  
  # initialize summary writer
  os.system("rm -rd {}".format(args.log_dir))
  writer = SummaryWriter(args.log_dir)

  # list with batches
  train_batches = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
  test_batches = [os.path.join(test_dir, f) for f in os.listdir(test_dir)]
  for e in range(p.AVC_epochs):
    print("INFO: epoch {} of {}".format(e, p.AVC_epochs))
    loss, acc = list(), list()    
    for batch in tqdm(train_batches):
      with open(batch,"rb") as f:
        audio, video, label = pickle.load(f)
      loss_batch, acc_batch = model_trainer.train(audio, video, label, model, optimizer, criterion)
      loss.append(loss_batch)
      acc.append(acc_batch)
    writer.add_scalar('Loss/train_{}'.format(id_log), sum(loss)/len(loss), e)
    writer.add_scalar('Acc/train_{}'.format(id_log), sum(acc)/len(acc), e)
    
    loss, acc = list(), list()    
    for batch in tqdm(test_batches):
      with open(batch,"rb") as f:
        audio, video, label = pickle.load(f)
      loss_batch, acc_batch = model_trainer.test(audio, video, label, model, criterion)
      loss.append(loss_batch)
      acc.append(acc_batch)
    
    print(sum(acc)/len(acc))
    print(sum(loss)/len(loss))
    writer.add_scalar('Loss/test_{}'.format(id_log), sum(loss)/len(loss), e)
    writer.add_scalar('Acc/test_{}'.format(id_log), sum(acc)/len(acc), e)




