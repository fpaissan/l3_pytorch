import src.model.model_trainer as model_trainer
import src.parameters as par # to fix
from src.model.model_trainer import avcNet
import src.model.model_parameters as p # to fix
import torch.optim as optim

import numpy as np
import torchvision
import torch
import os
import argparse
import pickle

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
    return parser.parse_args()

if __name__ == "__main__":
  args = parse_arguments()

  # create train_dir and test_dir variables
  train_dir = os.path.join(args.feat_dir, 'train')
  test_dir = os.path.join(args.feat_dir, 'test')

  # initialize optimizer
  model = avcNet() 
  optimizer = optim.Adadelta(model.parameters(), lr=p.lr)  

  # list with batches
  train_batches = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]

  for batch in train_batches:
    with open(batch,"rb") as f:
      audio, video, label = pickle.load(f)
    model_trainer.train(audio, video, label, model, optimizer)
    

