from src.features.avc_dataloader import VGGSound_Dataset
from src.model.avc_trainer import avcNet_generator
import src.model.avc_trainer as model_trainer
import src.model.model_parameters as p

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torchvision
import torch

from tqdm import tqdm
import numpy as np
import datetime
import argparse
import random
import pickle
import gzip
import time
import os

# CUDA_VISIBLE_DEVICES=0 python3.6 avc.py --data-dir /scratch/gcerutti/VGGsound/data/processed/

torch.set_default_tensor_type(torch.FloatTensor)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Moves data from a single folder to train test folder')
    parser.add_argument('--data-dir',
                        action='store',
                        type=str,
                        help='Path to directory where data files are stored')
    parser.add_argument('--log-dir',
                        action='store',
                        type=str,
                        help='Path to directory where data files are stored',
                        default = "/home/gcerutti/workspace/runs_AVC/")
    parser.add_argument('--ckp-dir',
                        action='store',
                        type=str,
                        help='Path to directory where data files are stored',
                        default = "/scratch/gcerutti/VGGsound/ckp/")
    parser.add_argument('--batch-size',
                    action='store',
                    type=int,
                    help='Path to directory where data files are stored',
                    default = 16)
    return parser.parse_args()

if __name__ == "__main__":
  args = parse_arguments()

  # year-month-day-hour-minute-second
  id_log = str(datetime.datetime.now()).split('.')[0].replace(" ", "-")  # create train_dir and test_dir variables
  
  train_dir = os.path.join(args.data_dir, 'train/*')
  test_dir = os.path.join(args.data_dir, 'test/*')
  os.makedirs(args.ckp_dir, exist_ok = True)

  # initialize optimizer 
  model = avcNet_generator()
  model.cuda()

  print(model)
  
  model.optimizer = optim.Adam(model.parameters(), lr=p.AVC_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=p.AVC_weightdecay, amsgrad=False)
  model.criterion = F.nll_loss
  # model.scheduler = optim.lr_scheduler.StepLR(model.optimizer, 16, gamma=0.94)

  # initialize summary writer
  os.system("rm -rd {}".format(args.log_dir))
  writer = SummaryWriter(args.log_dir, filename_suffix=id_log + "_AVC")

  # list with batches
  train_dataloader = DataLoader(VGGSound_Dataset(train_dir, transform=True), batch_size = args.batch_size, shuffle=True, num_workers=20)
  test_dataloader = DataLoader(VGGSound_Dataset(test_dir, transform=True), batch_size = args.batch_size, shuffle=True, num_workers=20)
  best_loss = np.inf


  for e in range(p.AVC_epochs):
    print("INFO: epoch {} of {}".format(e + 1, p.AVC_epochs))
    loss, acc = list(), list()    
    # to be replaced with dataloader
    for batch in tqdm(train_dataloader):
      audio, video, label = batch
      loss_batch, acc_batch = model_trainer.train(audio, video, label, model)
      loss.append(loss_batch)
      acc.append(acc_batch)
    writer.add_scalar('AVC_Loss/train_{}'.format(id_log), sum(loss)/len(loss), e)
    writer.add_scalar('AVC_Acc/train_{}'.format(id_log), sum(acc)/len(acc), e)
    
    loss, acc = list(), list()    
    for batch in tqdm(test_dataloader):
      audio, video, label = batch
      loss_batch, acc_batch = model_trainer.test(audio, video, label, model)
      loss.append(loss_batch)
      acc.append(acc_batch)

    # model.scheduler.step()    
    test_loss = sum(loss)/len(loss)
    if test_loss < best_loss:
      print("INFO: saving checkpoint!")
      best_loss = test_loss
      torch.save(model.state_dict(), os.path.join(args.ckp_dir, '{}_AVC_best_val.ckp'.format(id_log)))
    writer.add_scalar('AVC_Loss/test_{}'.format(id_log), sum(loss)/len(loss), e)
    writer.add_scalar('AVC_Acc/test_{}'.format(id_log), sum(acc)/len(acc), e)




