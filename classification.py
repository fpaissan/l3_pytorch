import src.model.classification_trainer as model_trainer
from src.features.esc_dataloader import ESC50_Dataset
from src.model.avc_trainer import avcNet_generator
import src.model.model_parameters as p
import argparse

from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import torch.optim as optim
import torch.nn as nn
import torch

from tqdm import tqdm
import numpy as np
import datetime
import pickle
import random
import time
import os

# CUDA_VISIBLE_DEVICES=3 python3.6 classification.py --data-dir /scratch/gcerutti/VGGsound/data/open_l3

def parse_arguments():
    parser = argparse.ArgumentParser(description='Moves data from a single folder to train test folder')
    parser.add_argument('--data-dir',
                        action='store',
                        type=str,
                        help='Path to directory where fold folders are stored')
    parser.add_argument('--batch-size',
                        action='store',
                        type=int,
                        default=16)
    parser.add_argument('--log-dir',
                        action='store',
                        type=str,
                        help='Path to directory where data files are stored',
                        default = "/home/gcerutti/workspace/runs/")
    parser.add_argument('--ckp-dir',
                        action='store',
                        type=str,
                        help='Path to directory where data files are stored',
                        default = "/scratch/gcerutti/VGGsound/ckp/")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # year-month-day-hour-minute-second
    id_log = str(datetime.datetime.now()).split('.')[0].replace(" ", "-")  # create train_dir and test_dir variables

    # initialize summary writer
    os.system("rm -rd {}".format(args.log_dir))
    writer = SummaryWriter(args.log_dir)

    # Extract train/test/val folds
    test_acc = list()
    fold_idx = [0, 1, 2, 3, 4]
    for fold in fold_idx:
        classModel = model_trainer.ClassificationNet()

        classModel.optimizer = optim.Adam(classModel.parameters(), lr=p.CLASS_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=p.CLASS_weightdecay, amsgrad=False)
        classModel.criterion = F.nll_loss

        test_dataloader = DataLoader(ESC50_Dataset(args.data_dir, [fold]), batch_size = args.batch_size, shuffle = True, num_workers = p.ESC_numWorkers)
        train_val_idx = list(set(fold_idx) - set([fold]))
        esc50_dataset = ESC50_Dataset(args.data_dir, train_val_idx)
        train_dataset, val_dataset = random_split(esc50_dataset, [int(np.floor((1 - p.CLASS_VALRATE) * len(esc50_dataset))), int(np.ceil(p.CLASS_VALRATE * len(esc50_dataset)))])
        print("[INFO]: {} {} {}".format(len(esc50_dataset), len(train_dataset), len(val_dataset)))
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=p.ESC_numWorkers)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=p.ESC_numWorkers)

        best_loss = np.inf
        for e in range(p.CLASS_epochs):
            print("INFO: epoch {} of {}".format(e + 1, p.CLASS_epochs))

            loss, acc = list(), list()
            for batch in tqdm(train_dataloader):
                embedding, label = batch

                loss_batch, acc_batch = model_trainer.train(embedding, label, classModel)
                loss.append(loss_batch)
                acc.append(acc_batch)
                
            writer.add_scalar('Loss_{}/train_{}'.format(fold, id_log), sum(loss)/len(loss), e)
            writer.add_scalar('Acc_{}/train_{}'.format(fold, id_log), sum(acc)/len(acc), e)
            
            loss, acc = list(), list()    
            for batch in tqdm(val_dataloader):
                embedding, label = batch
                loss_batch, acc_batch = model_trainer.test(embedding, label, classModel)
                loss.append(loss_batch)
                acc.append(acc_batch)
            
            val_loss = sum(loss)/len(loss)
            if val_loss < best_loss:
               print("INFO: saving checkpoint!")
               best_loss = val_loss
               torch.save(classModel.state_dict(), os.path.join(args.ckp_dir, 'CLASSIFICATION_best_val_{}.ckp'.format(fold)))

            writer.add_scalar('Loss_{}/val_{}'.format(fold, id_log), sum(loss)/len(loss), e)
            writer.add_scalar('Acc_{}/val_{}'.format(fold, id_log), sum(acc)/len(acc), e)

        for batch in tqdm(test_dataloader):
            embedding, label = batch
            loss_batch, acc_batch = model_trainer.train(embedding, label, classModel)
            loss.append(loss_batch)
            acc.append(acc_batch)

        test_acc.append(sum(acc) / len(acc))

    print("[INFO] Test accuracy vector: {}".format(test_acc))
    print("[INFO] Average test accuracy is {}".format(sum(test_acc) / len(test_acc)))