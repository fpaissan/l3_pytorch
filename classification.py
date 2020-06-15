import src.model.classification_trainer as model_trainer
from src.features.esc_dataloader import ESC50_Dataset
from src.model.avc_trainer import avcNet_generator
import src.model.model_parameters as p
import argparse

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch

from tqdm import tqdm
import numpy as np
import pickle
import random
import time
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Moves data from a single folder to train test folder')
    parser.add_argument('--data-dir',
                        action='store',
                        type=str,
                        help='Path to directory where fold folders are stored')
    parser.add_argument('--trained-model',
                        action='store',
                        type=str,
                        help='Path to trained model')
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

    id_log = str(int(time.time()))

    # Load model from checkpoint
    avcModel = avcNet_generator()
    avcModel.load_state_dict(torch.load(args.trained_model))

    # Create classification model
    classModel = model_trainer.ClassificationNet(avcModel.audioNet).double()

    classModel.optimizer = optim.Adam(classModel.parameters(), lr=p.CLASS_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=p.CLASS_weightdecay, amsgrad=False)
    classModel.criterion = nn.CrossEntropyLoss()

    # initialize summary writer
    os.system("rm -rd {}".format(args.log_dir))
    writer = SummaryWriter(args.log_dir)

    # Extract train/test/val folds
    folders_fold = os.listdir(args.data_dir)
    fold_idx = [0, 1, 2, 3, 4]
    for i, fold in enumerate(folders_fold):
        test_dataloader = DataLoader(ESC50_Dataset(args.data_dir, i), batch_size = args.batch_size, shuffle = True, num_workers = p.ESC_numWorkers)
        val_dataloader = DataLoader(ESC50_Dataset(args.data_dir, (i + 1) % len(folders_fold)), batch_size = args.batch_size, shuffle = True, num_workers = p.ESC_numWorkers)
        train_idx = list(set(fold_idx) - set([i, (i + 1) % len(folders_fold)]))
        train_dataloaders = [DataLoader(ESC50_Dataset(args.data_dir, train_idx[0]), batch_size = args.batch_size, shuffle = True, num_workers = p.ESC_numWorkers),
                            DataLoader(ESC50_Dataset(args.data_dir, train_idx[1]), batch_size = args.batch_size, shuffle = True, num_workers = p.ESC_numWorkers),
                            DataLoader(ESC50_Dataset(args.data_dir, train_idx[2]), batch_size = args.batch_size, shuffle = True, num_workers = p.ESC_numWorkers)]

        best_loss = np.inf
        for e in range(p.AVC_epochs):
            print("INFO: epoch {} of {}".format(e + 1, p.CLASS_epochs))

            for train_loader in train_dataloaders:
                loss, acc = list(), list()
                for batch in tqdm(train_loader):
                    audio, label = batch

                    loss_batch, acc_batch = model_trainer.train(audio.double(), label, classModel)
                    loss.append(loss_batch)
                    acc.append(acc_batch)
                
                writer.add_scalar('Loss/train_{}'.format(id_log), sum(loss)/len(loss), e)
                writer.add_scalar('Acc/train_{}'.format(id_log), sum(acc)/len(acc), e)
            
            loss, acc = list(), list()    
            for batch in tqdm(val_dataloader):
               with open(batch,"rb") as f:
                   audio, label = pickle.load(f)
               loss_batch, acc_batch = model_trainer.test(audio, label, classModel)
               loss.append(loss_batch)
               acc.append(acc_batch)
            
            val_loss = sum(loss)/len(loss)
            if val_loss < best_loss:
               print("INFO: saving checkpoint!")
               best_loss = val_loss
               torch.save(classModel.state_dict(), os.path.join(args.ckp_dir, 'CLASSIFICATION_best_val.ckp'))

            writer.add_scalar('Loss/test_{}'.format(id_log), sum(loss)/len(loss), e)
            writer.add_scalar('Acc/test_{}'.format(id_log), sum(acc)/len(acc), e)