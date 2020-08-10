from torch.utils.data import Dataset, DataLoader
from src.features.utils import *
import soundfile as sf
import resampy
import random
import pickle
import glob
import os

class ESC50_Dataset(Dataset):
    """VGGSound dataset."""
    def __init__(self, data_dir, fold_idx):
        self.file_list = glob.glob(os.path.join(data_dir,"fold{}/*".format(fold_idx + 1)))
        with open(os.path.join(data_dir, 'scalers.pkl'), 'rb') as fp:
            self.minmaxScaler, self.stdScaler = pickle.load(fp) 
    def __len__(self):          
        return len(self.file_list)
    
    def __getitem__(self, i):
        with open(self.file_list[i],'rb') as f:
            embedding, label = pickle.load(f)
        
        return (self.stdScaler.transform(self.minmaxScaler.transform(embedding)), label)

if __name__ == "__main__":
    train_set = ESC50_Dataset("/scratch/gcerutti/ESC-50/processed",1)
    dataloader = DataLoader(train_set, batch_size = 1, shuffle = True, num_workers = 1)

    for i_batch, batch in enumerate(dataloader):
        example = batch
        print("{} {}".format(example[0].shape, example[1].shape))
    
