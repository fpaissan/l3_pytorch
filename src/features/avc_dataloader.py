from torch.utils.data import Dataset, DataLoader
from src.features.utils import *
import soundfile as sf
import resampy
import random
import pickle
import glob
import os

class VGGSound_Dataset(Dataset):
    """VGGSound dataset."""

    def __init__(self, data_dir):
      self.file_list = glob.glob(data_dir)
    
    def __len__(self):          
        return len(self.file_list)

    def __getitem__(self, i):
      with open(self.file_list[i], 'rb') as f:
            spectrogram, frame, label = pickle.load(f)

      return (spectrogram, frame[0]/255, label)

if __name__ == "__main__":
  train_set = VGGSound_Dataset("/media/fpaissan/DATA/Dataset/VGGSound/processed/train/*")
  # print(len(train_set))
  example = train_set[0]
  print("{} {} {}".format(example[0].shape, example[1].shape, example[2].shape))

  dataloader = DataLoader(train_set, batch_size = 4, shuffle = True, num_workers = 1)

  for i_batch, batch in enumerate(dataloader):
    example = batch
    print("{} {} {}".format(example[0].shape, example[1].shape, example[2].shape))
  # playing with a dataloader
