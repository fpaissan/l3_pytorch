from torch.utils.data import Dataset, DataLoader
from src.features.utils import *
import soundfile as sf
import resampy
import random
import os

class ESC50_Dataset(Dataset):
    """VGGSound dataset."""

    def __init__(self, data_dir):
      self.data_dir = data_dir    # Path to 'audio' folder
      self.file_list = os.listdir(data_dir)

    def __len__(self):          
        return len(self.file_list)

    def __getitem__(self, i):
      audio_path = os.path.join(self.data_dir, self.file_list[i])

      audioSignal, sr = sf.read(audio_path, dtype='int16', always_2d=True)
      audioSignal = audioSignal.mean(axis=-1).astype('int16')
      audioSignal = resampy.resample(audioSignal, sr, 48000)
      
      spectrograms = audio_feat(audioSignal, 48000)
      
      basename = file_list[i].split('.')[0]    # Actually not the exact fold division  

      class_label = int(basename.split('-')[-1])
      
      return spectrograms, class_label

if __name__ == "__main__":
  train_set = ESC50_Dataset("/scratch/gcerutti/VGGsound/data/Split/train/")
  example = train_set[1]
  print("{} {} {}".format(example[0].shape, example[1].shape, example[2].shape))

  dataloader = DataLoader(train_set, batch_size = 4, shuffle = True, num_workers = 1)

  for i_batch, batch in enumerate(dataloader):
    example = batch
    print("{} {} {}".format(example[0].shape, example[1].shape, example[2].shape))
  # playing with a dataloader
