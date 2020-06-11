import os
from torch.utils.data import Dataset, DataLoader
import random
from src.features.utils import *
import resampy
import soundfile as sf

class VGGSound_Dataset(Dataset):
    """VGGSound dataset."""

    def __init__(self, data_dir):

      # data_path already specify train or test
      self.data_dir = data_dir
      audio_file_list = os.listdir(os.path.join(data_dir, 'audio'))
      video_file_list = os.listdir(os.path.join(data_dir, 'video'))

      audio_file_list = [audio_file_list[i].split('.')[:-1][0] for i in range(len(audio_file_list))]
      video_file_list = [video_file_list[i].split('.')[:-1][0] for i in range(len(video_file_list))]

      self.file_list = list((set(audio_file_list)).intersection( set(video_file_list)))
      self.file_list.sort()
    
    def __len__(self):          
        return len(self.file_list)

    def __getitem__(self, i):
      not_check = True
      while(not_check):
        if(i % 2 == 0): #Save label 1
          audio_path = os.path.join(self.data_dir, 'audio', self.file_list[i] + '.flac')
          video_path = os.path.join(self.data_dir, 'video', self.file_list[i] + '.mp4')        

        else:           #Save label 0
          
          rand_i = random.randint(0, len(self.file_list) - 1)
          while(rand_i == i):
              rand_i = random.randint(0, len(self.file_list) - 1)

          audio_path = os.path.join(self.data_dir, 'audio', self.file_list[rand_i] + '.flac')
          video_path = os.path.join(self.data_dir, 'video', self.file_list[i] + '.mp4')

        frame, check = get_frame(video_path)
        if check == -1:
          print("ERROR")
          not_check = True
        else:
          not_check = False
        
        audioSignal, sr = sf.read(audio_path, dtype='int16', always_2d=True)
        audioSignal = resampy.resample(audioSignal, sr, 48000)
        spectrogram = get_spectrogram(audioSignal.mean(axis=-1).astype('int16'), sr)

      return (spectrogram, frame, np.asarray([i % 2, 1 - (i % 2)]))

# if __name__ == "__main__":
#   train_set = VGGSound_Dataset("/scratch/gcerutti/VGGsound/data/Split/train/")
#   example = train_set[1]  
#   print("{} {} {}".format(example[0].shape, example[1].shape, example[2].shape))

#   dataloader = DataLoader(train_set, batch_size = 4, shuffle = True, num_workers = 1)

#   for i_batch, batch in enumerate(dataloader):
#     example = batch
#     print("{} {} {}".format(example[0].shape, example[1].shape, example[2].shape))
#   # playing with a dataloader
