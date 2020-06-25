from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from src.features.utils import *
import soundfile as sf
from PIL import Image
import numpy as np
import resampy
import skimage
import random
import pickle
import glob
import os

def adjust_saturation(rgb_img, factor):
    """
        factor: Multiplicative scaling factor to be applied to saturation
    """
    hsv_img = skimage.color.rgb2hsv(rgb_img)
    imin, imax = skimage.dtype_limits(hsv_img, clip_negative = False)
    hsv_img[:,:,1] = np.clip(hsv_img[:,:,1] * factor, imin, imax)
    return skimage.color.hsv2rgb(hsv_img)


def adjust_brightness(rgb_img, delta):
    """
    delta: Additive (normalized) gain factor applied to each pixel
    """
    imin, imax = skimage.dtype_limits(rgb_img, clip_negative = False)
    # Convert delta into the range of the image data
    delta = rgb_img.dtype.type((imax - imin) * delta)
    return np.clip(rgb_img + delta, imin, imax)

def saturation_and_brightness(frame_data):
    if random.random() < 0.5:
        # Add saturation jitter
        saturation_factor = np.float32(random.random() + 0.5)
        frame_data = adjust_saturation(frame_data, saturation_factor)

        # Add brightness jitter
        max_delta = 32. / 255.
        brightness_delta = np.float32((2*random.random() - 1) * max_delta)
        frame_data = adjust_brightness(frame_data, brightness_delta)
    else:
        # Add brightness jitter
        max_delta = 32. / 255.
        brightness_delta = np.float32((2*random.random() - 1) * max_delta)
        frame_data = adjust_brightness(frame_data, brightness_delta)

        # Add saturation jitter
        saturation_factor = np.float32(random.random() + 0.5)
        frame_data = adjust_saturation(frame_data, saturation_factor)
    return frame_data


class VGGSound_Dataset(Dataset):
    """VGGSound dataset."""

    def __init__(self, data_dir, transform = False):
        self.file_list = glob.glob(data_dir)
        self.transform_compose = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            np.array,
            saturation_and_brightness,
            ])
        self.transform = transform
    def __len__(self):          
        return len(self.file_list)

    def __getitem__(self, i):
        with open(self.file_list[i], 'rb') as f:
            spectrogram, frame, label = pickle.load(f)
            frame = Image.fromarray(frame.astype('uint8'), 'RGB')
        if self.transform:
            frame = self.transform_compose(frame)
            frame = np.transpose(frame, (2,0,1))
        return (spectrogram, frame/255, label)

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
