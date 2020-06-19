import features_parameters as par

from tqdm import tqdm
from utils import *
import argparse
import resampy
import random
import pickle
import gzip
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description='Moves data from a single folder to train test folder')
    parser.add_argument('--data-dir',
                        action='store',
                        type=str,
                        help='Path to directory where data files are stored')
 
    parser.add_argument('--output-dir',
                        action='store',
                        type=str,
                        help='Path to directory where output files will be stored')

    return parser.parse_args()

def extract_features(data_dir, output_dir, limit = -1):

    audio_file_list = os.listdir(os.path.join(data_dir, 'audio'))
    video_file_list = os.listdir(os.path.join(data_dir, 'video'))

    audio_file_list = [audio_file_list[i].split('.')[:-1][0] for i in range(len(audio_file_list))]
    video_file_list = [video_file_list[i].split('.')[:-1][0] for i in range(len(video_file_list))]

    file_list = list((set(audio_file_list)).intersection( set(video_file_list)))
    file_list.sort()

    if limit == -1:
      limit = len(file_list)
    for i in tqdm(range(len(file_list))):
        if(i % 2 == 0): #Save label 1
            audio_path = os.path.join(data_dir, 'audio', file_list[i] + '.flac')
            video_path = os.path.join(data_dir, 'video', file_list[i] + '.mp4')        

        else:           #Save label 0
            rand_i = random.randint(0, len(file_list))
            while(rand_i == i):
                rand_i = random.randint(0, len(file_list))

            audio_path = os.path.join(data_dir, 'audio', file_list[rand_i] + '.flac')
            video_path = os.path.join(data_dir, 'video', file_list[i] + '.mp4')
        
        frame, check = get_frame(video_path)
        if check == -1:
          continue

        audioSignal, sr = sf.read(audio_path, dtype='int16', always_2d=True)
        audioSignal = resampy.resample(audioSignal, sr, 48000)
        spectrogram = get_spectrogram(audioSignal.mean(axis=-1).astype('int16'), sr)
        
        with open(os.path.join(output_dir, file_list[i] + '.pkl'), 'wb') as f:
            pickle.dump([spectrogram, frame, np.asarray([i % 2, 1 - (i % 2)])], f)


if __name__ == "__main__":
    args = parse_arguments()
    sub_folders = ['train', 'test']
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'test'), exist_ok=True)

    for s in sub_folders:
        data_dir = os.path.join(args.data_dir, s)
        output_dir = os.path.join(args.output_dir, s)
        extract_features(data_dir, output_dir, limit = par.limit[s])
