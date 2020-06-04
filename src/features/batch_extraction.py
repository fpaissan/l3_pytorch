import features_parameters as par

from utils import *
import argparse
import random
import pickle
from tqdm import tqdm
import os
import gzip

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

    file_list = os.listdir(os.path.join(data_dir, 'audio'))
    audioBatch = []
    videoBatch = []
    labelBatch = []

    if limit == -1:
      limit = len(file_list)
    for i in tqdm(range(len(file_list[:limit]))):
        if(i % 2 == 0): #Save label 1
            audio_path = os.path.join(data_dir, 'audio', file_list[i].split('.')[:-1][0] + '.flac')
            video_path = os.path.join(data_dir, 'video', file_list[i].split('.')[:-1][0] + '.mp4')        

        else:           #Save label 0
            rand_i = random.randint(0, len(file_list))
            while(rand_i == i):
                rand_i = random.randint(0, len(file_list))

            audio_path = os.path.join(data_dir, 'audio', file_list[rand_i].split('.')[:-1][0] + '.flac')
            video_path = os.path.join(data_dir, 'video', file_list[i].split('.')[:-1][0] + '.mp4')
        
        frame, check = get_frame(video_path)
        if check == -1:
          continue
        spectrogram = get_spectrogram(audio_path)

        audioBatch.append(spectrogram)
        videoBatch.append(frame)
        labelBatch.append(np.asarray([i % 2, 1 - (i % 2)]))

        if(i % par.batchSize == (par.batchSize - 1)):
            audioBatch, videoBatch, labelBatch = \
            np.asarray(audioBatch, dtype = np.float32), np.asarray(videoBatch, dtype = np.float32), np.asarray(labelBatch, dtype = np.double) 

            batch = [audioBatch, videoBatch, labelBatch]
            with open(os.path.join(output_dir, 'batch_' + str(int(i / par.batchSize)) + '.pkl'), 'wb') as f:
                pickle.dump(batch, f)

            audioBatch = []
            videoBatch = []
            labelBatch = []


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

