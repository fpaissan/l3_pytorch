import features_parameters as par
from utils import *

from tqdm import tqdm
import argparse
import librosa
import random
import pickle
import gzip
import os

def parse_arguments():
    parser = argparse.ArgumentParser(description='Moves data from a single folder to train test folder')
    parser.add_argument('--data-dir',
                        action='store',
                        type=str,
                        help='Path to directory where audio (wav) files are stored')
 
    parser.add_argument('--output-dir',
                        action='store',
                        type=str,
                        help='Path to directory where output files will be stored')

    return parser.parse_args()

def audio_feat(audio, sr):
    hop_length = int(par.ESC50_hopsize * sr)
    frame_length = sr

    x = librosa.util.utils.frame(audio, frame_length=frame_length, hop_length=hop_length).T # Audio frames
    specs = list()
    for frame in x:
        specs.append(get_spectrogram(frame, sr))

    return np.asarray(specs, np.double) # Maybe add a dimension

def extract_features(data_dir, output_dir, limit = -1):
    file_list = os.listdir(data_dir)
    # audio_file_list = [audio_file_list[i].split('.')[:-1][0] for i in range(len(audio_file_list))]

    audioBatch = []
    labelBatch = []

    if limit == -1:
      limit = len(file_list)
    for i in tqdm(range(len(file_list))):
        audio_path = os.path.join(data_dir, 'audio', file_list[i] + '.wav')

        audioSignal, sr = sf.read(audio_path, dtype='int16', always_2d=True)
        audioSignal = audioSignal.mean(axis=-1).astype('int16')

        spectrograms = audio_feat(audioSignal, sr)

        audioBatch.append(spectrograms)

        class_label = int(file_list[i].split('-')[-1])
        labelBatch.append(class_label)

        if(i % par.ESC50_batchSize == (par.ESC50_batchSize - 1)):
            try:
              audioBatch, labelBatch = np.asarray(audioBatch, dtype = np.float32), np.asarray(labelBatch, dtype = np.double)

            except Exception as e:
              print(e)
              print(len(audioBatch))
              [print(audioBatch[i].shape) for i in range(len(audioBatch))]
              input(audio_path)
              audioBatch = []
              videoBatch = []
              labelBatch = []  
              continue

            batch = [audioBatch, videoBatch, labelBatch]
            with open(os.path.join(output_dir, 'batch_' + str(int(i / par.batchSize)) + '.pkl'), 'wb') as f:
                pickle.dump(batch, f)

            audioBatch = []
            videoBatch = []
            labelBatch = []


if __name__ == "__main__":
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    extract_features(args.data_dir, args.output_dir, limit = par.limit[s])

