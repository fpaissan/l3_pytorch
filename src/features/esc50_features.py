import features_parameters as par
from utils import *

from tqdm import tqdm
import argparse
import resampy
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
    hop_length = int(par.ESC_hopsize * sr)
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
        audio_path = os.path.join(data_dir, file_list[i])

        audioSignal, sr = sf.read(audio_path, dtype='int16', always_2d=True)
        audioSignal = audioSignal.mean(axis=-1).astype('int16')
        audioSignal = resampy.resample(audioSignal, sr, 48000)
        
        spectrograms = audio_feat(audioSignal, 48000)

        audioBatch.append(spectrograms)
        
        basename = file_list[i].split('.')[0]  

        class_label = int(basename.split('-')[-1])
        labelBatch.append(class_label)

        if(i % par.ESC_batchsize == (par.ESC_batchsize - 1)):
            try:
                audioBatch, labelBatch = np.asarray(audioBatch, dtype = np.float32), np.asarray(labelBatch, dtype = np.double)

            except Exception as e:
                print(e)
                audioBatch = []
                labelBatch = []  
                continue

            batch = [audioBatch, labelBatch]
            with open(os.path.join(output_dir, 'fold' + basename.split('-')[0], 'batch_' + str(int(i / par.batchSize)) + '.pkl'), 'wb') as f:
                pickle.dump(batch, f)

            audioBatch = []
            labelBatch = []

    try:
        audioBatch, labelBatch = np.asarray(audioBatch, dtype = np.float32), np.asarray(labelBatch, dtype = np.double)

    except Exception as e:
        print(e)

    batch = [audioBatch, labelBatch]
    with open(os.path.join(output_dir, 'fold' + basename.split('-')[0], 'batch_' + str(int(i / par.batchSize)) + '.pkl'), 'wb') as f:
        pickle.dump(batch, f)

if __name__ == "__main__":
    args = parse_arguments()
    
    for i in range(5):
        os.makedirs(os.path.join(args.output_dir, 'fold' + str(i)), exist_ok=True)

    extract_features(args.data_dir, args.output_dir, limit = par.ESC_limit)

