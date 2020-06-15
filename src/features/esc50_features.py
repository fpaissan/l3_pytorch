from src.model.avc_trainer import avcNet_generator
import src.features.features_parameters as feat_p
from src.features.utils import *

import torch

from tqdm import tqdm
import argparse
import resampy
import librosa
import inspect
import random
import pickle
import gzip
import sys
import os


def parse_arguments():
    parser = argparse.ArgumentParser(description='Moves data from a single folder to train test folder')
    parser.add_argument('--data-dir',
                        action='store',
                        type=str,
                        help='Path to directory where audio (wav) files are stored')

    parser.add_argument('--trained-model',
                        action='store',
                        type=str,
                        help='Path to directory where audio (wav) files are stored')        
 
    parser.add_argument('--output-dir',
                        action='store',
                        type=str,
                        help='Path to directory where output files will be stored')

    return parser.parse_args()

def audio_feat(audio, sr):
    hop_length = int(feat_p.ESC_hopsize * sr)
    frame_length = sr

    x = librosa.util.utils.frame(audio, frame_length=frame_length, hop_length=hop_length).T # Audio frames
    
    specs = list()
    for frame in x:
        specs.append(get_spectrogram(frame, sr, axis = 1))

    return np.asarray(specs, np.float32) # Maybe add a dimension

def extract_features(model, data_dir, output_dir, limit = -1):
    model.cuda()
    model.eval()
    for par in model.parameters():
      par.requires_grad = False

    file_list = os.listdir(data_dir)
    # audio_file_list = [audio_file_list[i].split('.')[:-1][0] for i in range(len(audio_file_list))]

    if limit == -1:
        limit = len(file_list)
    for i in tqdm(range(len(file_list))):
        audio_path = os.path.join(data_dir, file_list[i])

        audioSignal, sr = sf.read(audio_path, dtype='int16', always_2d=True)
        audioSignal = audioSignal.mean(axis=-1).astype('int16')
        audioSignal = resampy.resample(audioSignal, sr, 48000)
        
        spectrograms = audio_feat(audioSignal, 48000)

        spectrograms = torch.from_numpy(spectrograms)
        spectrograms = spectrograms.to("cuda")
        
        features = model.forward(spectrograms)
        features = features.cpu().numpy()

        basename = file_list[i].split('.')[0]
        class_label = int(basename.split('-')[-1])

        with open(os.path.join(output_dir, 'fold' + basename.split('-')[0], basename + '.pkl'), 'wb') as f:
            pickle.dump((features, class_label), f)

if __name__ == "__main__":
    args = parse_arguments()

    avcModel = avcNet_generator()
    avcModel.load_state_dict(torch.load(args.trained_model))

    for i in range(1, 6):
        os.makedirs(os.path.join(args.output_dir, 'fold' + str(i)), exist_ok=True)

    extract_features(avcModel.audioNet, args.data_dir, args.output_dir, limit = feat_p.ESC_limit)

