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
                        default=None,
                        help='Path to directory where audio (wav) files are stored')        
 
    parser.add_argument('--output-dir',
                        action='store',
                        type=str,
                        help='Path to directory where output files will be stored')

    return parser.parse_args()

def extract_features(model, data_dir, output_dir, limit = -1, is_openl3 = False):
    if(model != None):
        model = model.audioNet
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
        
        spectrograms = audio_feat(audioSignal, 48000, is_openl3=is_openl3)

        if(not is_openl3):
            spectrograms = torch.from_numpy(spectrograms)
            spectrograms = spectrograms.to("cuda")
        
            features = model.forward(spectrograms)
            features = features.cpu().numpy()
        else:
            features = spectrograms

        basename = file_list[i].split('.')[0]
        class_label = int(basename.split('-')[-1])

        with open(os.path.join(output_dir, 'fold' + basename.split('-')[0], basename + '.pkl'), 'wb') as f:
            pickle.dump((features, class_label), f)

if __name__ == "__main__":
    args = parse_arguments()

    if(not args.trained_model == None):
        avcModel = avcNet_generator()
        avcModel.load_state_dict(torch.load(args.trained_model))

    else:
        avcModel = None
        is_openl3 = True

    for i in range(1, 6):
        os.makedirs(os.path.join(args.output_dir, 'fold' + str(i)), exist_ok=True)

    extract_features(avcModel, args.data_dir, args.output_dir, limit = feat_p.ESC_limit, is_openl3 = is_openl3)

