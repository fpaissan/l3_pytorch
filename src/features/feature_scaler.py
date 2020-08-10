from src.model.avc_trainer import avcNet_generator
import src.features.features_parameters as feat_p
from src.features.utils import *

import torch

from tqdm import tqdm
import argparse
import pickle
import gzip
import sys
import os

import glob

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


def extract_scalers(data_dir, output_dir, limit=-1):
    file_list = glob.glob(data_dir)

    feat_list = list()
    for i in tqdm(range(len(file_list))):
        with open(file_list[i], 'rb') as fp:
            features, _ = pickle.load(fp)
        feat_list.append(features)

    feat_array = np.asarray(feat_list)

    print(len(file_list), feat_array.shape)

    #     audio_path = os.path.join(data_dir, file_list[i])
    #
    #     audioSignal, sr = sf.read(audio_path, dtype='int16', always_2d=True)
    #     audioSignal = audioSignal.mean(axis=-1).astype('int16')
    #     audioSignal = resampy.resample(audioSignal, sr, 48000)
    #
    #     if (not is_openl3):
    #         spectrograms = audio_feat(audioSignal, 48000, is_openl3=is_openl3)
    #
    #     if (not is_openl3):
    #         spectrograms = torch.from_numpy(spectrograms)
    #         spectrograms = spectrograms.to("cuda")
    #
    #         features = model.forward(spectrograms)
    #         features = features.cpu().numpy()
    #
    #     basename = file_list[i].split('.')[0]
    #     class_label = int(basename.split('-')[-1])
    #
    #     if (not is_openl3):
    #         with open(os.path.join(output_dir, 'fold' + basename.split('-')[0], basename + '.pkl'), 'wb') as f:
    #             pickle.dump((features, class_label), f)
    #
    #     if (is_openl3):
    #         # Creating audio list to avoid GPU memory overflow while inferencing
    #         audioSignal_folds[basename.split('-')[0]].append(audioSignal)
    #         classLabel_folds[basename.split('-')[0]].append(class_label)
    #
    # for fold in range(5):  # iterate in folds
    #     spectrograms_openL3 = audio_feat(audioSignal_folds[str(fold + 1)], 48000, is_openl3=is_openl3)
    #     for x, sample in enumerate(spectrograms_openL3):
    #         with open(os.path.join(output_dir, 'fold{}'.format(fold + 1), '{}.pkl'.format(x)), 'wb') as f:
    #             pickle.dump((sample, classLabel_folds[str(fold + 1)][x]), f)

if __name__ == "__main__":
    args = parse_arguments()

    for i in range(5):
        extract_scalers(args.data_dir + '/fold{}/*'.format(i + 1), args.output_dir, limit=feat_p.ESC_limit)
