from src.model.avc_trainer import avcNet_generator
import src.features.features_parameters as feat_p
from src.features.utils import *

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch

from tqdm import tqdm
import argparse
import pickle

import glob
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


def extract_scalers(data_dir, output_dir, limit=-1):
    file_list = glob.glob(data_dir)

    feat_list = list()
    for i in tqdm(range(len(file_list))):
        with open(file_list[i], 'rb') as fp:
            features, _ = pickle.load(fp)
        feat_list.append(features)

    feat_array = np.asarray(feat_list)
    feat_array = feat_array.reshape((-1, 512)) 
    print(feat_array.shape)
    minmaxScaler = MinMaxScaler()
    stdScaler = StandardScaler()

    feat_array = minmaxScaler.fit_transform(feat_array)
    stdScaler.fit(feat_array)

    with open(os.path.join(output_dir, 'scalers.pkl'), 'wb') as fp:
        pickle.dump((minmaxScaler, stdScaler), fp)

if __name__ == "__main__":
    args = parse_arguments()

    for i in range(5):
        extract_scalers(args.data_dir + '/fold{}/*'.format(i + 1), args.output_dir, limit=feat_p.ESC_limit)
