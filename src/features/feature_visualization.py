import features_parameters as par

from utils import *
import argparse
import random
import pickle
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
#GM
# /scratch/gcerutti/VGGsound/data/Split
# /scratch/gcerutti/VGGsound/processed


def parse_arguments():
    parser = argparse.ArgumentParser(description='Moves data from a single folder to train test folder')
    parser.add_argument('--feat-dir',
                        action='store',
                        type=str,
                        help='Path to directory where features are stored')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    with open(os.path.join(args.feat_dir, 'train', 'batch_0.pkl'), 'rb') as f:
      audio, video, label = pickle.load(f)
    print("INFO: first element, related to AUDIO, is a list with lenght {}".format(len(audio)))
    print("INFO: second element, related to VIDEO, is a list with lenght {}".format(len(video)))
    print("INFO: third element, related to LABEL, is a list with lenght {}".format(len(label)))
    print("AUDIO: each element in the batch has shape {}".format(audio[0].shape))
    print("VIDEO: each element in the batch has shape {}".format(video[0].shape))
    print("LABEL: each element in the batch has shape {}".format(label[0].shape))
    
    for i in range(128):
      if label[i][0] == 1:
        plt.imshow(audio[i])
        plt.show()
        plt.imshow(video[i])
        plt.show()
