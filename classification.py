import src.model.classification_trainer as classification_trainer
from src.model.avc_trainer import avcNet_generator
import src.model.model_parameters as p
import argparse

import torch.optim as optim
import torch.nn as nn
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description='Moves data from a single folder to train test folder')
    parser.add_argument('--trained-model',
                        action='store',
                        type=str,
                        help='Path to trained model')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Load model from checkpoint
    avcOptimizer = optim.Adam(model.parameters(), lr=p.AVC_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=p.AVC_weightdecay, amsgrad=False)
    avcCriterion = nn.CrossEntropyLoss()

    avcModel = avcNet_generator(avcOptimizer, avcCriterion)
    avcModel.load_state_dict(torch.load(args.trained_model))

    # Create classification model
    classOptimizer = optim.Adam(model.parameters(), lr=p.AVC_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=p.AVC_weightdecay, amsgrad=False)
    classCriterion = nn.CrossEntropyLoss()

    classModel = classification_trainer.ClassificationNet(avcModel.audioNet, classOptimizer, classCriterion)

