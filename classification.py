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
    avcModel = avcNet_generator()
    avcModel.load_state_dict(torch.load(args.trained_model, map_location=torch.device('cpu')))

    # Create classification model
    classModel = classification_trainer.ClassificationNet(avcModel.audioNet)

    classModel.classOptimizer = optim.Adam(classModel.parameters(), lr=p.CLASS_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=p.CLASS_weightdecay, amsgrad=False)
    classModel.classCriterion = nn.CrossEntropyLoss()

    # call train and test
