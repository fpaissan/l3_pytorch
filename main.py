import src.model.model_trainer as model_trainer
import src.parameters as par

import numpy as np
import torchvision
import torch

#Net debugging
X = []
X.append(torch.tensor(np.ones(shape=(par.AVC_batchSize, par.AUDIO_C, par.AUDIO_H, par.AUDIO_W))))
X.append(torch.tensor(np.ones(shape=(par.AVC_batchSize, par.VIDEO_C, par.VIDEO_H, par.VIDEO_W))))

model_trainer.train(X, None)