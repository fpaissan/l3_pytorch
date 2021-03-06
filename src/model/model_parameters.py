import numpy as np

MODEL_TYPE = 'default'

#Audio model parameters
AUDIO_W = 197
AUDIO_H = 257
AUDIO_C = 1
AUDIO_asr = 48000           # Hz
AUDIO_window_dur = 1        # Seconds
AUDIO_n_dft = 512
AUDIO_n_hop = 242
AUDIO_cmpMult = 1
AUDIO_channels = np.array([64, 128, 256])*AUDIO_cmpMult

#Video model parameters
VIDEO_W = 224
VIDEO_H = 224
VIDEO_C = 3
VIDEO_cmpMult = 1
VIDEO_channels = np.array([64, 128, 256])*VIDEO_cmpMult

#Merged model parameters
double_convolution = False
AVC_weightdecay = 1e-4
AVC_lr = 1e-4
AVC_epochs = 150
# AVC_batchSize = (GPU_Memory - GPU_Offset) * 1e9 / (VIDEO_C * VIDEO_H * VIDEO_W + AUDIO_C * AUDIO_H * AUDIO_W * 8)
# # Multiplied by 8 because Spectrogram is in double

#Classification parameters
NUM_CLASSES = {'esc50': 50}
CLASS_lr = 1e-4
CLASS_weightdecay = 1e-2
CLASS_epochs = 300
ESC_numWorkers = 10
CLASS_VALRATE = 0.15
