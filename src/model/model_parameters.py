import numpy as np

MODEL_TYPE = 'default'

#Audio model parameters
AUDIO_weightdecay = 1e-5
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
VIDEO_weightdecay = 1e-5
VIDEO_W = 224
VIDEO_H = 224
VIDEO_C = 3
VIDEO_cmpMult = 1
VIDEO_channels = np.array([64, 128, 256])*VIDEO_cmpMult

#Merged model parameters
AVC_weightdecay = 0
AVC_lr = 1e-4
AVC_epochs = 1000
# AVC_batchSize = (GPU_Memory - GPU_Offset) * 1e9 / (VIDEO_C * VIDEO_H * VIDEO_W + AUDIO_C * AUDIO_H * AUDIO_W * 8)
# # Multiplied by 8 because Spectrogram is in double

#Classification parameters
NUM_CLASSES = {'esc50': 50}
CLASS_lr = 1e-4
CLASS_weightdecay = 0
