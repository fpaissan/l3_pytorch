import numpy as np

# Data/Feature
GPU_Memory = 4  #GB
GPU_Offset = 3  #GB

AUDIO_W = 197
AUDIO_H = 257
AUDIO_C = 1

VIDEO_W = 224
VIDEO_H = 224
VIDEO_C = 3

AUDIO_asr = 48000           # Hz
AUDIO_window_dur = 1        # Seconds
AUDIO_n_dft = 512
AUDIO_n_hop = 242

DUMMY_PARAMETER = 5.0

batchSize = 2 #int(np.floor((GPU_Memory - GPU_Offset) * 1e9 / (VIDEO_C * VIDEO_H * VIDEO_W + AUDIO_C * AUDIO_H * AUDIO_W * 8)))

limit = {'train': -1,
         'test': -1}


# ESC-50 related
ESC_batchsize = 16
ESC_hopsize = 0.1
ESC_limit = -1
