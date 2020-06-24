import src.features.features_parameters as par
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import librosa
import openl3
import cv2

def frame(data, window_length, hop_length):
  num_samples = data.shape[0]
  num_frames = 1 + int(np.floor((num_samples - window_length) / hop_length))
  shape = (num_frames, window_length) + data.shape[1:]
  strides = (data.strides[0] * hop_length,) + data.strides
  return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def periodic_hann(window_length):
  return 0.5 - (0.5 * np.cos(2 * np.pi / window_length *
                             np.arange(window_length)))


def stft_magnitude(signal, fft_length,
                   hop_length=None,
                   window_length=None):
  frames = frame(signal, window_length, hop_length)     # Audio packets - 512 packets
  window = periodic_hann(window_length)
  windowed_frames = frames * window

  return np.abs(np.fft.rfft(windowed_frames, int(fft_length)))

def audio_feat(audio, sr, openl3=False):
  if(not openl3):
    hop_length = int(par.ESC_hopsize * sr)
    frame_length = sr

    x = librosa.util.utils.frame(audio, frame_length=frame_length, hop_length=hop_length).T # Audio frames
    
    specs = list()
    for frame in x:
        specs.append(get_spectrogram(frame, sr, axis = 1))

    return np.asarray(specs, np.double) # Maybe add a dimension
  else:
    emb, ts = openl3.get_audio_embedding(audio, sr, embedding_size=512)
    input(emb.shape)

def get_frame(video_path):
    ret = False
    while(ret == False):
      cap = cv2.VideoCapture(video_path)
      video = np.zeros((10,1,224,224), dtype = np.uint8)    
      for i in range(10):
        ret, frame = cap.read()
        if(ret):
            frame = cv2.resize(frame, (224,224), interpolation = cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            video[i,0,:,:] = frame
        else:
            print("ERROR: -1 in the get frame function, frame index {} file {}".format(i, video_path))
            cap.release()
            break
    cap.release()
    return video, 0


def get_spectrogram(audioSignal, sr, axis = 0):
    if sr != 48000:
      print("ERROR: sampling rate is not 48kHz")    
    to_pad = int(1 * sr - audioSignal.shape[0])
    if to_pad <= 0:
      to_remove = int(to_pad * -1)
      audioSignal = audioSignal[to_remove:]
    if to_pad > 0:
      if len(audioSignal.shape) > 1:
        padding_array = np.zeros((to_pad, audioSignal.shape[1]), dtype='int16')      
      else:
        padding_array = np.zeros((to_pad,), dtype='int16')      
      audioSignal = np.concatenate( (padding_array,audioSignal) )

    spec = stft_magnitude(audioSignal, par.AUDIO_n_dft, par.AUDIO_n_hop, par.AUDIO_n_dft)
    spec = np.log(np.clip(spec, 1e-12, None) / par.DUMMY_PARAMETER).T
    spec = spec.astype(np.float32)
    
    return np.expand_dims(spec, axis=0)
