import numpy as np
import sounddevice as sd
import soundfile as sf
import os
import wave
from scipy.io import wavfile
from python_speech_features import mfcc
from fastdtw import fastdtw

MODEL_DIR = "model/"
words = [name for name in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, name))]

def getMFCC(filename):
  rate, signal = wavfile.read(filename)
  features = mfcc(signal, rate, numcep=39)
  return features

def dtw(x, y):
  distance, _ = fastdtw(x, y)
  return distance

def recognize_from_file(filename, template):
  sample_features = getMFCC(filename)
  dist = {}

  for word, template_word in template.items():
    dist[word] = dtw(sample_features, template_word)

  return min(dist, key=dist.get)

def recognize_from_mic(template):
  fs = 44100
  duration = 3
  myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
  sd.wait()
  sf.write('output.wav', myrecording, fs)
  return recognize_from_file('output.wav', template)

def calculate_accuracy():
  correct, total = 0

  templates = {}
  
  for word in words:
    template_filename = os.path.join(MODEL_DIR, word, "template.wav")
    templates[word] = getMFCC(template_filename)

  for word in words:
    for i in range(1, 31):
      testing_filename = os.path.join(MODEL_DIR, word, str(i) + ".wav")
      recognized_word = recognize_from_file(testing_filename, templates)

      if recognized_word == word:
        correct += 1
      total += 1

  return (correct / total) * 100