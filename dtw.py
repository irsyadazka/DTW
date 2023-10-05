import numpy as np
import sounddevice as sd
import soundfile as sf
import os
import wave
from scipy.io import wavfile
from python_speech_features import mfcc
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, sqeuclidean, cosine, correlation, chebyshev, cityblock, minkowski

MODEL_DIR = "model/"
words = [name for name in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, name))]

def getMFCC(filename):
  rate, signal = wavfile.read(filename)
  features = mfcc(signal, rate, numcep=39, nfft=2000)
  return features

def DTW(x, y):
  distance, _ = fastdtw(x, y, dist=sqeuclidean)
  return distance

def recognize_from_file(filename, template):
  sample_features = getMFCC(filename)
  dist = {}

  for word, template_word in template.items():
    dist[word] = DTW(sample_features, template_word)

  return min(dist, key=dist.get)

def recognize_from_mic(template, i):
  fs = 44100
  duration = 3
  print("Speak now!")
  myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
  sd.wait()
  if not os.path.exists('mics'):
        os.mkdir('mics')

  output_path = os.path.join('mics', f'output-{i}.wav')
  sf.write(output_path, myrecording, fs)
  return recognize_from_file(output_path, template)

def calculate_accuracy(words, method = "file"):
  correct = 0
  total = 0
  
  templates = {}

  for word in words:
    template_filename = os.path.join(MODEL_DIR, word, "Azka Cewek.wav")
    templates[word] = getMFCC(template_filename)

  if method == "mic":
    replay = True
    i = 1
    while replay:
      from_mic = recognize_from_mic(templates, i)
      print(f"Recognized word from mic: {from_mic}")
      replay = input("Replay? (y/n) ") == "y"
      i += 1
    return
  else:
    for word in words:
      print(f"\nComparison #{total + 1}")
      
      testing_filename = os.path.join(MODEL_DIR, word, "Azka Cowok.wav")
      recognized_word = recognize_from_file(testing_filename, templates)

      print(f"Recognized word: {recognized_word}")
      print(f"Word: {word}")

      if recognized_word == word:
        correct += 1
      total += 1

  return (correct / total) * 100

method = input("Method (file/mic): ")
if method == "mic":
  calculate_accuracy(words, method)
  exit()


print(f"List of words:")

i = 1
for word in words:
  print(f"{i}. {word}")
  i += 1

output = calculate_accuracy(words)
print(f"\nAccuracy: {output}")