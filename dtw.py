import numpy as np
import sounddevice as sd
import soundfile as sf
import os
import wave
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank

MODEL_DIR = "model/"
words = [name for name in os.listdir(MODEL_DIR) if os.path.isdir(os.path.join(MODEL_DIR, name))]

def getMFCC(filename):
  rate, signal = wavfile.read(filename)

  if len(signal.shape) == 2:
    signal = np.mean(signal, axis=1)

  features = logfbank(signal, rate, nfilt=39, nfft=2000)
  return features

def sqeuclidean(a, b):
  return np.sum((a - b) ** 2)

def manhattan(a, b):
  return np.sum(np.abs(a - b))

def minkowski(a, b, p = 2):
  return np.sum(np.abs(a - b) ** p) ** (1 / p)

def DTW(x, y, dist_func=None):
  if dist_func is None:
    dist_func = lambda a, b: np.linalg.norm(a - b)

  len_x, len_y = len(x), len(y)
  dtw_matrix = np.zeros((len_x + 1, len_y + 1))
  dtw_matrix[0, 1:] = np.inf
  dtw_matrix[1:, 0] = np.inf

  for i in range(1, len_x + 1):
    for j in range(1, len_y + 1):
      cost = dist_func(x[i-1], y[j-1])
      dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])

  return dtw_matrix[len_x, len_y]

def recognize_from_file(filename, template):
  sample_features = getMFCC(filename)
  dist = {}

  for word, template_word in template.items():
    dist[word] = DTW(sample_features, template_word, dist_func=lambda a, b: sqeuclidean(a, b))

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
      for test_file in ["Azka Cowok.wav", "Eja.wav", "Azka Cwekcwok.wav"]:
        print(f"\nComparison #{total + 1}")

        testing_filename = os.path.join(MODEL_DIR, word, test_file)
        recognized_word = recognize_from_file(testing_filename, templates)

        print(f"Testing file: {testing_filename}")
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