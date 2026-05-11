import numpy as np
import soundfile as sf
import os

os.makedirs('data/raw_audio', exist_ok=True)
sr = 22050

t = np.linspace(0, 5, sr * 5)
low_freq = np.sin(2 * np.pi * 100 * t)
sf.write('data/raw_audio/low_freq.wav', low_freq, sr)

high_freq = np.sin(2 * np.pi * 8000 * t)
sf.write('data/raw_audio/high_freq.wav', high_freq, sr)

noise = np.random.randn(len(t))
sf.write('data/raw_audio/noise.wav', noise, sr)
