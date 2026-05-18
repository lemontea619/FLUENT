import os
import numpy as np
import soundfile as sf
import pytest

# We need to import extract_features from python/extractor.py
# Assuming tests are run from root and python module is discoverable or we add it to sys.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from extractor import extract_features

def test_extract_features(tmp_path):
    """
    Test extract_features to ensure it returns a valid numpy array with shape (26,)
    and contains no NaN or Inf values.
    """
    # 1. Generate dummy audio file dynamically
    sr = 22050
    # 1 seconds of audio to be sure it is >= 2048 samples
    t = np.linspace(0, 2, sr * 2)
    # A simple sine wave
    y = np.sin(2 * np.pi * 440 * t)

    # Save the dummy audio to a temporary file
    dummy_wav_path = str(tmp_path / "dummy_audio.wav")
    sf.write(dummy_wav_path, y, sr)

    # 2. Call the function
    features = extract_features(dummy_wav_path)

    # 3. Assertions
    # Output must be a numpy.ndarray
    assert isinstance(features, np.ndarray), "Output must be a numpy ndarray"

    # Output shape must be (26,)
    assert features.shape == (26,), f"Expected shape (26,), got {features.shape}"

    # Output must not contain NaN or Inf
    assert not np.isnan(features).any(), "Output contains NaN values"
    assert not np.isinf(features).any(), "Output contains Inf values"
