import os
import glob
import librosa
import numpy as np
import json
from sklearn.preprocessing import StandardScaler

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)[1:]
    S, _ = librosa.magphase(librosa.stft(y))
    centroid = np.mean(librosa.feature.spectral_centroid(S=S))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    rolloff = np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo_tuple = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo = tempo_tuple[0][0] if isinstance(tempo_tuple[0], np.ndarray) else tempo_tuple[0]
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    key = np.argmax(chroma_mean)
    mode = 1.0 if chroma_mean[(key + 4) % 12] > chroma_mean[(key + 3) % 12] else 0.0
    features = np.concatenate([
        mfcc_mean,
        [centroid, flatness, rolloff, bandwidth],
        [tempo, key, mode],
        chroma_mean[:7]
    ])
    return features

def create_dataset(input_dir, output_json):
    audio_files = sorted(glob.glob(os.path.join(input_dir, '*.wav')))
    feature_dict = {}
    all_features = []
    file_ids = []
    for file_path in audio_files:
        file_id = os.path.basename(file_path).split('.')[0]
        features = extract_features(file_path)
        all_features.append(features)
        file_ids.append(file_id)
    all_features = np.array(all_features)
    if len(all_features) > 0:
        scaler = StandardScaler()
        all_features_scaled = scaler.fit_transform(all_features)
        for i, file_id in enumerate(file_ids):
            feature_dict[file_id] = all_features_scaled[i].tolist()
    dataset = {
        "cols": all_features.shape[1] if len(all_features) > 0 else 26,
        "data": feature_dict
    }
    with open(output_json, 'w') as f:
        json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    create_dataset('data/segments', 'data/dataset.json')
