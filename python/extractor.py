import argparse
import glob
import json
import os
import sys

import joblib
import librosa
import numpy as np
import soundfile as sf
from sklearn.preprocessing import StandardScaler


def extract_features(file_path: str) -> np.ndarray:
    y, sr = librosa.load(file_path, sr=None)
    n_fft = min(2048, len(y))
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft)
    mfcc_mean = np.mean(mfcc, axis=1)[1:]
    S, _ = librosa.magphase(librosa.stft(y, n_fft=n_fft))
    centroid = np.mean(librosa.feature.spectral_centroid(S=S))
    flatness = np.mean(librosa.feature.spectral_flatness(y=y, n_fft=n_fft))
    rolloff = np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, n_fft=n_fft)
    tempo_tuple = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo = (
        tempo_tuple[0][0] if isinstance(tempo_tuple[0], np.ndarray) else tempo_tuple[0]
    )
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    key = np.argmax(chroma_mean)
    mode = 1.0 if chroma_mean[(key + 4) % 12] > chroma_mean[(key + 3) % 12] else 0.0
    features = np.concatenate(
        [
            mfcc_mean,
            np.array([centroid, flatness, rolloff, bandwidth]),
            np.array([tempo, key, mode]),
            chroma_mean[:7],
        ]
    )
    return features


MIN_DURATION_SECONDS = 2.0


def create_dataset(input_dir: str, output_json: str) -> None:
    audio_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
    total = len(audio_files)
    feature_dict = {}
    all_features = []
    file_ids = []
    for i, file_path in enumerate(audio_files, 1):
        file_id = os.path.basename(file_path).split(".")[0]
        info = sf.info(file_path)
        duration = info.frames / info.samplerate if info.samplerate else 0.0
        if duration < MIN_DURATION_SECONDS:
            print(
                f"[{i}/{total}] Skip {file_id}: too short ({duration:.2f}s)",
                file=sys.stderr,
                flush=True,
            )
            continue
        print(f"[{i}/{total}] Extracting: {file_id}", file=sys.stderr, flush=True)
        features = extract_features(file_path)
        all_features.append(features)
        file_ids.append(file_id)
    all_features = np.array(all_features)
    if len(all_features) > 0:
        scaler = StandardScaler()
        all_features_scaled = scaler.fit_transform(all_features)
        scaler_path = os.path.join(os.path.dirname(output_json), "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}", file=sys.stderr)
        for i, file_id in enumerate(file_ids):
            feature_dict[file_id] = all_features_scaled[i].tolist()
    dataset = {
        "cols": all_features.shape[1] if len(all_features) > 0 else 26,
        "data": feature_dict,
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Dataset created with {len(file_ids)} segments", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features from audio segments")
    parser.add_argument(
        "--input",
        type=str,
        default="data/segments",
        help="Input directory with segment WAV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/dataset.json",
        help="Output JSON file for dataset",
    )

    args = parser.parse_args()
    create_dataset(args.input, args.output)
