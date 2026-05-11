import os
import glob
import librosa
import soundfile as sf
import numpy as np

def segment_audio(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    audio_files = glob.glob(os.path.join(input_dir, '*.wav'))
    segment_id = 1

    for file_path in audio_files:
        y, sr = librosa.load(file_path, sr=None)
        hop_length = 512
        if len(y) < sr * 1:
             out_path = os.path.join(output_dir, f"example-{segment_id:03d}.wav")
             sf.write(out_path, y, sr)
             segment_id += 1
             continue
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length)
        k = max(2, len(y) // (sr * 2))
        try:
            boundaries = librosa.segment.agglomerative(mfcc, k)
            boundary_samples = librosa.frames_to_samples(boundaries, hop_length=hop_length)
            boundary_samples = np.unique(np.concatenate([[0], boundary_samples, [len(y)]]))
            for i in range(len(boundary_samples) - 1):
                start = boundary_samples[i]
                end = boundary_samples[i+1]
                if end - start > 0:
                    segment_y = y[start:end]
                    out_path = os.path.join(output_dir, f"example-{segment_id:03d}.wav")
                    sf.write(out_path, segment_y, sr)
                    segment_id += 1
        except Exception:
            out_path = os.path.join(output_dir, f"example-{segment_id:03d}.wav")
            sf.write(out_path, y, sr)
            segment_id += 1

if __name__ == "__main__":
    segment_audio('data/raw_audio', 'data/segments')
