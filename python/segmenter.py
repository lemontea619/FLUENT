import argparse
import glob
import os
import sys

import librosa
import numpy as np
import soundfile as sf


def merge_segments(segments, target_count, min_duration, sr):
    durations = [(end - start) / sr for start, end in segments]

    def merge_pair(i, j):
        a = min(i, j)
        b = max(i, j)
        segments[a] = (segments[a][0], segments[b][1])
        del segments[b]
        durations[a] = (segments[a][1] - segments[a][0]) / sr
        del durations[b]

    while len(segments) > target_count or any(d < min_duration for d in durations):
        if len(segments) <= target_count and all(d >= min_duration for d in durations):
            break

        shortest_idx = min(range(len(segments)), key=lambda i: durations[i])
        if shortest_idx == 0:
            merge_with = 1
        elif shortest_idx == len(segments) - 1:
            merge_with = shortest_idx - 1
        else:
            left = durations[shortest_idx - 1]
            right = durations[shortest_idx + 1]
            merge_with = shortest_idx - 1 if left >= right else shortest_idx + 1

        merge_pair(shortest_idx, merge_with)

    while len(segments) < target_count:
        largest_idx = max(range(len(segments)), key=lambda i: durations[i])
        start, end = segments[largest_idx]
        midpoint = (start + end) // 2
        if midpoint == start or midpoint == end:
            break
        segments[largest_idx:largest_idx + 1] = [(start, midpoint), (midpoint, end)]
        durations[largest_idx:largest_idx + 1] = [
            (midpoint - start) / sr,
            (end - midpoint) / sr,
        ]

    return segments


def segment_audio(
    input_dir: str,
    output_dir: str,
    num_segments: int = 8,
    min_segment_duration: float = 2.0,
) -> None:
    """
    Segment audio files into structural sections and remove very short segments.

    Args:
        input_dir: Directory containing input WAV files
        output_dir: Directory to save segmented WAV files
        num_segments: Number of final segments per file
        min_segment_duration: Minimum allowed segment duration in seconds
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_files = glob.glob(os.path.join(input_dir, "*.wav"))
    total = len(audio_files)
    if total == 0:
        print("No WAV files found.", file=sys.stderr)
        return
    segment_id = 1

    for i, file_path in enumerate(audio_files, 1):
        print(
            f"[{i}/{total}] Segmenting: {os.path.basename(file_path)}",
            file=sys.stderr,
            flush=True,
        )
        y, sr = librosa.load(file_path, sr=None)
        hop_length = 512
        duration_sec = len(y) / sr
        if duration_sec < min_segment_duration:
            print(
                f"Skip {os.path.basename(file_path)}: shorter than {min_segment_duration}s",
                file=sys.stderr,
            )
            continue

        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length)
        initial_k = min(
            max(num_segments * 4, int(np.ceil(duration_sec / min_segment_duration)) + num_segments),
            120,
        )
        try:
            boundaries = librosa.segment.agglomerative(mfcc, initial_k)
            boundary_samples = np.unique(
                np.concatenate(
                    [[0], librosa.frames_to_samples(boundaries, hop_length=hop_length), [len(y)]]
                )
            )
            segments = [
                (boundary_samples[j], boundary_samples[j + 1])
                for j in range(len(boundary_samples) - 1)
                if boundary_samples[j + 1] - boundary_samples[j] > 0
            ]
            segments = merge_segments(segments, num_segments, min_segment_duration, sr)

            for start, end in segments:
                if end - start <= 0:
                    continue
                segment_y = y[start:end]
                out_path = os.path.join(output_dir, f"example-{segment_id:03d}.wav")
                sf.write(out_path, segment_y, sr)
                segment_id += 1
        except Exception as e:
            print(f"Error segmenting {file_path}: {e}", file=sys.stderr)
            out_path = os.path.join(output_dir, f"example-{segment_id:03d}.wav")
            sf.write(out_path, y, sr)
            segment_id += 1

    print(f"Done: {segment_id - 1} segments written to {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment audio files into structural sections")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw_audio",
        help="Input directory with WAV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/segments",
        help="Output directory for segments",
    )
    parser.add_argument(
        "--num-segments",
        type=int,
        default=8,
        help="Number of segments per song (default: 8)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=2.0,
        help="Minimum duration for each final segment in seconds",
    )

    args = parser.parse_args()
    segment_audio(args.input, args.output, args.num_segments, args.min_duration)
