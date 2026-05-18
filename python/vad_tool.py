import argparse
import sys
import os
import json
import librosa
import numpy as np

def perform_vad(input_file):
    try:
        # Load audio using librosa
        y, sr = librosa.load(input_file, sr=16000, mono=True)

        # Use librosa's effects.split to find non-silent intervals
        # top_db specifies the threshold (in dB) below reference to consider as silence
        intervals = librosa.effects.split(y, top_db=30)

        regions = []
        for interval in intervals:
            start_time = float(interval[0]) / sr
            end_time = float(interval[1]) / sr
            # Only keep segments longer than 0.5 seconds
            if end_time - start_time >= 0.5:
                regions.append({
                    "start": round(start_time, 3),
                    "end": round(end_time, 3)
                })

        # Merge segments that are very close to each other (e.g., < 0.5s apart)
        merged_regions = []
        for r in regions:
            if not merged_regions:
                merged_regions.append(r)
            else:
                last_r = merged_regions[-1]
                if r["start"] - last_r["end"] < 0.5:
                    last_r["end"] = r["end"]
                else:
                    merged_regions.append(r)

        print(json.dumps({
            "success": True,
            "regions": merged_regions
        }))
        return 0
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input audio file")

    args = parser.parse_args()
    sys.exit(perform_vad(args.input))
