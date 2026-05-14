import argparse
import sys
import os
import json
import glob
from pydub import AudioSegment

def process_regions(input_file, title, regions, out_dir):
    try:
        os.makedirs(out_dir, exist_ok=True)

        safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c in (' ', '-', '_')]).rstrip()
        safe_title = safe_title.replace(" ", "_")
        if not safe_title:
            safe_title = "Audio"

        existing_files = glob.glob(os.path.join(out_dir, f"{safe_title}_*.wav"))
        max_idx = 0
        for f in existing_files:
            base = os.path.basename(f)
            try:
                idx_str = base.replace(f"{safe_title}_", "").replace(".wav", "")
                idx = int(idx_str)
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                pass

        next_idx = max_idx + 1

        # Load the whole audio file
        audio = AudioSegment.from_file(input_file)

        results = []
        for region in regions:
            start_ms = int(region.get("start", 0) * 1000)
            end_ms = int(region.get("end", 0) * 1000)

            if end_ms <= start_ms:
                continue

            cropped = audio[start_ms:end_ms]

            # Peak normalize to -12 dBFS
            # pydub's max_dBFS gives the peak value.
            # We want max_dBFS to be -12.0
            peak_dbfs = cropped.max_dBFS
            change_db = -12.0 - peak_dbfs
            normalized = cropped.apply_gain(change_db)

            out_filename = f"{safe_title}_{next_idx:03d}.wav"
            out_filepath = os.path.join(out_dir, out_filename)
            out_filepath = os.path.abspath(out_filepath)

            # Save it
            normalized.export(out_filepath, format="wav")
            results.append({
                "id": out_filename.replace(".wav", ""),
                "file": out_filename
            })
            next_idx += 1

        print(json.dumps({"success": True, "saved_files": results}))
        return 0
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input temporary audio file")
    parser.add_argument("--title", required=True, help="Video title")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--regions", required=True, help="JSON string of regions [{'start': 1.0, 'end': 5.0}]")

    args = parser.parse_args()

    try:
        regions = json.loads(args.regions)
    except json.JSONDecodeError:
        print(json.dumps({"success": False, "error": "Invalid regions JSON"}))
        sys.exit(1)

    sys.exit(process_regions(args.input, args.title, regions, args.out_dir))
