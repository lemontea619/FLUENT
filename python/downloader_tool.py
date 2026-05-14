import argparse
import sys
import os
import json
import yt_dlp

def download_audio(url, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # We want to download audio and get the title to use it later
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(out_dir, '%(title)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'Unknown_Video')

            # The actual file will be converted to .wav by postprocessor
            # We can construct the path.
            # yt-dlp sanitizes the title sometimes, so we ask ydl for the final filename
            expected_filename = ydl.prepare_filename(info)
            # Replace extension with .wav
            base, _ = os.path.splitext(expected_filename)
            wav_path = base + '.wav'

            # Normalize path
            wav_path = os.path.abspath(wav_path)

            result = {
                "success": True,
                "title": title,
                "file_path": wav_path
            }
            print(json.dumps(result))
            return 0
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="YouTube URL to download")
    parser.add_argument("--out-dir", required=True, help="Temporary output directory")
    args = parser.parse_args()

    sys.exit(download_audio(args.url, args.out_dir))
