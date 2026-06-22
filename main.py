import json
import argparse
import subprocess
import os
import sys
import glob

# --- FastAPI 追加部分 ---
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/analyze")
def analyze(url: str):
    """Web API から YouTube URL を受け取って処理する"""
    try:
        segments, key_info, chord_info = run_analysis(url)
        return {
            "segments": segments,
            "key": key_info,
            "chords": chord_info
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- ここまで FastAPI 追加部分 ---


try:
    import librosa
    import numpy as np
    from scipy.ndimage import median_filter
    from sklearn.cluster import AgglomerativeClustering
    from scipy.sparse import lil_matrix
except ImportError:
    print("librosa または numpy がインストールされていません。")
    print("pip install librosa numpy を実行してください。")


# ====== 既存の関数群（変更なし） ======

def estimate_key_and_chords(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key_idx = np.argmax(chroma_avg)
    estimated_key = keys[key_idx]

    n_segments = 8
    hop_size = chroma.shape[1] // n_segments
    progression = []
    for i in range(n_segments):
        chunk = chroma[:, i*hop_size : (i+1)*hop_size]
        if chunk.shape[1] == 0: continue
        root_idx = np.argmax(np.mean(chunk, axis=1))
        third_major = (root_idx + 4) % 12
        third_minor = (root_idx + 3) % 12
        is_minor = np.mean(chunk[third_minor]) > np.mean(chunk[third_major])
        chord = keys[root_idx] + ("m" if is_minor else "")
        progression.append(chord)

    chord_str = " -> ".join(progression)
    return estimated_key, chord_str


def get_segments_from_chapters(url):
    cmd = ['yt-dlp', '--dump-json', '--flat-playlist', url]
    result = subprocess.run(cmd, capture_output=True, text=True)

    try:
        if result.returncode != 0:
            print("メタデータの取得に失敗しました。")
            return None

        info = json.loads(result.stdout)
        chapters = info.get('chapters')

        if not chapters:
            print("この動画にはチャプターが設定されていません。")
            return None

        target_map = {
            "1_Intro": ["intro", "イントロ", "序奏"],
            "1_A-Melody": ["aメロ", "a-melody", "verse 1", "v1", "1番 a"],
            "1_B-Melody": ["bメロ", "b-melody", "pre-chorus", "1番 b"],
            "1_Chorus": ["サビ", "chorus", "hook", "1番 サビ"]
        }

        found_segments = {}

        for chapter in chapters:
            title = chapter['title'].lower()
            for label, keywords in target_map.items():
                if label not in found_segments:
                    if any(kw in title for kw in keywords):
                        found_segments[label] = {
                            "title": chapter['title'],
                            "start": chapter['start_time'],
                            "end": chapter['end_time']
                        }

        return found_segments
    except Exception as e:
        print(f"解析中にエラー: {e}")
        return None


def get_segments_via_librosa(audio_path):
    print("音響解析を開始します...")
    y, sr = librosa.load(audio_path, duration=120)

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    combined_features = np.vstack([chroma, mfcc])
    stacked_features = librosa.feature.stack_memory(combined_features, n_steps=10, delay=3)

    rec = librosa.segment.recurrence_matrix(stacked_features, k=5, width=3, sym=True)

    n_frames = stacked_features.shape[1]
    connectivity = lil_matrix((n_frames, n_frames), dtype=int)
    for i in range(n_frames):
        if i > 0:
            connectivity[i, i-1] = 1
            connectivity[i-1, i] = 1
    connectivity = connectivity.tocsr()

    model = AgglomerativeClustering(n_clusters=6, connectivity=connectivity, linkage='ward')
    cluster_labels = model.fit_predict(stacked_features.T)

    kernel_size = max(1, int(sr // 512 * 7))
    if kernel_size % 2 == 0: kernel_size += 1
    cluster_labels = median_filter(cluster_labels, size=kernel_size)

    changes = np.where(np.diff(cluster_labels) != 0)[0]
    novelty = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = librosa.util.peak_pick(novelty, pre_max=100, post_max=100, pre_avg=100, post_avg=100, delta=0.5, wait=200)

    all_candidates = np.unique(np.concatenate([changes, peaks]))

    num_frames = len(cluster_labels)
    targets = [
        librosa.time_to_frames(25, sr=sr),
        librosa.time_to_frames(55, sr=sr),
        librosa.time_to_frames(85, sr=sr)
    ]

    selected_bounds = [0]
    min_duration_frames = librosa.time_to_frames(10.0, sr=sr)

    for i, target in enumerate(targets):
        remaining_segments = 3 - i
        max_allowed_frame = num_frames - (remaining_segments * min_duration_frames)
        min_allowed_frame = selected_bounds[-1] + min_duration_frames

        valid_candidates = all_candidates[(all_candidates >= min_allowed_frame) & (all_candidates <= max_allowed_frame)]

        if len(valid_candidates) > 0:
            closest_candidate_frame = valid_candidates[np.argmin(np.abs(valid_candidates - target))]
            selected_bounds.append(closest_candidate_frame)
        else:
            selected_bounds.append(max(min(target, max_allowed_frame), min_allowed_frame))

    bound_times = librosa.frames_to_time(selected_bounds, sr=sr)

    section_names = ["1_Intro", "1_A-Melody", "1_B-Melody", "1_Chorus"]
    found_segments = {}

    total_duration = librosa.get_duration(y=y, sr=sr)
    for i in range(len(section_names)):
        start_t = bound_times[i]
        end_t = bound_times[i+1] if i + 1 < len(bound_times) else total_duration
        found_segments[section_names[i]] = {
            "title": f"Analyzed {section_names[i]}",
            "start": round(float(start_t), 2),
            "end": round(float(end_t), 2)
        }

    return found_segments


def download_specific_sections(url, segments, key_info, chord_info):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    output_dir = os.path.join(data_dir, "segments")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    existing_files = glob.glob(os.path.join(output_dir, "example-*.wav"))
    max_id = 0
    for f in existing_files:
        basename = os.path.basename(f)
        try:
            num = int(basename.split('-')[1].split('.')[0].split('_')[0])
            if num > max_id:
                max_id = num
        except:
            pass

    next_start_id = max_id + 1

    temp_source = os.path.join(output_dir, "temp_full_audio_for_splitting.mp3")
    if os.path.exists(temp_source):
        os.remove(temp_source)

    subprocess.run(['yt-dlp', '-x', '--audio-format', 'mp3', '-o', temp_source, url], check=True)

    downloaded_info = []
    for i, (label, info) in enumerate(segments.items()):
        current_id = next_start_id + i
        segment_id = f"example-{current_id:03d}"
        file_name = f"{segment_id}.wav"
        output_path = os.path.join(output_dir, file_name)

        duration = info['end'] - info['start']
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-ss', str(info['start']),
            '-t', str(duration),
            '-i', temp_source,
            '-vn', '-af', 'loudnorm=I=-21:TP=-9.0:LRA=7,volume=-3dB',
            '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
            output_path
        ]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        downloaded_info.append({
            "id": segment_id,
            "title": f"{label}: {info['title']}",
            "file": file_name,
            "key": key_info,
            "chords": chord_info
        })

    if os.path.exists(temp_source):
        os.remove(temp_source)

    json_path = os.path.join(output_dir, "segments.json")
    all_segments = []
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                all_segments = json.load(f)
        except:
            pass

    all_segments.extend(downloaded_info)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, ensure_ascii=False, indent=2)

    dataset_path = os.path.join(data_dir, "dataset.json")
    dataset = {}
    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                loaded_data = json.load(f)
                if isinstance(loaded_data, dict):
                    dataset = loaded_data
        except:
            pass

    if not isinstance(dataset, dict) or "data" not in dataset:
        dataset = {"data": {}}

    for item in downloaded_info:
        seg_id = item["id"]
        if seg_id not in dataset["data"]:
            dataset["data"][seg_id] = {}
        dataset["data"][seg_id]["global_key"] = key_info
        dataset["data"][seg_id]["global_chords"] = chord_info

    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)


# ====== ここから追加：CLI と API 共通の処理 ======

def run_analysis(url: str):
    print("チャプターを解析中...")
    segments = get_segments_from_chapters(url)

    if not segments:
        print("チャプターなし → 音響解析へ切り替え")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "data")
        temp_full_audio = os.path.join(data_dir, "temp_analysis.mp3")
        os.makedirs(data_dir, exist_ok=True)

        subprocess.run(['yt-dlp', '-x', '--audio-format', 'mp3', '-o', temp_full_audio, url], check=True)

        y_full, sr_full = librosa.load(temp_full_audio, duration=120)
        key_info, chord_info = estimate_key_and_chords(y_full, sr_full)

        segments = get_segments_via_librosa(temp_full_audio)
        if os.path.exists(temp_full_audio):
            os.remove(temp_full_audio)
    else:
        key_info, chord_info = "Unknown", "Unknown"

    return segments, key_info, chord_info


# ====== CLI 用 main()（元のまま） ======

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="YouTube URL")
    args = parser.parse_args()

    print("=== Fluent YouTube Audio Splitter ===")
    url = args.url if args.url else input("YouTubeのURLを入力してください: ")

    segments, key_info, chord_info = run_analysis(url)

    if segments:
        print(f"{len(segments)} 個のセクションを特定しました。")
        download_specific_sections(url, segments, key_info, chord_info)
        print("\n完了しました。")
    else:
        raise ValueError("セグメントを特定できませんでした。")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
