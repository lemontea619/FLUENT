import json
import argparse
import subprocess
import os
import glob
try:
    import librosa
    import numpy as np
    from scipy.ndimage import median_filter
    from sklearn.cluster import AgglomerativeClustering # AgglomerativeClustering をインポート
    from scipy.sparse import lil_matrix # 疎行列作成のためにインポート
except ImportError:
    print("librosa または numpy がインストールされていません。")
    print("pip install librosa numpy を実行してください。")

def estimate_key_and_chords(y, sr):
    """楽曲全体の調と簡易的なコード進行を分析する"""
    # クロマグラムの計算
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    # --- 調（Key）の推定 ---
    # 各音（C, C#, ...）の平均エネルギー
    chroma_avg = np.mean(chroma, axis=1)
    keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    # 簡易的なメジャーキー判定
    key_idx = np.argmax(chroma_avg)
    estimated_key = keys[key_idx]
    
    # --- コード進行の推定（簡易版） ---
    # 楽曲を8つの区間に分けて、それぞれの区間で最も強い音をルートとする
    n_segments = 8
    hop_size = chroma.shape[1] // n_segments
    progression = []
    for i in range(n_segments):
        chunk = chroma[:, i*hop_size : (i+1)*hop_size]
        if chunk.shape[1] == 0: continue
        root_idx = np.argmax(np.mean(chunk, axis=1))
        # 簡易的にメジャー/マイナーを判定（第3音の強さで比較）
        third_major = (root_idx + 4) % 12
        third_minor = (root_idx + 3) % 12
        is_minor = np.mean(chunk[third_minor]) > np.mean(chunk[third_major])
        chord = keys[root_idx] + ("m" if is_minor else "")
        progression.append(chord)
    
    chord_str = " -> ".join(progression)
    return estimated_key, chord_str

def get_segments_from_chapters(url):
    """YouTubeのチャプター情報から1番の各セクションの時間を取得する"""
    cmd = ['yt-dlp', '--dump-json', '--flat-playlist', url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    try:
        if result.returncode != 0:
            print("メタデータの取得に失敗しました。")
            return None

        info = json.loads(result.stdout)
        chapters = info.get('chapters')

        if not chapters:
            print("この動画にはチャプターが設定されていないため、自動分割できません。")
            # チャプターがない場合のデバッグ用に出力
            print(f"取得したタイトル: {info.get('title')}")
            return None

        # 抽出したいセクションとキーワードの定義（より柔軟に）
        target_map = {
            "1_Intro": ["intro", "イントロ", "序奏"],
            "1_A-Melody": ["aメロ", "a-melody", "verse 1", "v1", "1番 a"],
            "1_B-Melody": ["bメロ", "b-melody", "pre-chorus", "1番 b"],
            "1_Chorus": ["サビ", "chorus", "hook", "1番 サビ"]
        }

        found_segments = {}

        print("見つかったチャプター一覧:")
        for chapter in chapters:
            title = chapter['title'].lower()
            print(f" - {title} ({chapter['start_time']}s)")
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
        print(f"解析中にエラーが発生しました: {e}")
        return None

def get_segments_via_librosa(audio_path):
    """Librosaを使用して音響解析を行い、4つのセクションに分割する"""
    print("音響解析を開始します（これには数十秒かかる場合があります）...")
    # 120秒（1番の目安）をロード
    y, sr = librosa.load(audio_path, duration=120)
    
    # 和音成分(Chroma)と音色成分(MFCC)を抽出
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    
    # 特徴量を結合し、時間的な変化を捉えるためにスタックする
    combined_features = np.vstack([chroma, mfcc])
    # ブログの「構造の自動検出」にならい、スタックと時間遅延で文脈を持たせる
    stacked_features = librosa.feature.stack_memory(combined_features, n_steps=10, delay=3)

    # --- ブログ記事の「楽曲構造の自動検出」をより忠実に再現 ---
    # 1. 自己回帰行列（Recurrence Matrix）を計算
    # ブログでは `rec` と呼ばれているもの。類似度を計算。
    rec = librosa.segment.recurrence_matrix(stacked_features, k=5, width=3, sym=True)

    # 2. 接続性行列 (Connectivity Matrix) を作成
    # sklearn.cluster.AgglomerativeClustering の connectivity 引数に渡すための疎行列
    # 時間的に隣接するフレームのみを接続すると定義します。
    # これにより、クラスタリングが時間的な連続性を尊重するようになります。
    n_frames = stacked_features.shape[1]
    connectivity = lil_matrix((n_frames, n_frames), dtype=int)
    for i in range(n_frames):
        if i > 0:
            connectivity[i, i-1] = 1
            connectivity[i-1, i] = 1 # 双方向
    connectivity = connectivity.tocsr() # CSR形式に変換

    # 3. Agglomerative Clustering を実行
    # k=6 はブログの例に合わせ、イントロ、A、B、サビ、間奏などを想定
    # stacked_features.T は (n_samples, n_features) の形式にするため
    model = AgglomerativeClustering(n_clusters=6, connectivity=connectivity, linkage='ward')
    cluster_labels = model.fit_predict(stacked_features.T)
    # --- ここまでブログ記事の「楽曲構造の自動検出」をより忠実に再現 ---

    # 4. メディアンフィルタで細かな変化を強力に除去
    kernel_size = max(1, int(sr // 512 * 7)) # 7秒程度の窓で平滑化し、細かい分割を抑制する
    if kernel_size % 2 == 0: kernel_size += 1
    cluster_labels = median_filter(cluster_labels, size=kernel_size)

    # 5. ラベルが変化するフレームを探す (これが主要な境界候補)
    changes = np.where(np.diff(cluster_labels) != 0)[0]
    
    # 6. 音の急激な変化（ノベルティ）も考慮するが、感度を下げて大きな変化のみ拾う
    novelty = librosa.onset.onset_strength(y=y, sr=sr)
    peaks = librosa.util.peak_pick(novelty, pre_max=100, post_max=100, pre_avg=100, post_avg=100, delta=0.5, wait=200)
    
    all_candidates = np.unique(np.concatenate([changes, peaks]))

    num_frames = len(cluster_labels)
    targets = [
        librosa.time_to_frames(25, sr=sr),  # イントロ終了目安を少し後ろに
        librosa.time_to_frames(55, sr=sr),  # Aメロ終了目安
        librosa.time_to_frames(85, sr=sr)   # Bメロ終了目安
    ]

    selected_bounds = [0]
    # 各セグメントが最低 10秒 は確保されるように制限（細切れ防止）
    min_duration_frames = librosa.time_to_frames(10.0, sr=sr) 

    for i, target in enumerate(targets):
        remaining_segments = 3 - i
        max_allowed_frame = num_frames - (remaining_segments * min_duration_frames)
        min_allowed_frame = selected_bounds[-1] + min_duration_frames

        valid_candidates = all_candidates[(all_candidates >= min_allowed_frame) & (all_candidates <= max_allowed_frame)]
        
        if len(valid_candidates) > 0:
            # ターゲットに最も近い候補を選択
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
        # 次の境界がある場合はそこまで、ない場合は解析範囲の終わりまで (typo修正)
        if i + 1 < len(bound_times):
            end_t = bound_times[i+1]
        else:
            end_t = total_duration
        print(f"DEBUG: {section_names[i]} -> {start_t:.2f}s to {end_t:.2f}s")
        found_segments[section_names[i]] = {
            "title": f"Analyzed {section_names[i]}",
            "start": round(float(start_t), 2),
            "end": round(float(end_t), 2)
        }
    
    return found_segments

def download_specific_sections(url, segments, key_info, chord_info):
    """指定されたセクションを個別にダウンロードする"""
    # スクリプトの場所を基準に絶対パスを作成。どこから実行しても fluent/node/public/audio を指すようにします
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # server.js のデータディレクトリ構造に合わせます
    data_dir = os.path.join(script_dir, "data")
    output_dir = os.path.join(data_dir, "segments")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"\n[INFO] 保存先フォルダ: {output_dir}")

    # 既存の最大IDを確認して続きの番号を振る
    existing_files = glob.glob(os.path.join(output_dir, "example-*.wav"))
    max_id = 0
    for f in existing_files:
        basename = os.path.basename(f)
        try:
            num = int(basename.split('-')[1].split('.')[0].split('_')[0])
            if num > max_id:
                max_id = num
        except (ValueError, IndexError):
            pass

    next_start_id = max_id + 1

    # 一時的にフル音源をダウンロード（1番の部分だけなど、少し長めに取得）
    temp_source = os.path.join(output_dir, "temp_full_audio_for_splitting.mp3")
    if os.path.exists(temp_source):
        os.remove(temp_source)

    print("\n[STEP 1] YouTubeから音源をダウンロード中...")
    subprocess.run(['yt-dlp', '-x', '--audio-format', 'mp3', '-o', temp_source, url], check=True)

    print("\n[STEP 2] 音声を分析・セグメントに分割中...")

    if not os.path.exists(temp_source):
        print("エラー: 音源のダウンロードに失敗しました。")
        return

    downloaded_info = []
    for i, (label, info) in enumerate(segments.items()):
        current_id = next_start_id + i
        segment_id = f"example-{current_id:03d}"
        file_name = f"{segment_id}.wav"
        output_path = os.path.join(output_dir, file_name)
        print(f" -> 分割作成中: {file_name} ({info['start']}s 〜 {info['end']}s)")
        
        duration = info['end'] - info['start']
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-ss', str(info['start']), # 入力の前に置くことで高速に該当箇所へ移動
            '-t', str(duration),
            '-i', temp_source,
            '-vn', '-af', 'loudnorm=I=-21:TP=-9.0:LRA=7,volume=-3dB',
            '-acodec', 'pcm_s16le', '-ar', '44100', '-ac', '2',
            output_path
        ]
        # 実行時のエラーをキャッチできるようにします
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        downloaded_info.append({
            "id": segment_id,
            "title": f"{label}: {info['title']}",
            "file": file_name,
            "key": key_info,
            "chords": chord_info
        })
    
    # 一時ファイルを削除
    if os.path.exists(temp_source):
        os.remove(temp_source)

    json_path = os.path.join(output_dir, "segments.json")
    print(f"[STEP 3] 管理リストを更新中: {json_path}")
    all_segments = []
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                all_segments = json.load(f)
        except json.JSONDecodeError:
            pass

    all_segments.extend(downloaded_info)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_segments, f, ensure_ascii=False, indent=2)

    # [STEP 4] dataset.json を更新
    dataset_path = os.path.join(data_dir, "dataset.json")
    dataset = {}
    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
        except: pass
    
    if "data" not in dataset: dataset["data"] = {}
    
    for item in downloaded_info:
        seg_id = item["id"]
        if seg_id not in dataset["data"]:
            dataset["data"][seg_id] = {}
        # 調とコード進行を記録
        dataset["data"][seg_id]["global_key"] = key_info
        dataset["data"][seg_id]["global_chords"] = chord_info

    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"[STEP 4] dataset.json に調とコードを記録しました。")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", help="YouTube URL")
    args = parser.parse_args()

    print("=== Fluent YouTube Audio Splitter ===")
    url = args.url if args.url else input("YouTubeのURLを入力してください: ")
    
    if not url.startswith(("http://", "https://")):
        print("エラー: 有効なURLを入力してください。")
        print(f"現在の入力: {url[:50]}...")
        return

    print("チャプターを解析中...")
    segments = get_segments_from_chapters(url)
    
    if not segments:
        print("チャプターが見つかりませんでした。音響解析に切り替えます。")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # 一時的にフル音源をダウンロード
        data_dir = os.path.join(script_dir, "data")
        temp_full_audio = os.path.join(data_dir, "temp_analysis.mp3")
        os.makedirs(data_dir, exist_ok=True)
        subprocess.run(['yt-dlp', '-x', '--audio-format', 'mp3', '-o', temp_full_audio, url], check=True)

        if not os.path.exists(temp_full_audio):
            print("エラー: メイン音声ファイルのダウンロードに失敗しました。URLを確認してください。")
            return
        
        # 調とコード進行の分析（分割前に行う）
        y_full, sr_full = librosa.load(temp_full_audio, duration=120)
        key_info, chord_info = estimate_key_and_chords(y_full, sr_full)
        print(f"分析結果 - 調: {key_info}, コード進行: {chord_info}")

        segments = get_segments_via_librosa(temp_full_audio)
        if os.path.exists(temp_full_audio):
            os.remove(temp_full_audio)
    else:
        # チャプターがある場合も同様に分析が必要な場合はここに追加
        key_info, chord_info = "Unknown", "Unknown"

    if segments:
        print(f"{len(segments)} 個のセクションを特定しました。")
        download_specific_sections(url, segments, key_info, chord_info)
        print("\n完了しました。")
    else:
        print("セグメントを特定できませんでした。")


if __name__ == "__main__":
    main()
