"""
=============================================================================
Module B: Audio Feature Extractor
=============================================================================

segmenter.py で分割されたセグメント WAV ファイルから、
音響物理特徴を抽出し、A-MAP 形式の JSON データセットに変換する

【処理フロー】
1. data/segments/ の各セグメント WAV を読み込む
2. 各ファイルから 26 次元の音響特徴量を計算
3. 全セグメントを一括で StandardScaler で正規化（平均0、標準偏差1）
4. スケーラーパラメータを scaler.pkl に保存
5. スケール済み特徴量を A-MAP 形式 JSON（dataset.json）に出力

【26次元の構成】
  MFCC（1-13）         : 12次元   メル周波数ケプストラム係数（第0係数除外）
  Spectral（4次元）    : 4次元    Centroid, Flatness, Rolloff, Bandwidth
  Theoretical（3次元） : 3次元    Tempo (BPM), Key (0-11), Mode (Major/Minor)
  Chroma（7次元）      : 7次元    色彩度 (Chroma)
  ────────────────────────────
  合計                 : 26次元

【出力例】dataset.json
{
  "cols": 26,
  "data": {
    "example-001": [0.12, 0.45, -0.33, ..., 0.88],  ← スケール済み（平均0, 標準偏差1）
    "example-002": [0.33, -0.12, 0.55, ..., 0.02],
    ...
  }
}

【重要】
  - 全セグメントを一括で StandardScaler でフィット＆変換
    → 各セグメントは相互の統計量を基準に正規化される
    → 後から単体セグメントを推論する場合は scaler.pkl を必ずロード
  - MFCC の第0係数を除外（全体エネルギーで、感性特性に寄与しないため）
  - Tempo（BPM）は自動検出。ビート検出が失敗した場合は 0 になる可能性
  - Key は 0～11（C～B の 12 半音）。Mode は Major/Minor の二値化
  - 短すぎるセグメント（< 2048サンプル）はスキップ

【使用例】
  python extractor.py
  → data/segments → data/dataset.json（スケール済み）+ data/scaler.pkl

  python extractor.py --input my_segments --output my_dataset.json
  → カスタムパス指定
"""

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
    """
    【特徴量抽出関数】1つのセグメント WAV から 26 次元ベクトルを計算

    Args:
        file_path: セグメント WAV ファイルのパス

    Returns:
        26 次元の numpy 配列（スケール前の生値）
    """

    # ============================================================
    # Step 1: WAV ファイル読み込み
    # ============================================================
    # y: 波形データ（1次元配列）
    # sr: サンプリングレート（Hz）
    y, sr = librosa.load(file_path, sr=None)

    # ============================================================
    # Step 2: FFT ウィンドウサイズの決定
    # ============================================================
    # n_fft: 高速フーリエ変換（FFT）のウィンドウサイズ
    # min(2048, len(y)): セグメント長が 2048 より短い場合は、セグメント長全体を使用
    # （短すぎる FFT は周波数分解能が落ちるため）
    #
    # 例）セグメント長 100,000 サンプル
    #   → n_fft = min(2048, 100,000) = 2048
    #
    # 例）短いセグメント 1,000 サンプル（≈23ms at 44.1kHz）
    #   → n_fft = min(2048, 1,000) = 1,000
    n_fft = min(2048, len(y))

    # ============================================================
    # Step 3-a: MFCC（メル周波数ケプストラム係数）抽出
    # ============================================================
    # librosa.feature.mfcc は (n_mfcc, T) の 2 次元配列を返す
    #   - n_mfcc: MFCC 係数次数（13）
    #   - T: 時系列フレーム数
    #
    # 出力例）44.1kHz × 1秒のセグメント
    #   MFCC shape = (13, 86)  ← 86 フレーム
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft)

    # ============================================================
    # Step 3-b: MFCC を時系列で平均化（T 次元を 1 次元に）
    # ============================================================
    # np.mean(mfcc, axis=1) → (13,) に縮約（各係数の平均）
    # [1:] → 第0係数を除外（全体的なエネルギーレベルで、感性特性に寄与しない）
    #
    # 結果: mfcc_mean は 12 次元
    mfcc_mean = np.mean(mfcc, axis=1)[1:]

    # ============================================================
    # Step 4-a: Spectral Centroid（スペクトラルセントロイド）
    # ============================================================
    # STFT（短時間フーリエ変換）を計算
    # S: 振幅スペクトラム
    S, _ = librosa.magphase(librosa.stft(y, n_fft=n_fft))

    # Spectral Centroid: 周波数スペクトラムの「重心」（Hz）
    # 値が低い → 低周波が支配的（暗い音）
    # 値が高い → 高周波が支配的（明るい音）
    #
    # librosa が返す値は時系列（フレームごと）なので平均
    centroid = np.mean(librosa.feature.spectral_centroid(S=S))

    # ============================================================
    # Step 4-b: Spectral Flatness（スペクトラルフラットネス）
    # ============================================================
    # スペクトラムが「平ら」か「ピーク状」かを示す指標（0.0～1.0）
    # 1.0 に近い → ホワイトノイズのようなフラット（所有周波数成分）
    # 0.0 に近い → ピーク状（特定周波数が強調）
    flatness = np.mean(librosa.feature.spectral_flatness(y=y, n_fft=n_fft))

    # ============================================================
    # Step 4-c: Spectral Rolloff（スペクトラルロールオフ）
    # ============================================================
    # スペクトラルエネルギーの下から 85% が含まれる周波数（Hz）
    # 値が低い → エネルギーが低周波に集中
    # 値が高い → エネルギーが広い周波数帯に分布
    rolloff = np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr))

    # ============================================================
    # Step 4-d: Spectral Bandwidth（スペクトラル帯域幅）
    # ============================================================
    # スペクトラムの「幅」を表す。エネルギーが分散している度合い
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft))

    # Spectral 4次元ベクトル: [Centroid, Flatness, Rolloff, Bandwidth]

    # ============================================================
    # Step 5-a: Onset Detection（音のインパクト検出）
    # ============================================================
    # Onset: 音が始まる時点
    # Onset Strength: 各フレームのインパクト強度（0.0～1.0）
    # 用途: テンポ・ビート推定の入力
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, n_fft=n_fft)

    # ============================================================
    # Step 5-b: Tempo（テンポ）＆ Beat Track（ビート追跡）
    # ============================================================
    # librosa.beat.beat_track は (tempo, beats) タプルを返す
    #   - tempo: 推定 BPM（Beats Per Minute）
    #   - beats: 検出されたビート位置（フレーム番号）
    #
    # 例）Electronic Dance Music
    #   → tempo ≈ 120 BPM
    #
    # 例）Ballad
    #   → tempo ≈ 60 BPM
    tempo_tuple = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)

    # 戻り値の型チェック（バージョン依存）
    # 古いバージョンでは (np.ndarray, np.ndarray) を返すことがある
    tempo = (
        tempo_tuple[0][0] if isinstance(tempo_tuple[0], np.ndarray) else tempo_tuple[0]
    )

    # ============================================================
    # Step 6-a: Chroma Feature（色彩度）抽出
    # ============================================================
    # Chroma: 音の「色合い」を 12 半音（C, C#, D, ..., B）に分類
    # librosa.feature.chroma_cqt: Constant-Q Transform ベースのクロマ抽出
    #
    # 出力: (12, T) の 2 次元配列
    #   - 12: C～B の 12 半音
    #   - T: 時系列フレーム数
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    # Chroma を時系列で平均化 → (12,) ベクトル
    chroma_mean = np.mean(chroma, axis=1)

    # ============================================================
    # Step 6-b: Key Detection（調推定）
    # ============================================================
    # np.argmax(chroma_mean): 最も強いクロマ成分のインデックス
    # 結果: 0～11（C = 0, C# = 1, ..., B = 11）
    key = np.argmax(chroma_mean)

    # ============================================================
    # Step 6-c: Mode Detection（長調 vs 短調判定）
    # ============================================================
    # 簡易的な長短調判定：
    #   - (key + 4) % 12: 短調における「相対長調」のクロマ位置
    #   - (key + 3) % 12: 短調における「支配7度」のクロマ位置
    #
    # ロジック: (key + 4) のクロマ > (key + 3) のクロマ → 長調傾向 (1.0)
    #          otherwise → 短調傾向 (0.0)
    mode = 1.0 if chroma_mean[(key + 4) % 12] > chroma_mean[(key + 3) % 12] else 0.0

    # ============================================================
    # Step 7: 26 次元ベクトル構成
    # ============================================================
    # 【構成】
    # - mfcc_mean[1:]        : 12 次元（MFCC, 第0係数除外）
    # - Spectral 4次元       : 4 次元（Centroid, Flatness, Rolloff, Bandwidth）
    # - Theoretical 3次元    : 3 次元（Tempo, Key, Mode）
    # - chroma_mean[:7]      : 7 次元（Chroma の最初の 7 半音）
    # ──────────────────────
    # 合計: 12 + 4 + 3 + 7 = 26 次元
    features = np.concatenate(
        [
            mfcc_mean,
            np.array([centroid, flatness, rolloff, bandwidth]),
            np.array([tempo, key, mode]),
            chroma_mean[:7],
        ]
    )

    return features


# ============================================================
# 定数定義
# ============================================================
# MIN_SAMPLES: 特徴抽出に必要な最小サンプル数
# 2048 サンプル ≈ 93ms at 22050 Hz, ≈ 46ms at 44100 Hz
# これより短いセグメントは特徴抽出をスキップ
MIN_SAMPLES = 2048


def create_dataset(input_dir: str, output_json: str) -> None:
    """
    【データセット作成関数】
    セグメント WAV ディレクトリ → A-MAP 形式 JSON + StandardScaler

    Args:
        input_dir: セグメント WAV ファイルが格納されたディレクトリ（data/segments）
        output_json: 出力 JSON ファイルパス（data/dataset.json）
    """

    # ============================================================
    # Step 1: セグメント WAV ファイル一覧を取得
    # ============================================================
    # sorted(...) で example-001, example-002, ... の順序を保証
    audio_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
    total = len(audio_files)

    # 特徴量とファイル ID を格納するコンテナ
    feature_dict = {}  # {"example-001": [...], "example-002": [...], ...}
    all_features = []  # [[...], [...], ...] セグメント数 × 26
    file_ids = []  # ["example-001", "example-002", ...]

    # ============================================================
    # Step 2: 各セグメント WAV から特徴量を抽出
    # ============================================================
    for i, file_path in enumerate(audio_files, 1):
        # ファイル名から ID を抽出（例: example-001.wav → example-001）
        file_id = os.path.basename(file_path).split(".")[0]

        # ============================================================
        # Step 2-a: ファイルサイズチェック（短すぎるセグメントはスキップ）
        # ============================================================
        info = sf.info(file_path)
        duration = info.frames / info.samplerate if info.samplerate else 0.0
        if duration < MIN_DURATION_SECONDS:
            print(
                f"[{i}/{total}] Skip {file_id}: too short ({duration:.2f}s)",
                file=sys.stderr,
                flush=True,
            )
            continue

        # ============================================================
        # Step 2-b: 特徴量抽出
        # ============================================================
        print(f"[{i}/{total}] Extracting: {file_id}", file=sys.stderr, flush=True)
        features = extract_features(file_path)

        # 特徴量を蓄積
        all_features.append(features)
        file_ids.append(file_id)

    # ============================================================
    # Step 3: NumPy 配列に変換（セグメント数 × 26）
    # ============================================================
    # all_features: (N, 26) の 2 次元配列
    #   N = 抽出成功したセグメント数
    #   26 = 特徴量次元数
    all_features = np.array(all_features)

    # ============================================================
    # Step 4: StandardScaler で全セグメントを一括正規化
    # ============================================================
    if len(all_features) > 0:
        # StandardScaler: 各列（特徴量）ごとに平均0、標準偏差1に正規化
        #
        # 【重要】訓練と推論での使い分け
        # - fit_transform: 訓練時。全セグメントの統計量をフィットして変換
        # - transform: 推論時。fit 済み scaler を使って後続セグメントを変換
        #
        # 例）特徴量1（MFCC-1）が [-1.5, -0.5, 0.0, 0.5, 1.5] の場合
        #     → 平均 = 0, 標準偏差 = 1.0
        #     → スケール済み [-1.5, -0.5, 0, 0.5, 1.5]（そのまま）
        scaler = StandardScaler()
        all_features_scaled = scaler.fit_transform(all_features)

        # ============================================================
        # Step 4-a: StandardScaler パラメータを永続化
        # ============================================================
        # scaler.pkl: 後の推論時に、新規セグメントを同じスケール体系で変換するために必要
        scaler_path = os.path.join(os.path.dirname(output_json), "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved to {scaler_path}", file=sys.stderr)

        # ============================================================
        # Step 4-b: スケール済み特徴量を辞書化
        # ============================================================
        for i, file_id in enumerate(file_ids):
            # リスト化（JSON 出力用）
            feature_dict[file_id] = all_features_scaled[i].tolist()

    # ============================================================
    # Step 5: A-MAP 形式でまとめる
    # ============================================================
    # cols: 特徴量次元数（通常 26）
    # data: ID → 特徴量ベクトルのマッピング
    dataset = {
        "cols": all_features.shape[1] if len(all_features) > 0 else 26,
        "data": feature_dict,
    }

    # ============================================================
    # Step 6: JSON に書き出し
    # ============================================================
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Dataset created with {len(file_ids)} segments", file=sys.stderr)


if __name__ == "__main__":
    # ============================================================
    # コマンドラインオプション解析
    # ============================================================
    # 使用例:
    #   python extractor.py
    #   → data/segments → data/dataset.json
    #
    #   python extractor.py --input my_segments --output my_dataset.json
    #   → カスタムパス指定
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
