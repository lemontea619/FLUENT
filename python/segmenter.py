"""
=============================================================================
Module A: Audio Segmenter
=============================================================================

長い音声ファイルを、音響的に異なる部分で自動分割するスクリプト

【処理フロー】
1. raw_audio/ の WAV ファイルを読み込む
2. 各ファイルの MFCC（メル周波数ケプストラム係数）を計算
3. librosa.segment.agglomerative で「自然な切れ目」を検出
4. 切れ目ごとに分割し、各セグメントを example-NNN.wav として保存

【出力例】（--duration 2 の場合、3分曲で約90セグメント）
  example-001.wav (0.5秒)  ← ドラムキック
  example-002.wav (1.2秒)  ← スネアロール
  example-003.wav (2.1秒)  ← ハット
  ...

【粒度オプション】
  --duration 2  （デフォルト）
    → 2秒ごと、細粒度分割
    → セグメント数多い（3分曲 ≈ 90個）
    → リアルタイム推論・精密制御に適した
    → 推奨設定（FLUENT 設計意図と一致）

  --duration 5  （バランス型）
    → 5秒ごと、中程度分割
    → セグメント数中程度（3分曲 ≈ 36個）
    → ラベリング手間と情報量のバランス

  --duration 10  （大構造型）
    → 10秒ごと、粗い分割
    → セグメント数少ない（3分曲 ≈ 18個）
    → ラベリング手間最小化、ただし情報損失

【使用例】
  python segmenter.py
  → data/raw_audio → data/segments（デフォルト: 2秒単位）

  python segmenter.py --duration 5
  → バランス型（5秒単位）で実行

  python segmenter.py --input my_audio --output my_segments --duration 10
  → カスタムパス + 大構造型（10秒単位）で実行

【重要】
  - セグメントには自動的に example-001, example-002, ... という ID が割り当てられる
  - この ID は後の extractor.py・ラベリング UI で必ず参照される（絶対的な紐付け）
  - 1秒未満の短い音源はそのままセグメント扱いにされる
  - duration 値を変えると出力ファイル数が大きく変わるため、
    既存の dataset.json / labelset.json と互換性がなくなる点に注意
"""

import argparse
import glob
import os
import sys

import librosa
import numpy as np
import soundfile as sf


def segment_audio(
    input_dir: str, output_dir: str, duration_per_segment: int = 2
) -> None:
    """
    【メイン処理関数】

    Args:
        input_dir: 入力ディレクトリ（raw_audio/）
        output_dir: 出力ディレクトリ（segments/）
        duration_per_segment: 1セグメントの目標秒数（デフォルト: 2秒）
                             - 2 = 細かい分割（推奨、現状）
                             - 5 = バランス型
                             - 10 = 大構造型
    """

    # ============================================================
    # Step 0: 出力ディレクトリの作成・入力ファイルの収集
    # ============================================================
    os.makedirs(output_dir, exist_ok=True)
    audio_files = glob.glob(os.path.join(input_dir, "*.wav"))
    total = len(audio_files)

    if total == 0:
        print("No WAV files found.", file=sys.stderr)
        return

    # セグメント ID を全体で連続管理（複数ファイルを処理する場合の通し番号）
    segment_id = 1

    # ============================================================
    # Step 1: 各 WAV ファイルを反復処理
    # ============================================================
    for i, file_path in enumerate(audio_files, 1):
        print(
            f"[{i}/{total}] Segmenting: {os.path.basename(file_path)}",
            file=sys.stderr,
            flush=True,
        )

        # ============================================================
        # Step 2: WAV ファイルを読み込み（時系列の音声データと、サンプリングレートを取得）
        # ============================================================
        # y: 音声の波形データ（1次元配列。例: 44100Hz × 30秒 = 1,323,000サンプル）
        # sr: サンプリングレート（Hz。通常 44100 または 48000）
        y, sr = librosa.load(file_path, sr=None)
        duration_total = len(y) / sr

        # MFCC 計算用の "フレーム跳び幅"
        # hop_length が小さいほど時間分解能が上がるが計算量が増える
        # 512サンプル ≈ 11.6ms（44.1kHz の場合）
        hop_length = 512

        # ============================================================
        # Step 3: 1秒未満の短すぎる音源は分割せず、そのまま出力
        # ============================================================
        # len(y) はサンプル数、sr * 1 は 1秒分のサンプル数
        if len(y) < sr * 1:
            # 短い音源 → セグメント分割を行わない
            out_path = os.path.join(output_dir, f"example-{segment_id:03d}.wav")
            sf.write(out_path, y, sr)
            segment_id += 1
            continue

        # ============================================================
        # Step 4: MFCC（メル周波数ケプストラム係数）を計算
        # ============================================================
        # MFCC は「人間の耳に聞こえやすい周波数特性」を13次元で表現
        # librosa.feature.mfcc の出力: (13, T) の2次元配列
        #   - 13: MFCC係数の次元数
        #   - T: 時系列フレーム数（≈音声長 / hop_length）
        #
        # 例）44.1kHz × 30秒の音声の場合
        #   サンプル数: 1,323,000
        #   フレーム数: 1,323,000 / 512 ≈ 2,586フレーム
        #   MFCC出力: (13, 2586)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length)
        smoothed_features = mfcc

        # ============================================================
        # Step 5: セグメント数 k を自動決定
        # ============================================================
        # k: Agglomerative clustering の目標クラスタ数
        #
        # 計算式: max(2, len(y) // (sr * duration_per_segment))
        # 意味: 音声の長さを duration_per_segment 秒ごとに分割する、ただし最小2分割
        #
        # 例）30秒の音声、duration_per_segment=2（現状）
        #   k = max(2, 1,323,000 // (44,100 × 2))
        #     = max(2, 15)
        #     = 15 分割を目指す
        #
        # 例）30秒の音声、duration_per_segment=5（バランス型）
        #   k = max(2, 1,323,000 // (44,100 × 5))
        #     = max(2, 6)
        #     = 6 分割を目指す
        #
        # 短い音声（例: 5秒）
        #   k = max(2, 220,500 // (44,100 × 2)) = max(2, 2) = 2
        k = int(max(2, len(y) // (sr * duration_per_segment)))

        try:
            # ============================================================
            # Step 6: Agglomerative Clustering で分割境界を検出
            # ============================================================
            # librosa.segment.agglomerative:
            #   - 時系列データ（MFCC）から、「似ている部分」をグループ化
            #   - 最終的に k 個のクラスタになるまで統合
            #   - 戻り値: 各フレームが属するクラスタ ID（配列長 = MFCC のフレーム数）
            #
            # 例）MFCC フレーム数が 2586、k=15 の場合
            #   → boundaries = [0, 1, 2, ..., 2, 1, 0, ...]（クラスタ割り当て）
            cluster_labels = librosa.segment.agglomerative(smoothed_features, k)
            boundaries = np.where(np.diff(cluster_labels) != 0)[0] + 1

            # ============================================================
            # Step 7: フレーム番号 → サンプル番号に変換
            # ============================================================
            # agglomerative の出力はフレーム単位
            # → 実際の WAV ファイルを切る際はサンプル番号が必要
            #
            # 例）フレーム 100 → サンプル 100 × 512 = 51,200
            boundary_samples = librosa.frames_to_samples(
                boundaries, hop_length=hop_length
            )

            # ============================================================
            # Step 8: 分割点をソート・重複除去・先頭と末尾を固定
            # ============================================================
            # np.concatenate([[0], boundary_samples, [len(y)]]):
            #   - 0: ファイルの最初（必ず含める）
            #   - boundary_samples: 自動検出した分割点
            #   - len(y): ファイルの最後（必ず含める）
            #
            # 例）boundary_samples = [0, 1000, 2500, 5000, 6000]
            #   → np.unique([0, 0, 1000, 2500, 5000, 6000, len(y), len(y)])
            #   → [0, 1000, 2500, 5000, 6000, len(y)]（重複なし、昇順）
            boundary_samples = np.unique(
                np.concatenate(
                    [[0], librosa.frames_to_samples(boundaries, hop_length=hop_length), [len(y)]]
                )
            )

            # ============================================================
            # Step 9: 隣接する 2 つの分割点の間を 1 セグメントとして抽出
            # ============================================================
            # 例）boundary_samples = [0, 1000, 2500, 5000, len(y)]
            #   セグメント1: y[0:1000]
            #   セグメント2: y[1000:2500]
            #   セグメント3: y[2500:5000]
            #   セグメント4: y[5000:len(y)]
            for j in range(len(boundary_samples) - 1):
                start = boundary_samples[j]
                end = boundary_samples[j + 1]

                # 0長セグメント（start == end）は保存しない
                if end - start > 0:
                    # 波形データをスライス
                    segment_y = y[start:end]

                    # ファイル名: example-001.wav, example-002.wav, ...
                    # :03d は 3 桁ゼロパディング（001, 002, ..., 999）
                    out_path = os.path.join(output_dir, f"example-{segment_id:03d}.wav")

                    # WAV ファイルとして保存
                    sf.write(out_path, segment_y, sr)

                    # 次のセグメント ID へ
                    segment_id += 1

        except Exception as e:
            # ============================================================
            # Step 10: エラー処理（分割に失敗した場合）
            # ============================================================
            # Agglomerative clustering がデータ品質の問題で失敗した場合、
            # ファイル全体を 1 つのセグメントとして保存
            print(f"Error segmenting {file_path}: {e}", file=sys.stderr)
            out_path = os.path.join(output_dir, f"example-{segment_id:03d}.wav")
            sf.write(out_path, y, sr)
            segment_id += 1

    # ============================================================
    # 処理完了
    # ============================================================
    print(f"Done: {segment_id - 1} segments written to {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    # ============================================================
    # コマンドラインオプション解析
    # ============================================================
    # 使用例:
    #   python segmenter.py
    #   → デフォルト: data/raw_audio → data/segments（2秒単位、現状）
    #
    #   python segmenter.py --input my_audio --output my_segments
    #   → カスタム パス指定
    #
    #   python segmenter.py --duration 5
    #   → バランス型（5秒単位）
    #
    #   python segmenter.py --duration 10
    #   → 大構造型（10秒単位）
    parser = argparse.ArgumentParser(description="Segment audio files")
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
        "--duration",
        type=int,
        default=2,
        help="Target duration per segment in seconds (default: 2). Use 2 for fine-grained, 5 for balanced, 10 for macro structure",
    )

    args = parser.parse_args()
    segment_audio(args.input, args.output, duration_per_segment=args.duration)
