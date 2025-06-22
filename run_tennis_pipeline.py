import os
import cv2
import pandas as pd
from datetime import datetime
from pathlib import Path
import numpy as np
import time
import argparse
import sys

# --- 各モジュールから、実際に定義されているクラス/関数を正しくインポート ---
from balltracking import BallTracker
from feature_extractor_predict import TennisInferenceFeatureExtractor
from predict_lstm_model import TennisLSTMPredictor
from overlay_predictions import PredictionOverlay
from court_calibrator import CourtCalibrator
from hmm_postprocessor import HMMSupervisedPostprocessor
from typing import Optional
from cut_non_rally_segments import cut_rally_segments

def run_pipeline(args):
    """テニス分析の完全なパイプラインを実行する。"""
    pipeline_start_time = time.time()
    step_times = {}

    video_path = Path(args.video)
    
    if '_converted' in video_path.name:
        video_stem = Path(args.original_video_name).stem
    else:
        video_stem = video_path.stem

    output_dir = Path(args.output_dir)

    # --- 各種出力ディレクトリの作成 ---
    calibration_output_dir = output_dir / "00_calibration_data"
    tracking_output_dir = output_dir / "01_tracking_data"
    features_output_dir = output_dir / "02_extracted_features"
    predictions_output_dir = output_dir / "03_lstm_predictions"
    hmm_output_dir = output_dir / "03a_hmm_processed_predictions"
    rally_extract_output_dir = output_dir / "06_rally_extract"

    for p_dir in [calibration_output_dir, tracking_output_dir, features_output_dir, predictions_output_dir, hmm_output_dir, rally_extract_output_dir]:
        p_dir.mkdir(parents=True, exist_ok=True)

    print(f"パイプライン開始: {video_path}")
    print(f"フレームスキップ: {args.frame_skip}")

    # ★★★↓ここから修正↓★★★
    # --- ステップ 1: コートキャリブレーション (Web UIで実行済みのため、ファイル検索を行う) ---
    print("\n--- ステップ 1: コートキャリブレーションファイルの検索 ---")
    
    court_data_source_dir_for_feature_extraction = None
    # Web UIで保存された最新の座標ファイルを探す
    calibration_files = sorted(
        list(calibration_output_dir.glob(f"court_coords_{video_stem}_*.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    if calibration_files:
        # 最新のファイルを使用
        latest_calibration_file = calibration_files[0]
        # TennisInferenceFeatureExtractorはディレクトリを期待するので親ディレクトリを渡す
        court_data_source_dir_for_feature_extraction = str(latest_calibration_file.parent)
        print(f"✅ Web UIで設定された座標を使用します: {latest_calibration_file.name}")
    else:
        print(f"⚠️ Web UIで設定された座標ファイルが見つかりません。({calibration_output_dir} 内)")
        # この場合、court_data_source_dir_for_feature_extraction は None のまま
    # ★★★↑ここまで修正↑★★★

    # --- ステップ 2: ボールトラッキング ---
    print("\n--- ステップ 2: ボールトラッキング ---")
    step_2_start_time = time.time()
    tracker = BallTracker(
        model_path=args.yolo_model,
        imgsz=args.imgsz,
        save_training_data=True,
        data_dir=str(tracking_output_dir),
        frame_skip=args.frame_skip
    )
    fps, _, _, total_frames = tracker.initialize_video_processing(str(video_path))
    
    frame_count = 0
    while True:
        ret, frame, frame_number = tracker.read_next_frame()
        if not ret:
            break
        frame_count = frame_number
        tracker.process_frame_core(frame, frame_number, is_lightweight=True)
    tracker.release_video_resources()
    tracker.save_tracking_features_with_video_info(video_stem, fps, total_frames)
    tracking_json_files = sorted(list(tracking_output_dir.glob(f"tracking_features_{video_stem}_*.json")), key=lambda p: p.stat().st_mtime, reverse=True)
    if not tracking_json_files:
        print(f"エラー: トラッキングJSONが見つかりません")
        return
    step_times["ステップ 2 (ボールトラッキング)"] = time.time() - step_2_start_time
    print(f"ステップ 2 完了 ({step_times['ステップ 2 (ボールトラッキング)']:.2f}秒)")

    # --- ステップ 3: 特徴量抽出 ---
    print("\n--- ステップ 3: 特徴量抽出 ---")
    step_3_start_time = time.time()
    feature_extractor = TennisInferenceFeatureExtractor(
        inference_data_dir=str(tracking_output_dir),
        court_data_dir=court_data_source_dir_for_feature_extraction # ★★★ 検索結果をここで使用
    )
    feature_extractor.features_dir = features_output_dir
    _, features_csv_path_str, _ = feature_extractor.run_feature_extraction(video_name=video_stem, save_results=True)
    if not features_csv_path_str:
        print("エラー: 特徴量抽出に失敗しました。")
        return
    features_csv_path = Path(features_csv_path_str)
    step_times["ステップ 3 (特徴量抽出)"] = time.time() - step_3_start_time
    print(f"ステップ 3 完了 ({step_times['ステップ 3 (特徴量抽出)']:.2f}秒)")

    # ...以降のステップは変更なし...
    # --- ステップ 4: LSTM予測 ---
    print("\n--- ステップ 4: LSTM予測 ---")
    step_4_start_time = time.time()
    predictor = TennisLSTMPredictor(
        models_dir=str(Path(args.lstm_model).parent),
        input_features_dir=str(features_output_dir)
    )
    predictor.predictions_output_dir = predictions_output_dir
    prediction_csv_path = predictor.run_prediction_for_file(
        model_set_path=Path(args.lstm_model),
        feature_csv_path=features_csv_path
    )
    if not prediction_csv_path:
        print("エラー: LSTM予測に失敗しました。")
        return
    step_times["ステップ 4 (LSTM予測)"] = time.time() - step_4_start_time
    print(f"ステップ 4 完了 ({step_times['ステップ 4 (LSTM予測)']:.2f}秒)")

    # --- ステップ 4.5: HMMによる後処理 ---
    print("\n--- ステップ 4.5: HMMによる後処理 ---")
    step_4_5_start_time = time.time()
    hmm_postprocessor = HMMSupervisedPostprocessor(verbose=False, random_state=42)
    hmm_processed_csv_path = prediction_csv_path
    if args.hmm_model_path and Path(args.hmm_model_path).exists():
        if hmm_postprocessor.load_hmm_model(Path(args.hmm_model_path)):
            if hmm_postprocessor.load_data(data_csv_path=prediction_csv_path, pred_col_name='predicted_phase'):
                smoothed_sequence_int = hmm_postprocessor.smooth()
                if smoothed_sequence_int is not None:
                    smoothed_labels = np.array([hmm_postprocessor.int_to_label.get(s, "UNKNOWN_STATE") for s in smoothed_sequence_int])
                    hmm_postprocessor.add_smoothed_results_to_df(smoothed_labels)
                    saved_path = hmm_postprocessor.save_results(hmm_postprocessor.df_loaded, prediction_csv_path, output_base_dir=hmm_output_dir)
                    if saved_path:
                        hmm_processed_csv_path = saved_path
    step_times["ステップ 4.5 (HMM後処理)"] = time.time() - step_4_5_start_time
    print(f"ステップ 4.5 完了 ({step_times['ステップ 4.5 (HMM後処理)']:.2f}秒)")

    # --- ステップ 6: ラリー抽出 ---
    print("\n--- ステップ 6: Rally区間の抽出 ---")
    step_6_start_time = time.time()
    rally_video_filename = f"{video_stem}_rallies.mp4"
    output_rally_video_path = rally_extract_output_dir / rally_video_filename

    cut_rally_segments(
        video_path=video_path,
        csv_path=hmm_processed_csv_path,
        output_path=output_rally_video_path,
        buffer_before=args.rally_buffer_before_seconds,
        buffer_after=args.rally_buffer_after_seconds,
        min_rally_duration=args.min_rally_duration_seconds,
        min_phase_duration=args.min_phase_duration_seconds
    )
    step_times["ステップ 6 (Rally区間の抽出)"] = time.time() - step_6_start_time
    print(f"ステップ 6 完了 ({step_times['ステップ 6 (Rally区間の抽出)']:.2f}秒)")
    
    print(f"\nパイプラインが完了しました。総処理時間: {time.time() - pipeline_start_time:.2f}秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='テニスビデオ分析パイプライン')
    parser.add_argument('--video', required=True, type=str, help='処理するビデオファイルのパス')
    parser.add_argument('--original_video_name', required=True, type=str, help='Webアプリからアップロードされた元のファイル名')
    
    args = parser.parse_args()
    
    # 固定モデルパスをargsに追加
    args.yolo_model = "models/yolo_model/best_5_31.pt"
    args.lstm_model = "models/lstm_model" # フォルダを指定
    args.hmm_model_path = "models/hmm_model/hmm_model_supervised.joblib"
    
    # パイプラインで固定するその他のパラメータ
    args.output_dir = 'tennis_pipeline_output'
    args.frame_skip = 10
    args.imgsz = 1920
    args.extract_rally_mode = True
    args.rally_buffer_before_seconds = 3.0
    args.rally_buffer_after_seconds = 2.0
    args.min_rally_duration_seconds = 2.0
    args.min_phase_duration_seconds = 0.5
    
    run_pipeline(args)