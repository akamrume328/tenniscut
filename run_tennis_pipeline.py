# run_tennis_pipeline.py (修正後)

import os
import cv2
import pandas as pd
from datetime import datetime
from pathlib import Path
import numpy as np
import time
import argparse
import sys
from types import SimpleNamespace
import redis
import json

# --- 各モジュールから、実際に定義されているクラス/関数を正しくインポート ---
from balltracking import BallTracker
from feature_extractor_predict import TennisInferenceFeatureExtractor
from predict_lstm_model import TennisLSTMPredictor
from overlay_predictions import PredictionOverlay
from court_calibrator import CourtCalibrator
from hmm_postprocessor import HMMSupervisedPostprocessor
from typing import Optional
from cut_non_rally_segments import cut_rally_segments

# --- Redisクライアントの初期化 ---
# CeleryのBrokerと同じRedisインスタンスに接続します
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping() # 接続確認
    print("✅ Redisへの接続に成功しました。")
except redis.exceptions.ConnectionError as e:
    print(f"❌ Redisへの接続に失敗しました: {e}")
    print("   Redisサーバーが起動しているか、ホストとポートが正しいか確認してください。")
    redis_client = None

def update_progress(task_id, status, current_step=0, total_steps=0, step_name="", result_path=None, error_message=None):
    """進捗情報をRedisに書き込む関数"""
    if not redis_client:
        return
    
    progress_key = f"pipeline-progress:{task_id}"
    
    data = {
        "status": status, # "processing", "completed", "error"
        "current_step": current_step,
        "total_steps": total_steps,
        "step_name": step_name,
        "timestamp": datetime.now().isoformat(),
    }
    if result_path:
        data["result_path"] = result_path
    if error_message:
        data["error_message"] = error_message
        
    try:
        # HSETを使って複数のフィールドを一度に設定
        redis_client.hset(progress_key, mapping=data)
        # 1時間後にキーが自動的に削除されるように設定
        redis_client.expire(progress_key, 3600) 
    except Exception as e:
        print(f"❌ Redisへの進捗書き込み中にエラーが発生しました: {e}")


def run_pipeline(args):
    """テニス分析の完全なパイプラインを実行する。"""
    pipeline_start_time = time.time()
    step_times = {}

    task_id = args.task_id
    total_steps = 6 # このパイプラインの総ステップ数

    try:
        # --- パイプラインの各処理 ---
        video_path = Path(args.video)
        
        if '_converted' in video_path.name:
            video_stem = Path(args.original_video_name).stem
        else:
            video_stem = video_path.stem

        output_dir = Path(args.output_dir)

        # ディレクトリ作成
        calibration_output_dir = output_dir / "00_calibration_data"
        tracking_output_dir = output_dir / "01_tracking_data"
        features_output_dir = output_dir / "02_extracted_features"
        predictions_output_dir = output_dir / "03_lstm_predictions"
        hmm_output_dir = output_dir / "03a_hmm_processed_predictions"
        rally_extract_output_dir = output_dir / "06_rally_extract"
        for p_dir in [calibration_output_dir, tracking_output_dir, features_output_dir, predictions_output_dir, hmm_output_dir, rally_extract_output_dir]:
            p_dir.mkdir(parents=True, exist_ok=True)

        # --- ステップ 1: コートキャリブレーション (ファイル検索) ---
        update_progress(task_id, "processing", 1, total_steps, "コートキャリブレーションファイルの検索")
        # (このステップは非常に短いので、開始と終了の報告は省略)
        court_data_source_dir_for_feature_extraction = None
        calibration_files = sorted(list(calibration_output_dir.glob(f"court_coords_{video_stem}_*.json")), key=lambda p: p.stat().st_mtime, reverse=True)
        if calibration_files:
            latest_calibration_file = calibration_files[0]
            court_data_source_dir_for_feature_extraction = str(latest_calibration_file.parent)
            print(f"✅ Web UIで設定された座標を使用します: {latest_calibration_file.name}")
        else:
            print(f"⚠️ Web UIで設定された座標ファイルが見つかりません。")

        # --- ステップ 2: ボールトラッキング ---
        update_progress(task_id, "processing", 2, total_steps, "ボールトラッキング")
        step_2_start_time = time.time()
        tracker = BallTracker(model_path=args.yolo_model, imgsz=args.imgsz, save_training_data=True, data_dir=str(tracking_output_dir), frame_skip=args.frame_skip)
        fps, _, _, total_frames = tracker.initialize_video_processing(str(video_path))
        while True:
            ret, frame, frame_number = tracker.read_next_frame()
            if not ret: break
            tracker.process_frame_core(frame, frame_number, is_lightweight=True)
        tracker.release_video_resources()
        tracker.save_tracking_features_with_video_info(video_stem, fps, total_frames)
        tracking_json_files = sorted(list(tracking_output_dir.glob(f"tracking_features_{video_stem}_*.json")), key=lambda p: p.stat().st_mtime, reverse=True)
        if not tracking_json_files: raise FileNotFoundError("トラッキングJSONが見つかりません")
        step_times["ステップ 2 (ボールトラッキング)"] = time.time() - step_2_start_time
        print(f"ステップ 2 完了 ({step_times['ステップ 2 (ボールトラッキング)']:.2f}秒)")

        # --- ステップ 3: 特徴量抽出 ---
        update_progress(task_id, "processing", 3, total_steps, "特徴量抽出")
        step_3_start_time = time.time()
        feature_extractor = TennisInferenceFeatureExtractor(inference_data_dir=str(tracking_output_dir), court_data_dir=court_data_source_dir_for_feature_extraction)
        feature_extractor.features_dir = features_output_dir
        _, features_csv_path_str, _ = feature_extractor.run_feature_extraction(video_name=video_stem, save_results=True)
        if not features_csv_path_str: raise RuntimeError("特徴量抽出に失敗しました。")
        features_csv_path = Path(features_csv_path_str)
        step_times["ステップ 3 (特徴量抽出)"] = time.time() - step_3_start_time
        print(f"ステップ 3 完了 ({step_times['ステップ 3 (特徴量抽出)']:.2f}秒)")

        # --- ステップ 4: LSTM予測 ---
        update_progress(task_id, "processing", 4, total_steps, "AIモデルによる局面予測 (LSTM)")
        step_4_start_time = time.time()
        predictor = TennisLSTMPredictor(models_dir=str(Path(args.lstm_model).parent), input_features_dir=str(features_output_dir))
        predictor.predictions_output_dir = predictions_output_dir
        prediction_csv_path = predictor.run_prediction_for_file(model_set_path=Path(args.lstm_model), feature_csv_path=features_csv_path)
        if not prediction_csv_path: raise RuntimeError("LSTM予測に失敗しました。")
        step_times["ステップ 4 (LSTM予測)"] = time.time() - step_4_start_time
        print(f"ステップ 4 完了 ({step_times['ステップ 4 (LSTM予測)']:.2f}秒)")

        # --- ステップ 5: HMMによる後処理 ---
        update_progress(task_id, "processing", 5, total_steps, "予測結果の平滑化 (HMM)")
        step_5_start_time = time.time()
        hmm_postprocessor = HMMSupervisedPostprocessor(verbose=False)
        hmm_processed_csv_path = prediction_csv_path
        if args.hmm_model_path and Path(args.hmm_model_path).exists():
            if hmm_postprocessor.load_hmm_model(Path(args.hmm_model_path)):
                if hmm_postprocessor.load_data(data_csv_path=prediction_csv_path, pred_col_name='predicted_phase'):
                    smoothed_sequence_int = hmm_postprocessor.smooth()
                    if smoothed_sequence_int is not None:
                        smoothed_labels = np.array([hmm_postprocessor.int_to_label.get(s, "UNKNOWN_STATE") for s in smoothed_sequence_int])
                        df_hmm = hmm_postprocessor.df_loaded.copy()
                        df_hmm.loc[hmm_postprocessor.final_mask_for_hmm, 'predicted_phase'] = smoothed_labels
                        saved_path = hmm_postprocessor.save_results(df_hmm, prediction_csv_path, output_base_dir=hmm_output_dir)
                        if saved_path: hmm_processed_csv_path = saved_path
        step_times["ステップ 5 (HMM後処理)"] = time.time() - step_5_start_time
        print(f"ステップ 5 完了 ({step_times['ステップ 5 (HMM後処理)']:.2f}秒)")

        # --- ステップ 6: ラリー抽出 ---
        update_progress(task_id, "processing", 6, total_steps, "ラリー動画の切り出し")
        step_6_start_time = time.time()
        rally_video_filename = f"{video_stem}_rallies.mp4"
        output_rally_video_path = rally_extract_output_dir / rally_video_filename
        cut_rally_segments(video_path=video_path, csv_path=hmm_processed_csv_path, output_path=output_rally_video_path,
                           buffer_before=args.rally_buffer_before_seconds, buffer_after=args.rally_buffer_after_seconds,
                           min_rally_duration=args.min_rally_duration_seconds, min_phase_duration=args.min_phase_duration_seconds)
        step_times["ステップ 6 (Rally区間の抽出)"] = time.time() - step_6_start_time
        print(f"ステップ 6 完了 ({step_times['ステップ 6 (Rally区間の抽出)']:.2f}秒)")
        
        # --- 完了報告 ---
        final_video_path_str = str(output_rally_video_path.name)
        update_progress(task_id, "completed", total_steps, total_steps, "完了", result_path=final_video_path_str)
        print(f"\n✅ パイプラインが正常に完了しました。総処理時間: {time.time() - pipeline_start_time:.2f}秒")
        
        return final_video_path_str

    except Exception as e:
        # --- エラー報告 ---
        import traceback
        error_message = f"パイプライン実行中にエラーが発生しました: {e}\n{traceback.format_exc()}"
        print(f"❌ {error_message}")
        update_progress(task_id, "error", error_message=error_message)
        return None


def execute_pipeline(video_path, original_video_name, task_id):
    """
    Celeryタスクから呼び出されるためのメイン関数。
    固定のパラメータを設定し、パイプラインを実行する。
    """
    print("--- execute_pipeline関数が呼び出されました ---")
    
    args = SimpleNamespace(
        video=video_path,
        original_video_name=original_video_name,
        task_id=task_id, # タスクIDを追加
        yolo_model="models/yolo_model/best_5_31.pt",
        lstm_model="models/lstm_model",
        hmm_model_path="models/hmm_model/hmm_model_supervised.joblib",
        output_dir='tennis_pipeline_output',
        frame_skip=10,
        imgsz=1920,
        rally_buffer_before_seconds=3.0,
        rally_buffer_after_seconds=2.0,
        min_rally_duration_seconds=2.0,
        min_phase_duration_seconds=0.5
    )
    
    return run_pipeline(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='テニスビデオ分析パイプライン')
    parser.add_argument('--video', required=True, type=str, help='処理するビデオファイルのパス')
    parser.add_argument('--original_video_name', required=True, type=str, help='Webアプリからアップロードされた元のファイル名')
    # ★★★ タスクIDを受け取るための引数を追加 ★★★
    parser.add_argument('--task-id', required=True, type=str, help='この処理に対応するCeleryのタスクID')
    
    cli_args = parser.parse_args()
    
    print("--- コマンドラインから直接実行します ---")
    
    execute_pipeline(
        video_path=cli_args.video,
        original_video_name=cli_args.original_video_name,
        task_id=cli_args.task_id
    )