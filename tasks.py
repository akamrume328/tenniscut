from celery import Celery
import subprocess
import sys
import os
import ffmpeg
import json
import cv2
import time # ★ timeモジュールをインポート

celery = Celery('tasks', 
                broker='redis://localhost:6379/0', 
                backend='redis://localhost:6379/0')

STATS_FILE = 'performance_stats.json'

def convert_to_standard_mp4(source_path, output_path):
    print(f"バックグラウンドで動画をMP4形式に変換します (GPU使用): {source_path} -> {output_path}")
    try:
        (ffmpeg.input(source_path).output(output_path, vcodec='h264_nvenc', acodec='aac', pix_fmt='yuv420p').overwrite_output().run(capture_stdout=True, capture_stderr=True))
        print("動画変換が完了しました (GPU)。")
        return True
    except ffmpeg.Error:
        print("GPUエンコードに失敗したため、CPUでの変換を試みます...")
        try:
            (ffmpeg.input(source_path).output(output_path, vcodec='libx264', acodec='aac', pix_fmt='yuv420p').overwrite_output().run(capture_stdout=True, capture_stderr=True))
            print("動画変換が完了しました (CPU)。")
            return True
        except ffmpeg.Error as e2:
            print("CPUでの変換にも失敗しました:", e2.stderr.decode())
            return False

@celery.task(bind=True)
def run_analysis_task(self, original_video_path, original_filename):
    """ 動画変換と分析パイプラインをバックグラウンドで実行するCeleryタスク """
    print(f"Celeryタスク開始: {original_filename}")

    total_frames = 0
    try:
        with open(STATS_FILE, 'r') as f:
            stats = json.load(f)
        cap = cv2.VideoCapture(original_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        estimated_seconds = total_frames * stats.get('avg_time_per_frame', 0.05)
        # 状態を「進行中」として、目安時間をmeta情報に保存
        self.update_state(state='PROGRESS', meta={'estimated_time': estimated_seconds})
    except Exception as e:
        print(f"目安時間の計算中にエラー: {e}")

    start_time = time.time() # Pythonの正しい時間取得方法

    base_name = os.path.splitext(original_filename)[0]
    upload_folder = os.path.dirname(original_video_path)
    converted_mp4_path = os.path.join(upload_folder, f"{base_name}_converted.mp4")

    if not convert_to_standard_mp4(original_video_path, converted_mp4_path):
        return {'status': 'Failure', 'error': '動画のMP4変換に失敗しました。'}

    try:
        result = subprocess.run([
            sys.executable, 'run_tennis_pipeline.py',
            '--video', converted_mp4_path, '--original_video_name', original_filename
        ], check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')

        duration = time.time() - start_time # Pythonの正しい処理時間計算

        if total_frames > 0:
            try:
                with open(STATS_FILE, 'r+') as f:
                    stats = json.load(f)
                    stats['total_duration'] = stats.get('total_duration', 0.0) + duration
                    stats['total_frames'] = stats.get('total_frames', 0) + total_frames
                    if stats['total_frames'] > 0:
                         stats['avg_time_per_frame'] = stats['total_duration'] / stats['total_frames']
                    f.seek(0)
                    json.dump(stats, f, indent=4)
                    f.truncate()
                print(f"パフォーマンス実績を更新しました: 1フレームあたり {stats['avg_time_per_frame']:.3f} 秒")
            except Exception as e:
                print(f"実績の更新中にエラー: {e}")

        result_filename = f"{base_name}_rallies.mp4"
        return {'status': 'Success', 'result_file': result_filename, 'stdout': result.stdout}

    except subprocess.CalledProcessError as e:
        return {'status': 'Failure', 'error': e.stderr}