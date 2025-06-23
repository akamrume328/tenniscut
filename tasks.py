from celery import Celery
import subprocess
import sys
import os
import ffmpeg
import json
import cv2
import time

celery = Celery('tasks', 
                broker='redis://localhost:6379/0', 
                backend='redis://localhost:6379/0')

STATS_FILE = 'performance_stats.json'

def convert_to_standard_mp4(source_path, output_path):
    # この関数は変更ありません
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

# ★★★ ここから下の関数を丸ごと置き換えてください ★★★
@celery.task(bind=True)
def run_analysis_task(self, original_video_path, original_filename):
    """ 動画変換と分析パイプラインをバックグラウンドで実行するCeleryタスク（リアルタイムログ対応版） """
    print(f"Celeryタスク開始: {original_filename}")

    total_frames = 0
    try:
        # この部分は変更ありません（処理時間の目安計算）
        with open(STATS_FILE, 'r') as f:
            stats = json.load(f)
        cap = cv2.VideoCapture(original_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        estimated_seconds = total_frames * stats.get('avg_time_per_frame', 0.05)
        self.update_state(state='PROGRESS', meta={'estimated_time': estimated_seconds, 'status': '動画変換中...'})
    except Exception as e:
        print(f"目安時間の計算中にエラー: {e}")

    start_time = time.time()

    base_name = os.path.splitext(original_filename)[0]
    upload_folder = os.path.dirname(original_video_path)
    converted_mp4_path = os.path.join(upload_folder, f"{base_name}.mp4")

    if not convert_to_standard_mp4(original_video_path, converted_mp4_path):
        self.update_state(state='FAILURE', meta={'status': '動画のMP4変換に失敗しました。'})
        return {'status': 'Failure', 'error': '動画のMP4変換に失敗しました。'}
    
    self.update_state(state='PROGRESS', meta={'estimated_time': estimated_seconds, 'status': '分析パイプラインを実行中...'})

    try:
        # 実行するコマンドをリスト形式で準備
        command = [
            sys.executable, 'run_tennis_pipeline.py',
            '--video', converted_mp4_path,
            '--original_video_name', original_filename
        ]
        
        print(f"--- 分析パイプラインを実行します ---\nコマンド: {' '.join(command)}")

        # subprocess.Popenを使い、リアルタイムで出力を取得
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # エラー出力を標準出力にまとめる
            encoding='utf-8',
            errors='ignore',
            bufsize=1,  # 行バッファリングを有効に
            universal_newlines=True
        )

        # リアルタイムで出力を読み取り、ログに表示
        for line in iter(process.stdout.readline, ''):
            log_line = line.strip()
            if log_line:
                print(log_line) # Celeryワーカーのコンソールにログを表示
                # Web UIに進捗を通知
                self.update_state(state='PROGRESS', meta={'estimated_time': estimated_seconds, 'status': log_line})
        
        # プロセスの終了を待つ
        process.wait()

        # パイプラインの実行が成功したかチェック
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)

        # --- 正常終了した場合の処理 ---
        print("--- 分析パイプラインが正常に完了しました ---")
        duration = time.time() - start_time
        
        # パフォーマンス統計の更新
        if total_frames > 0:
            try:
                # (統計更新のロジックは変更なし)
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

        # 最終的な結果ファイル名を組み立てて返す
        result_filename = f"{base_name}_rallies.mp4"
        return {'status': 'Success', 'result_file': result_filename}

    except subprocess.CalledProcessError as e:
        error_message = f"分析パイプラインの実行に失敗しました (終了コード: {e.returncode})。"
        print(error_message)
        self.update_state(state='FAILURE', meta={'status': error_message})
        return {'status': 'Failure', 'error': error_message}
    except Exception as e:
        # その他の予期せぬエラー
        print(f"タスク実行中に予期せぬエラーが発生しました: {e}")
        self.update_state(state='FAILURE', meta={'status': f'予期せぬエラー: {str(e)}'})
        return {'status': 'Failure', 'error': str(e)}