from celery import Celery
import subprocess
import sys
import os

# Celeryアプリケーションを作成
# 第1引数は現在のモジュール名、brokerとbackendにRedisのURLを指定
celery = Celery('tasks', 
                broker='redis://localhost:6379/0', 
                backend='redis://localhost:6379/0')

@celery.task
def run_analysis_task(converted_video_path, original_filename):
    """
    時間のかかる動画分析パイプラインをバックグラウンドで実行するCeleryタスク。
    """
    print(f"Celeryタスク開始: {original_filename}")
    try:
        # run_tennis_pipeline.py をサブプロセスとして実行
        result = subprocess.run([
            sys.executable,
            'run_tennis_pipeline.py',
            '--video', converted_video_path,
            '--original_video_name', original_filename
        ], check=True, capture_output=True, text=True, encoding='utf-8', errors='ignore')

        print(f"タスク完了: {original_filename}")
        print(f"パイプライン出力: {result.stdout}")

        # 成功した場合、結果動画のファイル名を返す
        base_name = os.path.splitext(original_filename)[0]
        result_filename = f"{base_name}_rallies.mp4"
        return {'status': 'Success', 'result_file': result_filename}

    except subprocess.CalledProcessError as e:
        print(f"タスクでエラーが発生: {original_filename}")
        print(f"標準エラー出力: {e.stderr}")
        # 失敗した場合、エラーメッセージを返す
        return {'status': 'Failure', 'error': e.stderr}