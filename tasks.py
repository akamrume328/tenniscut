# tasks.py (最終提案版)

from celery import Celery
import sys
import os
import subprocess
import json
import cv2
import time

celery = Celery('tasks', 
                broker='redis://localhost:6379/0', 
                backend='redis://localhost:6379/0')

# convert_to_standard_mp4 と STATS_FILE の定義は変更なし
# ...

@celery.task(bind=True)
def run_analysis_task(self, original_video_path, original_filename):
    """
    分析パイプラインを独立したプロセスとして起動し、即座に終了するタスク。
    """
    print(f"Celeryタスク開始: {original_filename} の分析プロセスを起動します。")

    # ... 動画変換までのロジックは変更なし ...
    # ... （変換に失敗したらreturnする部分も同じ）
    base_name = os.path.splitext(original_filename)[0]
    upload_folder = os.path.dirname(original_video_path)
    converted_mp4_path = os.path.join(upload_folder, f"{base_name}_converted.mp4")

    if not convert_to_standard_mp4(original_video_path, converted_mp4_path):
        self.update_state(state='FAILURE', meta={'status': '動画のMP4変換に失敗しました。'})
        return {'status': 'Failure', 'error': '動画のMP4変換に失敗しました。'}

    # ★★★★★ ここからが大きな変更点 ★★★★★
    try:
        command = [
            sys.executable,  # Celeryワーカーと同じPythonインタプリタを使用
            'run_tennis_pipeline.py',
            '--video', converted_mp4_path,
            '--original_video_name', original_filename
        ]
        
        print(f"以下のコマンドでバックグラウンドプロセスを開始します: {' '.join(command)}")

        # subprocess.Popenを使い、プロセスを起動するだけ。
        # 標準出力やエラー出力をパイプに接続しないことで、プロセス間のI/Oをなくす。
        # これにより、分析プロセスは完全に独立して動作する。
        process = subprocess.Popen(command)
        
        # プロセスの起動を確認
        print(f"分析プロセスを起動しました。PID: {process.pid}")
        print("Celeryタスクはここで完了します。分析はバックグラウンドで継続されます。")

        # このタスクはすぐに成功を返す
        # 実際の分析結果は、別途ファイルやデータベースで確認する仕組みが必要
        self.update_state(state='PROCESSING', meta={'status': f'分析プロセス(PID: {process.pid})を開始しました。'})
        return {'status': 'Success', 'message': f'Process started with PID {process.pid}'}

    except Exception as e:
        print(f"プロセスの起動中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        self.update_state(state='FAILURE', meta={'status': f'プロセスの起動失敗: {str(e)}'})
        return {'status': 'Failure', 'error': str(e)}