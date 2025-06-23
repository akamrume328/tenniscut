# tasks.py (修正後)

import os
import sys
import subprocess
from celery import Celery
import redis
from datetime import datetime

# --- Celeryの設定 ---
celery = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')

# --- Redisクライアントの初期化 ---
# このファイルではRedisを直接使わないが、念のため残しておく
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    print("✅ CeleryタスクファイルからRedisへの接続に成功しました。")
except redis.exceptions.ConnectionError as e:
    print(f"❌ CeleryタスクファイルからRedisへの接続に失敗しました: {e}")
    redis_client = None

def update_progress_in_task(task_id, status, current_step=0, total_steps=0, step_name="", error_message=None):
    """進捗情報をRedisに書き込むヘルパー関数（エラー報告などに使用）"""
    if not redis_client: return
    progress_key = f"pipeline-progress:{task_id}"
    data = { "status": status, "current_step": str(current_step), "total_steps": str(total_steps), "step_name": step_name, "timestamp": datetime.now().isoformat() }
    if error_message: data["error_message"] = error_message
    redis_client.hset(progress_key, mapping=data)
    redis_client.expire(progress_key, 3600)


@celery.task(bind=True)
def run_analysis_task(self, original_video_path, original_filename):
    """
    動画変換と分析パイプラインの起動を、それぞれ独立したPythonスクリプトとして実行するタスク。
    """
    task_id = self.request.id
    print(f"Celeryタスク開始 (ID: {task_id}): {original_filename}")
    
    total_steps = 7 # 総ステップ数: 変換(1) + 分析(6)

    # --- ステップ 1: 動画変換スクリプトの実行 ---
    base_name = os.path.splitext(original_filename)[0]
    upload_folder = os.path.dirname(original_video_path)
    converted_mp4_path = os.path.join(upload_folder, f"{base_name}_converted.mp4")
    
    convert_script_path = os.path.join(os.path.dirname(__file__), 'convert_video.py')

    command_convert = [
        sys.executable,
        convert_script_path,
        '--input', original_video_path,
        '--output', converted_mp4_path,
        '--task-id', task_id,
        '--step-num', '1',
        '--total-steps', str(total_steps)
    ]

    try:
        print(f"[{task_id}] 動画変換スクリプトを同期的に実行します: {' '.join(command_convert)}")
        # subprocess.run を使って、変換スクリプトが完了するまで待機する
        # 注意: この間、Celeryワーカーはこのタスクに専念します。
        result = subprocess.run(command_convert, check=True, capture_output=True, text=True, encoding='utf-8')
        print(f"[{task_id}] 動画変換スクリプトが正常に完了しました。")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        # スクリプトがエラーコードで終了した場合
        error_message = f"動画変換スクリプトの実行に失敗しました。\n{e.stderr}"
        print(f"[{task_id}] ❌ {error_message}")
        # Redisへのエラー報告はスクリプト自身が行うので、ここではCeleryタスクを失敗させるだけでよい
        self.update_state(state='FAILURE', meta={'status': '動画変換に失敗しました。'})
        return {'status': 'Failure', 'error': error_message}
    except Exception as e:
        # その他の予期せぬエラー
        error_message = f"動画変換の呼び出し中に予期せぬエラー: {e}"
        print(f"[{task_id}] ❌ {error_message}")
        update_progress_in_task(task_id, "error", 1, total_steps, "動画変換失敗", error_message=error_message)
        self.update_state(state='FAILURE', meta={'status': error_message})
        return {'status': 'Failure', 'error': error_message}


    # --- ステップ 2: 分析パイプラインスクリプトの実行 ---
    pipeline_script_path = os.path.join(os.path.dirname(__file__), 'run_tennis_pipeline.py')

    command_analyze = [
        sys.executable,
        pipeline_script_path,
        '--video', converted_mp4_path,
        '--original_video_name', original_filename,
        '--task-id', task_id
    ]
    
    try:
        print(f"[{task_id}] 分析パイプラインスクリプトを非同期に実行します: {' '.join(command_analyze)}")
        # subprocess.Popen で分析プロセスをバックグラウンドで起動
        process = subprocess.Popen(command_analyze, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        
        print(f"[{task_id}] 分析プロセスを起動しました。PID: {process.pid}")
        
        # Celeryタスクの状態を「進行中」としてマーク
        self.update_state(state='PROGRESS', meta={'status': f'分析プロセス(PID: {process.pid})を開始しました。'})
        
        return {'status': 'Processing', 'message': f'Analysis process started with PID {process.pid}'}

    except Exception as e:
        import traceback
        error_message = f"分析プロセスの起動中にエラーが発生しました: {e}\n{traceback.format_exc()}"
        print(f"[{task_id}] ❌ {error_message}")
        update_progress_in_task(task_id, "error", 2, total_steps, "分析開始失敗", error_message=str(e))
        self.update_state(state='FAILURE', meta={'status': error_message})
        return {'status': 'Failure', 'error': str(e)}