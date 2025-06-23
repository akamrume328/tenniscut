# convert_video.py (新規作成)

import os
import sys
import subprocess
import argparse
import redis
from datetime import datetime
import json

# --- Redisクライアントの初期化 ---
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
except redis.exceptions.ConnectionError as e:
    print(f"❌ Redisへの接続に失敗しました: {e}", file=sys.stderr)
    redis_client = None
    # Redisに接続できなくても、変換処理自体は試みる
    # sys.exit(1) # 厳密にやるならここで終了

def update_progress(task_id, status, current_step=0, total_steps=0, step_name="", result_path=None, error_message=None):
    """進捗情報をRedisに書き込むヘルパー関数"""
    if not redis_client:
        print("警告: Redisクライアントが利用できないため、進捗を更新できません。", file=sys.stderr)
        return
    
    progress_key = f"pipeline-progress:{task_id}"
    
    data = {
        "status": status,
        "current_step": str(current_step),
        "total_steps": str(total_steps),
        "step_name": step_name,
        "timestamp": datetime.now().isoformat(),
    }
    if result_path:
        data["result_path"] = result_path
    if error_message:
        data["error_message"] = error_message
        
    try:
        redis_client.hset(progress_key, mapping=data)
        redis_client.expire(progress_key, 3600)
    except Exception as e:
        print(f"❌ Redisへの進捗書き込み中にエラー: {e}", file=sys.stderr)

def main():
    """メインの実行関数"""
    parser = argparse.ArgumentParser(description='動画をWeb互換のMP4形式に変換します。')
    parser.add_argument('--input', required=True, help='入力ビデオファイルのパス')
    parser.add_argument('--output', required=True, help='出力ビデオファイルのパス')
    parser.add_argument('--task-id', required=True, help='対応するCeleryタスクID')
    parser.add_argument('--step-num', required=True, type=int, help='このプロセスのステップ番号')
    parser.add_argument('--total-steps', required=True, type=int, help='パイプライン全体の総ステップ数')
    args = parser.parse_args()

    step_name = "動画をWeb再生用に変換中..."
    
    # 処理開始を進捗として報告
    update_progress(args.task_id, "processing", args.step_num, args.total_steps, step_name)
    
    print(f"[{args.task_id}] 動画変換を開始: {args.input} -> {args.output}")
    
    command = [
        'ffmpeg',
        '-i', args.input,
        '-y',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'fast',
        '-crf', '23',
        '-c:a', 'aac',
        '-b:a', '128k',
        args.output
    ]

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            error_message = f"ffmpegの実行に失敗 (コード: {process.returncode})。\nエラー:\n{stderr}"
            print(f"[{args.task_id}] ❌ {error_message}", file=sys.stderr)
            update_progress(args.task_id, "error", args.step_num, args.total_steps, "動画変換失敗", error_message=error_message)
            sys.exit(1) # エラーコード 1 で終了
            
        print(f"[{args.task_id}] ✅ 動画変換が正常に完了しました: {args.output}")
        # このスクリプトは成功/失敗をリターンコードで伝えるのみ。
        # 次のステップの進捗報告は呼び出し元のCeleryタスクが行う。
        sys.exit(0) # 正常終了

    except FileNotFoundError:
        error_message = "ffmpegコマンドが見つかりません。システムにffmpegがインストールされ、PATHが通っているか確認してください。"
        print(f"[{args.task_id}] ❌ {error_message}", file=sys.stderr)
        update_progress(args.task_id, "error", args.step_num, args.total_steps, "動画変換失敗", error_message=error_message)
        sys.exit(1)
    except Exception as e:
        error_message = f"動画変換中に予期せぬエラーが発生しました: {e}"
        print(f"[{args.taks_id}] ❌ {error_message}", file=sys.stderr)
        update_progress(args.task_id, "error", args.step_num, args.total_steps, "動画変換失敗", error_message=str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()