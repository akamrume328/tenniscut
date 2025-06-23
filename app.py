# app.py (全面的な修正)

import os
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from datetime import datetime
import redis
import json
from pathlib import Path

# tasks.py から Celery タスクをインポート
from tasks import run_analysis_task

# --- Flaskアプリと設定 ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['TENNIS_OUTPUT_FOLDER'] = 'tennis_pipeline_output'
app.config['MAX_CONTENT_LENGTH'] = 300 * 1024 * 1024  # 300MB
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}

# --- ディレクトリの作成 ---
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['TENNIS_OUTPUT_FOLDER'], exist_ok=True)

# --- Redisクライアントの初期化 ---
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    redis_client.ping()
    print("✅ FlaskアプリからRedisへの接続に成功しました。")
except redis.exceptions.ConnectionError as e:
    print(f"❌ FlaskアプリからRedisへの接続に失敗しました: {e}")
    print("   進捗表示機能と履歴の一部が正しく動作しない可能性があります。")
    redis_client = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            original_video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(original_video_path)
            
            # --- Celeryタスクを起動 ---
            # 動画変換と分析はすべてバックグラウンドタスクに任せる
            task = run_analysis_task.delay(original_video_path, filename)

            # タスク情報をRedisに保存 (履歴表示用)
            if redis_client:
                task_info = {
                    'task_id': task.id,
                    'filename': filename,
                    'upload_time': datetime.now().isoformat(),
                    'state': 'PENDING' # 初期状態
                }
                redis_client.hset('task_history', task.id, json.dumps(task_info))

            return redirect(url_for('history'))

    return render_template('index.html')

@app.route('/history')
def history():
    """タスクの履歴をRedisから取得して表示する"""
    tasks_for_template = []
    if redis_client:
        # Redisからすべてのタスク情報を取得
        all_tasks_json = redis_client.hgetall('task_history')
        
        # 時刻でソートするためにリストに変換
        sorted_tasks = sorted(all_tasks_json.items(), key=lambda item: json.loads(item[1])['upload_time'], reverse=True)
        
        for task_id, task_json in sorted_tasks:
            task_data = json.loads(task_json)
            tasks_for_template.append(task_data)

    return render_template('history.html', tasks=tasks_for_template)


@app.route('/check_status/<task_id>')
def check_task_status(task_id):
    """Celeryタスクの状態と、Redisに保存された詳細な進捗を取得する"""
    response = {'state': 'UNKNOWN', 'status': 'タスク情報が見つかりません。'}

    # まずCelery自体のタスク状態を確認
    task = run_analysis_task.AsyncResult(task_id)
    response['state'] = task.state

    # 次にRedisから詳細な進捗情報を取得
    if redis_client:
        progress_key = f"pipeline-progress:{task_id}"
        progress_data = redis_client.hgetall(progress_key)
        
        if progress_data:
            # Redisから取得したデータを整形
            status = progress_data.get("status")
            if status == "completed":
                response['state'] = 'SUCCESS'
                response['status'] = '完了'
                response['result'] = {'result_path': progress_data.get("result_path")}
            elif status == "error":
                response['state'] = 'FAILURE'
                response['status'] = 'エラーが発生しました'
                response['details'] = progress_data.get("error_message", "不明なエラーです。")
            else: # processing
                response['state'] = 'PROGRESS'
                response['status'] = '処理中...'
                response['details'] = {
                    'current_step': int(progress_data.get("current_step", 0)),
                    'total_steps': int(progress_data.get("total_steps", 0)),
                    'step_name': progress_data.get("step_name", "処理中...")
                }
        elif task.state == 'PENDING':
             response['status'] = '待機中...'
        elif task.state == 'FAILURE':
            response['status'] = 'タスクの起動に失敗しました。'
            response['details'] = str(task.info)
        else:
             # Redisにまだ情報がなく、タスクが実行中の場合
             response['state'] = 'PROGRESS'
             response['status'] = 'プロセスの初期化中...'
             response['details'] = {'step_name': '初期化中...'}

    # 履歴情報も更新
    if redis_client:
        task_info_json = redis_client.hget('task_history', task_id)
        if task_info_json:
            task_info = json.loads(task_info_json)
            if task_info.get('state') != response['state']:
                 task_info['state'] = response['state']
                 redis_client.hset('task_history', task_id, json.dumps(task_info))

    return jsonify(response)

@app.route('/download/<path:filename>')
def download_file(filename):
    """分析結果の動画をダウンロードさせる"""
    # セキュリティのため、ファイルパスを安全に解決
    directory = os.path.abspath(os.path.join(app.config['TENNIS_OUTPUT_FOLDER'], '06_rally_extract'))
    safe_path = os.path.join(directory, filename)
    if os.path.commonpath([safe_path, directory]) != directory:
        return "不正なリクエストです", 400
        
    return send_from_directory(directory, filename, as_attachment=True)
    
@app.route('/calibrate/<video_name>')
def calibrate(video_name):
    # uploadsフォルダから変換後のビデオを探す
    base_name = os.path.splitext(video_name)[0]
    converted_name = f"{base_name}_converted.mp4"
    video_path = Path(app.config['UPLOAD_FOLDER']) / converted_name
    
    # 最初のフレームを画像として抽出
    output_image_folder = Path('static/frames')
    output_image_folder.mkdir(exist_ok=True)
    output_image_path = output_image_folder / f"{video_path.stem}.jpg"
    
    if not output_image_path.exists():
        # ffmpegで最初のフレームを抽出
        cmd = ['ffmpeg', '-i', str(video_path), '-vframes', '1', '-q:v', '2', str(output_image_path)]
        subprocess.run(cmd)
        
    # 既存の座標データを取得
    calibration_dir = Path(app.config['TENNIS_OUTPUT_FOLDER']) / '00_calibration_data'
    coords_files = list(calibration_dir.glob(f"court_coords_{base_name}*.json"))
    latest_coords = None
    if coords_files:
        latest_file = max(coords_files, key=os.path.getctime)
        with open(latest_file, 'r') as f:
            latest_coords = json.load(f)

    return render_template('calibrate.html', 
                           video_name=video_name, 
                           image_path=str(output_image_path).replace('\\', '/'),
                           existing_coords=latest_coords)

@app.route('/save_coords', methods=['POST'])
def save_coords():
    data = request.get_json()
    video_name = data['video_name']
    coords = data['coords']
    
    base_name = os.path.splitext(video_name)[0]
    output_dir = Path(app.config['TENNIS_OUTPUT_FOLDER']) / '00_calibration_data'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"court_coords_{base_name}_{timestamp}.json"
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(coords, f, indent=4)
        
    return jsonify({'status': 'success', 'message': f'座標を {filename} として保存しました。'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)