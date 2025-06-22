import os
import subprocess
import ffmpeg
import sys
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, flash, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path
import cv2
import base64
import json
from datetime import datetime
from tasks import run_analysis_task, celery

# --- 設定 ---
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'tennis_pipeline_output/06_rally_extract' 
HISTORY_FILE = 'tasks_history.json'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'm4v', 'mkv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024 * 1024 # 上限を4GBに設定
app.secret_key = 'super secret key'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def get_task_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_task_to_history(task_id, original_filename):
    history = get_task_history()
    history.insert(0, {
        'task_id': task_id,
        'original_filename': original_filename,
        'timestamp': datetime.now().isoformat()
    })
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)

@app.route('/')
def history_page():
    tasks_with_status = []
    history = get_task_history()
    for task_info in history:
        task_result = run_analysis_task.AsyncResult(task_info['task_id'])
        task_info['state'] = task_result.state
        tasks_with_status.append(task_info)
    return render_template('history.html', tasks=tasks_with_status)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_standard_mp4(source_path, output_path):
    print(f"動画を標準的なMP4形式に変換します: {source_path} -> {output_path}")
    try:
        (ffmpeg.input(source_path).output(output_path, vcodec='libx264', acodec='aac', pix_fmt='yuv420p').overwrite_output().run(capture_stdout=True, capture_stderr=True))
        print("変換が完了しました。")
        return True
    except ffmpeg.Error as e:
        print("ffmpegエラー:", e.stderr.decode())
        flash(f"動画の変換に失敗しました: {e.stderr.decode()}")
        return False

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        flash('ファイルが選択されていません')
        return redirect(url_for('history_page'))
    file = request.files['video']
    if file.filename == '':
        flash('ファイル名がありません')
        return redirect(url_for('history_page'))
    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        file.save(upload_path)
        print(f"動画をアップロードしました: {upload_path}")
        return redirect(url_for('calibrate_page', video_name=original_filename))
    else:
        flash('許可されていないファイル形式です')
        return redirect(url_for('history_page'))

@app.route('/calibrate/<path:video_name>')
def calibrate_page(video_name):
    video_path = Path(app.config['UPLOAD_FOLDER']) / video_name
    if not video_path.exists():
        flash('指定された動画が見つかりません。')
        return redirect(url_for('history_page'))
    try:
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame_index = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            flash('動画フレームの読み込みに失敗しました。')
            return redirect(url_for('history_page'))
        
        # ★★★↓ここのコメントアウト(#)を削除して、回転処理を有効化↓★★★
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        height, width, _ = frame.shape
        return render_template('calibrate.html', frame_data=frame_b64, video_name=video_name, original_width=width, original_height=height)
    except Exception as e:
        print(f"キャリブレーションページの生成中にエラー: {e}")
        flash('ページの生成中にエラーが発生しました。')
        return redirect(url_for('history_page'))

@app.route('/save_coordinates', methods=['POST'])
def save_coordinates():
    data = request.get_json()
    video_name = data['video_name']
    coordinates = data['coordinates']
    
    # ファイル名の生成
    base_name = os.path.splitext(video_name)[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"court_coords_{base_name}_{timestamp}.json"
    filepath = os.path.join('tennis_pipeline_output', '00_calibration_data', filename)
    
    # キー名形式で保存（メタデータ付き）
    coordinates_data = {
        "top_left_corner": coordinates['top_left_corner'],
        "top_right_corner": coordinates['top_right_corner'],
        "bottom_left_corner": coordinates['bottom_left_corner'],
        "bottom_right_corner": coordinates['bottom_right_corner'],
        "net_left_ground": coordinates['net_left_ground'],
        "net_right_ground": coordinates['net_right_ground'],
        "_metadata": {
            "creation_time": datetime.now().isoformat(),
            "video_name": video_name,
            "coordinate_count": 6,
            "point_names": [
                "top_left_corner",
                "top_right_corner", 
                "bottom_left_corner",
                "bottom_right_corner",
                "net_left_ground",
                "net_right_ground"
            ]
        }
    }
    
    # ディレクトリ作成
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # ファイル保存
    with open(filepath, 'w') as f:
        json.dump(coordinates_data, f, indent=2)
    
    print(f"コート座標を保存しました（キー名形式）: {filepath}")
    
    return jsonify({
        'status': 'success',
        'redirect_url': url_for('start_analysis', video_name=video_name)
    })

@app.route('/start_analysis/<path:video_name>')
def start_analysis(video_name):
    """分析タスク（動画変換を含む）をCeleryに依頼する"""
    original_filename = secure_filename(video_name)
    original_video_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)

    if not os.path.exists(original_video_path):
        flash("分析対象の動画ファイルが見つかりません。")
        return redirect(url_for('history_page'))

    print(f"Celeryに分析タスク（変換込み）を依頼します: {original_filename}")
    
    # ★★★ タスクに渡す引数を、変換前の元の動画パスに変更 ★★★
    task = run_analysis_task.delay(original_video_path, original_filename)
    
    save_task_to_history(task.id, original_filename)
    
    return redirect(url_for('history_page'))


@app.route('/check_status/<task_id>')
def check_task_status(task_id):
    task = run_analysis_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {'state': task.state, 'status': '待機中...'}
    elif task.state == 'PROGRESS': # ★★★ 進行中状態を追加
        response = {'state': task.state, 'status': '処理中...', 'meta': task.info}
    elif task.state != 'FAILURE':
        response = {'state': task.state, 'status': '処理中...'}
        if task.state == 'SUCCESS':
            response = {'state': task.state, 'status': '完了', 'result': task.info}
    else:
        response = {'state': task.state, 'status': str(task.info)}
    return jsonify(response)
    
@app.route('/result/<filename>')
def show_result(filename):
    return render_template('result.html', filename=filename)

@app.route('/download/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)