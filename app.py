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
HISTORY_FILE = 'tasks_history.json' # ★★★ 履歴ファイル名を追加 ★★★
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'm4v', 'mkv'}

app = Flask(__name__)
# ... app.configは変更なし ...
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024 
app.secret_key = 'super secret key'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ★★★↓ここから履歴管理用の関数を追加↓★★★
def get_task_history():
    """タスク履歴をJSONファイルから読み込む"""
    if not os.path.exists(HISTORY_FILE):
        return []
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []

def save_task_to_history(task_id, original_filename):
    """タスク情報を履歴ファイルに保存する"""
    history = get_task_history()
    # 新しいタスクをリストの先頭に追加
    history.insert(0, {
        'task_id': task_id,
        'original_filename': original_filename,
        'timestamp': datetime.now().isoformat()
    })
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=4)

# ★★★↓トップページを履歴表示に変更↓★★★
@app.route('/')
def history_page():
    """タスクの履歴を一覧表示する"""
    tasks_with_status = []
    history = get_task_history()
    for task_info in history:
        task_result = run_analysis_task.AsyncResult(task_info['task_id'])
        task_info['state'] = task_result.state
        tasks_with_status.append(task_info)
    return render_template('history.html', tasks=tasks_with_status)

# ... allowed_file, convert_to_standard_mp4 は変更なし ...
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

# ★★★↓/upload後のリダイレクト先を履歴ページに変更↓★★★
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


# ... /calibrate, /save_coordinates は変更なし ...
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
        # frame = cv2.rotate(frame, cv2.ROTATE_180)
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
    video_name = data.get('video_name')
    points_list = data.get('points')
    if not video_name or not points_list or len(points_list) != 6:
        return {"status": "error", "message": "必要なデータが不足しています。"}, 400
    try:
        point_names = ["top_left_corner", "top_right_corner", "bottom_left_corner", "bottom_right_corner", "net_left_ground", "net_right_ground"]
        coordinates_dict = {name: [p['x'], p['y']] for name, p in zip(point_names, points_list)}
        video_stem = Path(video_name).stem.replace('_converted', '')
        output_dir = Path('tennis_pipeline_output') / '00_calibration_data'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"court_coords_{video_stem}_{timestamp}.json"
        save_path = output_dir / output_filename
        with open(save_path, 'w') as f:
            json.dump(coordinates_dict, f, indent=4)
        print(f"コート座標を正しい形式で保存しました: {save_path}")
        return {"status": "success", "redirect_url": url_for('start_analysis', video_name=video_name)}
    except Exception as e:
        print(f"座標の保存中にエラー: {e}")
        return {"status": "error", "message": "サーバー側で保存中にエラーが発生しました。"}, 500

# ★★★↓/start_analysis に履歴保存処理を追加し、リダイレクト先を履歴ページに変更↓★★★
@app.route('/start_analysis/<path:video_name>')
def start_analysis(video_name):
    original_filename = secure_filename(video_name)
    base_name, _ = os.path.splitext(original_filename)
    original_video_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    converted_mp4_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base_name}_converted.mp4")

    if not os.path.exists(converted_mp4_path):
        if not convert_to_standard_mp4(original_video_path, converted_mp4_path):
            flash("動画変換に失敗したため、分析を開始できません。")
            return redirect(url_for('history_page'))

    print(f"Celeryに分析タスクを依頼します: {original_filename}")
    task = run_analysis_task.delay(converted_mp4_path, original_filename)
    
    # ★★★ 履歴ファイルに今回のタスクを記録 ★★★
    save_task_to_history(task.id, original_filename)
    
    # ★★★ 履歴ページにリダイレクト ★★★
    return redirect(url_for('history_page'))


# ★★★↓/check_status は変更なし、/status と /result は不要になるので削除可能だが残しておく↓★★★
@app.route('/check_status/<task_id>')
def check_task_status(task_id):
    task = run_analysis_task.AsyncResult(task_id)
    if task.state == 'PENDING':
        response = {'state': task.state, 'status': '待機中...'}
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