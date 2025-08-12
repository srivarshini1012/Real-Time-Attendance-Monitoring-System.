import os
import cv2
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from sklearn.neighbors import KNeighborsClassifier
import joblib
from datetime import datetime
import csv
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Paths
UPLOAD_FOLDER = 'dataset'                # dataset/<name>/*.jpg
MODEL_DIR = 'trained_model'
CASCADE_PATH = 'haarcascade_frontalface_default.xml'
ATTENDANCE_CSV = 'attendance.csv'
THUMBS_DIR = 'recognized_thumbs'         # thumbnails saved here

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(THUMBS_DIR, exist_ok=True)

# Load cascade
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# helper: ensure CSV header
def ensure_csv():
    if not os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Date', 'Time'])

# helper: already marked check
def already_marked(name, date_str):
    if not os.path.exists(ATTENDANCE_CSV):
        return False
    with open(ATTENDANCE_CSV, 'r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) >= 2 and row[0] == name and row[1] == date_str:
                return True
    return False

def ensure_allowed_image_file(fname):
    return fname.lower().endswith(('.png', '.jpg', '.jpeg'))

# index
@app.route('/')
def index():
    return render_template('index.html')

# registration page
@app.route('/upload')
def upload_page():
    return render_template('upload.html')

# receive frames (exactly N frames expected) and save face crops
@app.route('/upload_frames', methods=['POST'])
def upload_frames():
    """
    Expects:
      - name (form)
      - frames[] (files)
    Saves detected/cropped faces (100x100 grayscale) into dataset/<name>/ as 1.jpg,2.jpg...
    """
    name_raw = request.form.get('name', '').strip()
    if not name_raw:
        return jsonify({'success': False, 'message': 'Missing name'}), 400
    name = secure_filename(name_raw)
    files = request.files.getlist('frames')
    if not files:
        return jsonify({'success': False, 'message': 'No frames'}), 400

    user_folder = os.path.join(UPLOAD_FOLDER, name)
    os.makedirs(user_folder, exist_ok=True)
    existing = len([f for f in os.listdir(user_folder) if ensure_allowed_image_file(f)])

    saved = 0
    for f in files:
        try:
            data = np.frombuffer(f.read(), np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            if len(faces) == 0:
                # fallback: store the resized full gray
                face_roi = cv2.resize(gray, (100,100))
            else:
                # largest face
                faces = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)
                x,y,w,h = faces[0]
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (100,100))
            save_path = os.path.join(user_folder, f"{existing + saved + 1}.jpg")
            cv2.imwrite(save_path, face_roi)
            saved += 1
        except Exception as e:
            continue

    if saved == 0:
        return jsonify({'success': False, 'message': 'No faces saved'}), 400
    return jsonify({'success': True, 'saved': saved})

# train model
@app.route('/train')
def train():
    X, y = [], []
    label_map = {}
    label_id = 0
    for user in sorted(os.listdir(UPLOAD_FOLDER)):
        user_folder = os.path.join(UPLOAD_FOLDER, user)
        if not os.path.isdir(user_folder):
            continue
        images = [f for f in os.listdir(user_folder) if ensure_allowed_image_file(f)]
        if not images:
            continue
        for img_name in images:
            img_path = os.path.join(user_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            try:
                face_flat = cv2.resize(img, (100,100)).flatten()
            except:
                continue
            X.append(face_flat)
            y.append(label_id)
        label_map[label_id] = user
        label_id += 1

    if not X:
        flash("No faces available. Register users first.", "error")
        return redirect(url_for('index'))

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X, y)
    joblib.dump(clf, os.path.join(MODEL_DIR, 'knn_model.pkl'))
    joblib.dump(label_map, os.path.join(MODEL_DIR, 'label_map.pkl'))
    flash("Model trained successfully.", "success")
    return redirect(url_for('index'))

# attendance page
@app.route('/attendance')
def attendance_page():
    return render_template('attendance.html')

# recognize a single frame (called by client). Returns names + bboxes and writes attendance & saves thumbs
@app.route('/recognize_frame', methods=['POST'])
def recognize_frame():
    """
    POST 'frame' (image). Returns JSON:
      { success: True, recognized: [ {name, bbox: [x,y,w,h]} , ... ], written: [names_written] }
    Coordinates are in pixel coords relative to the frame width/height.
    """
    if 'frame' not in request.files:
        return jsonify({'success': False, 'message': 'No frame'}), 400
    file = request.files['frame']
    data = np.frombuffer(file.read(), np.uint8)
    img_color = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img_color is None:
        return jsonify({'success': False, 'message': 'Invalid image'}), 400

    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    model_path = os.path.join(MODEL_DIR, 'knn_model.pkl')
    label_map_path = os.path.join(MODEL_DIR, 'label_map.pkl')
    if not os.path.exists(model_path) or not os.path.exists(label_map_path):
        return jsonify({'success': False, 'message': 'Model not trained'}), 400

    clf = joblib.load(model_path)
    label_map = joblib.load(label_map_path)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    recognized = []
    for (x,y,w,h) in faces:
        try:
            face_roi = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_roi, (100,100)).flatten().reshape(1,-1)
            pred = clf.predict(face_resized)[0]
            name = label_map.get(pred, "Unknown")
            recognized.append({'name': name, 'bbox': [int(x), int(y), int(w), int(h)]})
        except Exception:
            continue

    # write attendance and save thumbs
    ensure_csv()
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    time_str = now.strftime('%H:%M:%S')
    written = []
    for r in recognized:
        n = r['name']
        if n == "Unknown":
            continue
        if not already_marked(n, date_str):
            # write to CSV
            with open(ATTENDANCE_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([n, date_str, time_str])
            written.append(n)
            # save thumbnail
            x,y,w,h = r['bbox']
            # clamp coords
            h_img, w_img = gray.shape
            x = max(0, min(x, w_img-1))
            y = max(0, min(y, h_img-1))
            w = max(1, min(w, w_img-x))
            h = max(1, min(h, h_img-y))
            thumb = img_color[y:y+h, x:x+w]
            try:
                thumb = cv2.resize(thumb, (160,160))
            except:
                pass
            date_folder = os.path.join(THUMBS_DIR, date_str)
            os.makedirs(date_folder, exist_ok=True)
            ts = now.strftime('%H%M%S')
            safe_name = secure_filename(n)
            thumb_path = os.path.join(date_folder, f"{safe_name}_{ts}.jpg")
            cv2.imwrite(thumb_path, thumb)

    return jsonify({'success': True, 'recognized': recognized, 'written': written})

# admin & CSV download
@app.route('/admin')
def admin():
    rows = []
    if os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
    return render_template('admin.html', records=rows)


@app.route('/')
def home():
    return render_template('index.html')



@app.route('/download_csv')
def download_csv():
    if os.path.exists(ATTENDANCE_CSV):
        return send_file(os.path.join(os.getcwd(), ATTENDANCE_CSV), as_attachment=True)
    return "CSV not found", 404

if __name__ == '__main__':
    ensure_csv()
    app.run(debug=True)
