"""
GrievX Integrated Flask App
Supports both text and video-based complaint inputs
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# Configuration
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "GrievX_UI_and_model" / "artifacts"
MODEL_FILE = ARTIFACT_DIR / "grievx_resolution_model.joblib"
METADATA_FILE = ARTIFACT_DIR / "grievx_metadata.json"
WARD_FILE = BASE_DIR / "GrievX_UI_and_model" / "Final_MLA_and_Ward_member_Dataset.csv"
YOLO_MODEL_PATH = BASE_DIR / "runs" / "grievx_yolov8s_cpu" / "weights" / "best.pt"

UPLOAD_FOLDER = BASE_DIR / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

HISTORY_DIR = BASE_DIR / "history"
HISTORY_DIR.mkdir(exist_ok=True)
HISTORY_FILE = HISTORY_DIR / "grievance_history.jsonl"

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# YOLOv8 Class names
CLASS_NAMES = {
    0: "Crop_Damage",
    1: "Drainage_issue",
    2: "Drinking_water_shortage",
    3: "Industrial_Pollution_complaint",
    4: "Road_Damage"
}

# Map YOLO classes to text-based system categories
CATEGORY_MAPPING = {
    "Crop_Damage": "agriculture",
    "Drainage_issue": "rural development",
    "Drinking_water_shortage": "water",
    "Industrial_Pollution_complaint": "industries",
    "Road_Damage": "roads"
}

# Map YOLO classes to problem names
PROBLEM_NAME_MAPPING = {
    "Crop_Damage": "Crop damage complaint",
    "Drainage_issue": "Village drainage issue",
    "Drinking_water_shortage": "Drinking water shortage",
    "Industrial_Pollution_complaint": "Industrial pollution complaint",
    "Road_Damage": "Road damage complaint"
}

# Problem catalog for text input
PROBLEM_CATALOG = {
    "agriculture": ["Crop damage complaint"],
    "industries": ["Industrial pollution complaint"],
    "roads": ["Road damage complaint"],
    "rural development": ["Village drainage issue"],
    "water": ["Drinking water shortage"],
}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Load models at startup
print("Loading models...")
text_model = joblib.load(MODEL_FILE)
metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
ward_df = pd.read_csv(WARD_FILE)
yolo_model = YOLO(str(YOLO_MODEL_PATH))
print("Models loaded successfully!")


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return Path(filename).suffix.lower() in ALLOWED_VIDEO_EXTENSIONS


def extract_frames(video_path: Path, num_frames: int = 10) -> list:
    """Extract frames from video for prediction."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []
    
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames


def predict_from_video(video_path: Path, conf_threshold: float = 0.3) -> tuple[str, str, float] | None:
    """Predict problem category from video using YOLOv8."""
    frames = extract_frames(video_path, num_frames=10)
    if not frames:
        return None
    
    all_predictions = []
    for frame in frames:
        results = yolo_model.predict(frame, conf=conf_threshold, verbose=False)
        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                all_predictions.append((class_id, confidence))
    
    if not all_predictions:
        return None
    
    # Aggregate predictions (majority voting)
    class_counts = {}
    for class_id, conf in all_predictions:
        if class_id not in class_counts:
            class_counts[class_id] = []
        class_counts[class_id].append(conf)
    
    # Get class with highest count
    best_class = max(class_counts.items(), key=lambda x: len(x[1]))[0]
    avg_confidence = np.mean(class_counts[best_class])
    
    # Map to system values
    yolo_class_name = CLASS_NAMES[best_class]
    problem_category = CATEGORY_MAPPING[yolo_class_name]
    problem_name = PROBLEM_NAME_MAPPING[yolo_class_name]
    
    return problem_category, problem_name, avg_confidence


def predict_days(model, complaint_name: str, complaint_category: str) -> float:
    """Predict resolution days using text-based model."""
    sample = pd.DataFrame(
        [
            {
                "Problem_Name": complaint_name,
                "Problem_Category": complaint_category,
            }
        ]
    )
    return float(model.predict(sample)[0])


def normalize_key(value: str) -> str:
    """Normalize string for comparison."""
    return value.strip().lower()


def clean_value(value: Any) -> str:
    """Clean and convert value to string."""
    if pd.isna(value):
        return "Unknown"
    return str(value).strip()


def build_chain(metadata: dict, ward_row: pd.Series, complaint_category: str) -> list[dict]:
    """Build escalation chain."""
    category_lookup = metadata.get("minister_lookup", {})
    minister_info = category_lookup.get(normalize_key(complaint_category), {})

    ward_member = clean_value(ward_row.get("Ward_Member_Name"))
    mla_name = clean_value(ward_row.get("winning_cand"))
    chief_minister = clean_value(ward_row.get("Chief Minister"))
    governor = clean_value(ward_row.get("Governor"))

    return [
        {"stage": "Ward Member", "name": ward_member},
        {"stage": "MLA", "name": mla_name},
        {"stage": "Minister", "name": minister_info.get("Minister_Name", "Unknown"), "ministry": minister_info.get("Ministry", "Unknown")},
        {"stage": "Chief Minister", "name": chief_minister},
        {"stage": "Governor", "name": governor},
    ]


def unique_sorted(series: pd.Series) -> list[str]:
    """Get unique sorted values from series."""
    values = [clean_value(value) for value in series.dropna().tolist()]
    return sorted(dict.fromkeys(values))


def save_history_record(record: dict) -> None:
    """Save a complaint record to history."""
    cleaned_record = {}
    for k, v in record.items():
        if isinstance(v, float) and np.isnan(v):
            cleaned_record[k] = None
        else:
            cleaned_record[k] = v
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(cleaned_record) + "\n")


def load_history() -> list[dict]:
    """Load complaint history from file."""
    if not HISTORY_FILE.exists():
        return []
    
    records = []
    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    cleaned_record = {}
                    for k, v in record.items():
                        if isinstance(v, float) and np.isnan(v):
                            cleaned_record[k] = None
                        else:
                            cleaned_record[k] = v
                    records.append(cleaned_record)
                except Exception:
                    pass
    
    return records


def check_escalation_needed(history_data: Any, complaint_id: str, predicted_days: float) -> bool:
    """Check if complaint needs escalation based on predicted days."""
    if isinstance(history_data, pd.DataFrame):
        if history_data.empty:
            return False
        complaint_record = history_data[history_data.get('id', pd.Series(dtype=str)) == complaint_id]
        if complaint_record.empty:
            return False
        record = complaint_record.iloc[0]
    else:
        if not history_data:
            return False
        record = next((r for r in history_data if r.get('id') == complaint_id), None)
        if not record:
            return False
    
    timestamp_str = record.get('timestamp', '')
    if not timestamp_str:
        return False
    
    # Calculate days since submission
    try:
        submission_time = datetime.fromisoformat(timestamp_str)
        current_time = datetime.now()
        days_passed = (current_time - submission_time).days
        return days_passed > predicted_days
    except:
        return False


@app.route('/')
def index():
    """Render main page."""
    district_options = unique_sorted(ward_df["district"])
    category_options = list(PROBLEM_CATALOG.keys())
    return render_template('index.html', 
                         districts=district_options,
                         categories=category_options)


@app.route('/get_areas/<district>')
def get_areas(district):
    """Get areas for selected district."""
    district_filtered = ward_df[ward_df["district"].astype(str).str.strip().str.lower() == normalize_key(district)]
    area_options = unique_sorted(district_filtered["ac_name"])
    return jsonify(area_options)


@app.route('/get_wards/<district>/<area>')
def get_wards(district, area):
    """Get wards for selected area."""
    district_filtered = ward_df[ward_df["district"].astype(str).str.strip().str.lower() == normalize_key(district)]
    area_filtered = district_filtered[district_filtered["ac_name"].astype(str).str.strip().str.lower() == normalize_key(area)]
    ward_options = unique_sorted(area_filtered["Ward_Name"])
    return jsonify(ward_options)


@app.route('/get_problems/<category>')
def get_problems(category):
    """Get problem names for selected category."""
    problems = PROBLEM_CATALOG.get(category, [])
    return jsonify(problems)


@app.route('/predict_text', methods=['POST'])
def predict_text():
    """Handle text-based complaint prediction."""
    try:
        data = request.json
        
        district = data.get('district')
        area = data.get('area')
        ward_name = data.get('ward')
        complaint_category = data.get('category')
        complaint_name = data.get('problem_name')
        complaint_text = data.get('details', '')
        
        # Get ward row
        district_filtered = ward_df[ward_df["district"].astype(str).str.strip().str.lower() == normalize_key(district)]
        area_filtered = district_filtered[district_filtered["ac_name"].astype(str).str.strip().str.lower() == normalize_key(area)]
        ward_filtered = area_filtered[area_filtered["Ward_Name"].astype(str).str.strip().str.lower() == normalize_key(ward_name)]
        ward_row = ward_filtered.iloc[0] if not ward_filtered.empty else area_filtered.iloc[0]
        
        # Predict days
        predicted_days = predict_days(text_model, complaint_name, complaint_category)
        
        # Build escalation chain
        chain = build_chain(metadata, ward_row, complaint_category)
        
        # Generate complaint ID
        complaint_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to history
        record = {
            "id": complaint_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "problem_name": complaint_name,
            "category": complaint_category,
            "district": district,
            "area": area,
            "ward_name": ward_name,
            "predicted_days": predicted_days,
            "status": "open",
            "current_stage": chain[0]['stage'],
            "current_assignee": chain[0]['name'],
            "details": complaint_text,
            "from_video": False,
            "chain": chain
        }
        save_history_record(record)
        
        return jsonify({
            'success': True,
            'predicted_days': predicted_days,
            'chain': chain,
            'district': district,
            'area': area,
            'ward': ward_name,
            'category': complaint_category,
            'problem_name': complaint_name,
            'details': complaint_text,
            'complaint_id': complaint_id
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/predict_video', methods=['POST'])
def predict_video():
    """Handle video-based complaint prediction."""
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file provided'})
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv, webm, m4v'})
        
        # Save uploaded video
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{filename}"
        video_path = UPLOAD_FOLDER / safe_filename
        file.save(str(video_path))
        
        # Predict from video
        prediction_result = predict_from_video(video_path)
        
        if prediction_result is None:
            return jsonify({'success': False, 'error': 'Could not detect any issue from video. Please try a clearer video.'})
        
        problem_category, problem_name, confidence = prediction_result
        
        # Return prediction result for user to select location
        return jsonify({
            'success': True,
            'problem_category': problem_category,
            'problem_name': problem_name,
            'confidence': confidence,
            'video_filename': safe_filename
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/complete_video_prediction', methods=['POST'])
def complete_video_prediction():
    """Complete video prediction with location info."""
    try:
        data = request.json
        
        district = data.get('district')
        area = data.get('area')
        ward_name = data.get('ward')
        problem_category = data.get('category')
        problem_name = data.get('problem_name')
        complaint_text = data.get('details', '')
        video_filename = data.get('video_filename', '')
        
        # Get ward row
        district_filtered = ward_df[ward_df["district"].astype(str).str.strip().str.lower() == normalize_key(district)]
        area_filtered = district_filtered[district_filtered["ac_name"].astype(str).str.strip().str.lower() == normalize_key(area)]
        ward_filtered = area_filtered[area_filtered["Ward_Name"].astype(str).str.strip().str.lower() == normalize_key(ward_name)]
        ward_row = ward_filtered.iloc[0] if not ward_filtered.empty else area_filtered.iloc[0]
        
        # Predict days
        predicted_days = predict_days(text_model, problem_name, problem_category)
        
        # Build escalation chain
        chain = build_chain(metadata, ward_row, problem_category)
        
        # Generate complaint ID
        complaint_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to history
        record = {
            "id": complaint_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "problem_name": problem_name,
            "category": problem_category,
            "district": district,
            "area": area,
            "ward_name": ward_name,
            "predicted_days": predicted_days,
            "status": "open",
            "current_stage": chain[0]['stage'],
            "current_assignee": chain[0]['name'],
            "details": complaint_text,
            "from_video": True,
            "video_filename": video_filename,
            "chain": chain
        }
        save_history_record(record)
        
        return jsonify({
            'success': True,
            'predicted_days': predicted_days,
            'chain': chain,
            'district': district,
            'area': area,
            'ward': ward_name,
            'category': problem_category,
            'problem_name': problem_name,
            'details': complaint_text,
            'from_video': True,
            'complaint_id': complaint_id
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/update_complaint_status', methods=['POST'])
def update_complaint_status():
    """Update complaint status (solved or escalated)."""
    try:
        data = request.json
        complaint_id = data.get('complaint_id')
        action = data.get('action')  # 'solved' or 'escalate'
        
        records = load_history()
        if not records:
            return jsonify({'success': False, 'error': 'No history found'})
        
        # Find the complaint
        record_idx = None
        for i, rec in enumerate(records):
            if rec.get('id') == complaint_id:
                record_idx = i
                break
        
        if record_idx is None:
            return jsonify({'success': False, 'error': 'Complaint not found'})
        
        record = records[record_idx]
        
        if action == 'solved':
            record['status'] = 'closed'
            record['resolution_time'] = datetime.now().isoformat(timespec="seconds")
        elif action == 'escalate':
            # Get chain
            chain = record.get('chain', [])
            if not chain:
                # Reconstruct chain
                district = record.get('district')
                area = record.get('area')
                ward_name = record.get('ward_name')
                complaint_category = record.get('category')
                
                # Retrieve from ward_df
                district_filtered = ward_df[ward_df["district"].astype(str).str.strip().str.lower() == normalize_key(district)]
                area_filtered = district_filtered[district_filtered["ac_name"].astype(str).str.strip().str.lower() == normalize_key(area)]
                ward_filtered = area_filtered[area_filtered["Ward_Name"].astype(str).str.strip().str.lower() == normalize_key(ward_name)]
                ward_row = ward_filtered.iloc[0] if not ward_filtered.empty else (area_filtered.iloc[0] if not area_filtered.empty else pd.Series())
                
                chain = build_chain(metadata, ward_row, complaint_category)
                record['chain'] = chain
                
            current_stage = record.get('current_stage', 'Ward Member')
            
            # Find next stage
            stage_order = ['Ward Member', 'MLA', 'Minister', 'Chief Minister', 'Governor']
            current_idx = stage_order.index(current_stage) if current_stage in stage_order else 0
            next_idx = min(current_idx + 1, len(stage_order) - 1)
            
            record['current_stage'] = stage_order[next_idx]
            if next_idx < len(chain):
                record['current_assignee'] = chain[next_idx]['name']
            record['status'] = 'escalated'
            record['escalation_time'] = datetime.now().isoformat(timespec="seconds")
        
        # Save all records back
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            for rec in records:
                cleaned_rec = {}
                for k, v in rec.items():
                    if isinstance(v, float) and np.isnan(v):
                        cleaned_rec[k] = None
                    else:
                        cleaned_rec[k] = v
                f.write(json.dumps(cleaned_rec) + "\n")
        
        return jsonify({'success': True, 'updated_record': record})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/get_history')
def get_history():
    """Get complaint history."""
    try:
        records = load_history()
        return jsonify({'success': True, 'history': records})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
