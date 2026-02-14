from flask import Flask, render_template, request, url_for, redirect, flash, Response, jsonify
import os
import secrets
import cv2
import shutil
import time
import threading
import numpy as np
from deepface import DeepFace
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
FRAMES_FOLDER = os.path.join(BASE_DIR, 'static', 'frames')
CCTV_FOLDER = os.path.join(BASE_DIR, 'cctv_videos')
# Standard allowed extensions
ALLOWED_VIDEO = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(CCTV_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- GLOBAL STATE ---
outputFrame = None
lock = threading.Lock()

scanning_state = {
    'active': False,
    'status': 'idle',
    'queue': [],
    'current_video': None,
    'match_found': False,
    'match_data': None,
    'target_embedding': None
}

# --- HELPER: Sanitize ---
def sanitize_filenames():
    """Ensure all videos have safe ASCII names."""
    if not os.path.exists(CCTV_FOLDER): return
    for i, f in enumerate(os.listdir(CCTV_FOLDER)):
        if f.lower().endswith(ALLOWED_VIDEO):
            # Check for non-ascii or spaces
            if any(ord(c) > 127 for c in f) or ' ' in f:
                ext = os.path.splitext(f)[1]
                safe_name = f"cam_footage_{int(time.time())}_{i}{ext}"
                try:
                    os.rename(os.path.join(CCTV_FOLDER, f), os.path.join(CCTV_FOLDER, safe_name))
                    print(f"Renamed: {f} -> {safe_name}")
                except:
                    pass

def open_video_capture(path):
    """
    Robust video opener attempting multiple backends.
    """
    backends = [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF]
    for backend in backends:
        cap = cv2.VideoCapture(path, backend)
        if cap.isOpened():
            # Verify we can read a frame
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Reset
                print(f"Opened {os.path.basename(path)} with backend: {backend}")
                return cap
            cap.release()
    return None

# --- BACKGROUND THREAD: The Scanner ---
def scanner_thread_func(target_path):
    global outputFrame, scanning_state
    
    print("--- SCANNER THREAD STARTED ---")
    
    # 1. Populate Queue IMMEDIATELY
    sanitize_filenames()
    target_videos = []
    
    try:
        candidates = [f for f in os.listdir(CCTV_FOLDER) if f.lower().endswith(ALLOWED_VIDEO)]
        # Validate using Robust Opener
        for f in candidates:
            vpath = os.path.join(CCTV_FOLDER, f)
            c = open_video_capture(vpath)
            if c:
                target_videos.append(f)
                c.release()
            else:
                print(f"SKIPPING BROKEN VIDEO: {f} (All Backends Failed)")
    except Exception as e:
        print(f"Validation Error: {e}")
        target_videos = []
        
    scanning_state['queue'] = target_videos
    scanning_state['status'] = 'initializing'
    
    print(f"Queue populated: {len(target_videos)} VALID videos.")

    # 2. Calc Target Embedding
    try:
        # Switching to ArcFace (SOTA Accuracy) + OpenCV (Speed/Stability)
        print("Calculating Target Embedding with ArcFace...")
        embeds = DeepFace.represent(
            img_path=target_path,
            model_name="Facenet512", 
            detector_backend="opencv",
            enforce_detection=False
        )
        scanning_state['target_embedding'] = embeds[0]["embedding"]
        print("Target Embedding Ready.")
    except Exception as e:
        print(f"Error initializing AI: {e}")
        scanning_state['status'] = 'error'
        return

    scanning_state['status'] = 'scanning'
    
    videos = scanning_state['queue']

    for i, video_file in enumerate(videos):
        if scanning_state['match_found']: 
            print("Match previously found. Stopping loop.")
            break

        print(f"--- STARTING VIDEO {i+1}/{len(videos)}: {video_file} ---")
        scanning_state['current_video'] = video_file
        scanning_state['current_video_idx'] = i 
        
        # Temp Copy Strategy to fix path/permission issues
        temp_scan_path = os.path.join(CCTV_FOLDER, f"temp_scan_{i}.mp4")
        try:
            shutil.copy(os.path.join(CCTV_FOLDER, video_file), temp_scan_path)
            path = temp_scan_path
        except:
            path = os.path.join(CCTV_FOLDER, video_file)

        print(f"DEBUG: Scanning via temp path: {path}")
        cap = open_video_capture(path)
            
        if not cap:
            print(f"Failed to open {video_file} with any backend.")
            if os.path.exists(temp_scan_path): 
                try: os.remove(temp_scan_path)
                except: pass
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Opened {video_file}: {total_frames} frames @ {fps} FPS")

        frame_count = 0
        try:
            while cap.isOpened() and not scanning_state['match_found']:
                ret, frame = cap.read()
                if not ret:
                    print(f"End of video {video_file} reached.")
                    break
                
                frame_count += 1
                
                # Smart Resize - Good balance for crowds
                h, w = frame.shape[:2]
                target_w = 800 # Reduced from 1280 to 800 for RetinaFace Performance
                if w > target_w:
                    r = target_w / float(w)
                    dim = (target_w, int(h * r))
                    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

                # Smart Resize - Good for Speed
                h, w = frame.shape[:2]
                target_w = 800 # 800px for YuNet/SSD
                if w > target_w:
                    r = target_w / float(w)
                    dim = (target_w, int(h * r))
                    frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

                # Process every 5th frame
                if frame_count % 5 == 0:
                    start_time = time.time()
                    # print(f"DEBUG: Scanning frame {frame_count}")
                    try:
                        # try YuNet (Fastest DNN) or Fallback to SSD
                        # detector_backend='ssd' is acceptable.
                        
                        face_objs = DeepFace.extract_faces(
                            img_path=frame,
                            detector_backend='ssd', # Stick to SSD but OPTIMIZED
                            enforce_detection=False,
                            align=False
                        )
                        
                        num_faces = len(face_objs)
                        if num_faces > 0:
                            print(f"DEBUG: SSD Found {num_faces} faces ({time.time()-start_time:.2f}s)")
                        else:
                            print(f"DEBUG: SSD found 0 faces")

                        for obj in face_objs:
                            # facial_area keys: x, y, w, h
                            area = obj['facial_area']
                            x, y, w, h = area['x'], area['y'], area['w'], area['h']
                            
                            # Draw Blue Box for ALL Detected Faces
                            # BLUE Box = Detection
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                            
                            face_crop = frame[y:y+h, x:x+w]
                            
                            if face_crop.size > 0:
                                # Switch to Facenet512 (State of the Art Open Source)
                                emb_result = DeepFace.represent(
                                    img_path=face_crop,
                                    model_name="Facenet512",
                                    detector_backend="skip",
                                    enforce_detection=False
                                )
                                curr_emb = emb_result[0]["embedding"]
                                
                                # Cosine Distance
                                t = np.array(scanning_state['target_embedding'])
                                c = np.array(curr_emb)
                                
                                dot = np.dot(t, c)
                                norm_t = np.linalg.norm(t)
                                norm_c = np.linalg.norm(c)
                                
                                if norm_t > 0 and norm_c > 0:
                                    distance = 1 - (dot / (norm_t * norm_c)) 
                                else:
                                    distance = 2.0
                                
                                # Log distance
                                print(f"DEBUG: Face at ({x},{y}) Facenet Cosine: {distance:.2f}")

                                # Show Score and Distance
                                cv2.putText(frame, f"{distance:.2f}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                                cv2.putText(frame, f"Video: {video_file}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

                                # Facenet512 Threshold (Standard: 0.30)
                                # Using 0.40 (Relaxed)
                                if distance < 0.40:
                                    print(f"DEBUG: MATCH FOUND! Dist: {distance:.2f}")
                                    
                                    # Confidence: 0.0 -> 100%, 0.90 -> 50%
                                    confidence = max(0, min(100, (1.0 - distance) * 100))
                                    
                                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 4)
                                    cv2.putText(frame, f"MATCH! ({int(confidence)}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                                    
                                    match_file = f"match_{int(time.time())}.jpg"
                                    # Use outputFrame logic later, but save frame now
                                    try:
                                        cv2.imwrite(os.path.join(FRAMES_FOLDER, match_file), frame)
                                    except:
                                        pass

                                    # UPDATE STATE AT THE END TO PREVENT RACE CONDITION
                                    scanning_state['match_data'] = {
                                        'video_file': video_file,
                                        'similarity': f"{int(confidence)}%",
                                        'frame_path': f"static/frames/{match_file}"
                                    }
                                    
                                    # Signal User Interface NOW
                                    scanning_state['status'] = 'match_found'
                                    scanning_state['match_found'] = True
                                    
                                    time.sleep(5)
                                    break # Break Face Loop
                    except Exception as e:
                        print(f"DETECTOR ERROR: {e}")
                        pass # Ignore frame errors

                # Update Global Output Frame inside loop
                with lock:
                    outputFrame = frame.copy()
                
                time.sleep(0.01)

            cap.release()
            
        except Exception as e:
            print(f"Error processing video {video_file}: {e}")
            continue
        finally:
            # Clean up temp file
            if os.path.exists(temp_scan_path):
                try: os.remove(temp_scan_path)
                except: pass
    
    if not scanning_state['match_found']:
        scanning_state['status'] = 'completed'
        print("Scanning completed. No match found.")

# --- ROUTES ---

@app.route('/')
def home():
    sanitize_filenames()
    vids = []
    try:
        all_files = os.listdir(CCTV_FOLDER)
        print(f"DEBUG: All files in {CCTV_FOLDER}: {all_files}")
        vids = [f for f in all_files if f.lower().endswith(ALLOWED_VIDEO)]
        print(f"DEBUG: Filtered videos: {vids}")
    except Exception as e:
        print(f"DEBUG: Error listing videos: {e}")
        vids = []
    
    # Pass CCTV_FOLDER absolute path to template
    return render_template('index.html', video_count=len(vids), cctv_path=CCTV_FOLDER)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video_file' not in request.files:
        flash("No file part", "error")
        return redirect(url_for('home'))
        
    file = request.files['video_file']
    if file.filename == '':
        flash("No selected file", "error")
        return redirect(url_for('home'))
        
    if file and file.filename.lower().endswith(ALLOWED_VIDEO):
        filename = secure_filename(file.filename)
        # Add timestamp to avoid overwrites
        name, ext = os.path.splitext(filename)
        safe_name = f"{name}_{int(time.time())}{ext}"
        
        file.save(os.path.join(CCTV_FOLDER, safe_name))
        sanitize_filenames()
        flash("Video uploaded successfully!", "success")
        return redirect(url_for('home'))
    else:
        flash("Invalid file type. Allowed: MP4, AVI, MOV, MKV, WMV", "error")
        return redirect(url_for('home'))

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('child_photo')
    if not f: return redirect(url_for('home'))
    
    filename = f"target_{int(time.time())}.png"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(path)
    
    # Start Scanning Thread
    # Reset State
    global scanning_state
    scanning_state = {
        'active': True,
        'status': 'scanning',
        'queue': [],
        'current_video': None,
        'match_found': False,
        'match_data': None,
        'target_embedding': None,
        'target_filename': filename # Store for result page
    }
    
    t = threading.Thread(target=scanner_thread_func, args=(path,))
    t.daemon = True
    t.start()
    
    return redirect(url_for('scan_page'))

@app.route('/scan')
def scan_page():
    if not scanning_state['active'] and scanning_state['status'] == 'idle':
        return redirect(url_for('home'))
    return render_template('scanning.html', queue=scanning_state.get('queue', []))

@app.route('/video_feed')
def video_feed():
    # Simple Generator that yields the global outputFrame
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate():
    global outputFrame, lock
    
    # Initial Wait for frame
    for _ in range(50):
        if outputFrame is not None: break
        # Yield Loading Frame
        blank = np.zeros((480, 640, 3), np.uint8)
        cv2.putText(blank, "INITIALIZING AI...", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        _, buf = cv2.imencode('.jpg', blank)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
        time.sleep(0.1)

    while True:
        with lock:
            if outputFrame is None:
                continue
            
            # Encode
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag: continue
        
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + encodedImage.tobytes() + b'\r\n')
        time.sleep(0.02) # ~50 FPS max stream

@app.route('/check_status')
def check_status():
    return jsonify({
        'status': scanning_state['status'],
        'current_video': scanning_state['current_video'],
        'current_video_index': scanning_state.get('current_video_idx', 0)
    })

@app.route('/result')
def result():
    if not scanning_state['match_found']:
        # If accessing result page without match, show "No Match Found" state
        return render_template('result.html', match_found=False)
    
    data = scanning_state['match_data']
    # Build URL for target image
    target_fn = scanning_state.get('target_filename', 'default.png')
    target_url = url_for('static', filename='uploads/' + target_fn)
    
    return render_template('result.html', 
        match_found=True,
        video_name=data['video_file'],
        target_image_url=target_url,
        matched_frame_url=data.get('frame_path', ''),
        confidence_score=data.get('similarity', '0%').replace('%', ''),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S") # Add timestamp as requested
    )

if __name__ == '__main__':
    # Use threaded=True for Flask to handle multiple requests (video feed + status checks)
    app.run(debug=True, port=5000, threaded=True, use_reloader=False)
