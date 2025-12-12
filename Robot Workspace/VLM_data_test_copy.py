import cv2
import sys
import os
import json
import time
import threading
import numpy as np
import pyrealsense2 as rs
from flask import Flask, Response, request, render_template_string
import requests 
from ultralytics import YOLO 

app = Flask(__name__)
CONTROL_SERVICE_URL = "http://localhost:5051/action"

# === CONFIG ===
DATASET_DIR = "dataset"
ACTION_MAP = {
    'w': ("FORWARD",     "forward"),
    's': ("BACKWARD",    "backward"),
    'a': ("LEFT",        "turn_left"),
    'd': ("RIGHT",       "turn_right"),
    'u': ("HEART",       "heart"),
    # 'i': ("SIT",         "sit_down"),
    # 'o': ("STAND",       "get_up"),
    "p": ("HAND SHAKE",  "hand_shake")
}

# === GLOBAL STATE ===
pipeline = None
align = None
yolo_model = None
dataset_log = []
frame_idx = 0
latest_color = None
latest_depth = None
img_lock = threading.Lock()

def init_camera():
    global align 
    try:
        """Setup RealSense"""
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30) 
        pipeline.start(config)

        align = rs.align(rs.stream.color)     # align object to map depth pixels to color pixels (they're roughly 1 in apart, so this is necessary)

        print("Camera Connected (Color + Depth)")
        return pipeline
    except Exception as e:
        print(f"Camera Failed: {e}")
        return None
    
def load_existing_dataset():
    """Loads existing log.json so we don't overwrite previous data"""
    global dataset_log, frame_idx
    log_path = os.path.join(DATASET_DIR, "log.json")
    
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                dataset_log = json.load(f)
            
            # Find the highest ID to continue numbering correctly
            if dataset_log:
                max_id = max(item.get("id", 0) for item in dataset_log)
                frame_idx = max_id + 1
                print(f"Found existing dataset with {len(dataset_log)} entries.")
                print(f"Resuming at Frame Index: {frame_idx}")

        except Exception as e:
            print(f"Error reading existing log: {e}. Starting fresh.")

def save_data(color_img, depth_img, action_name, idx, detections, target_dist):
    """Saves the image and returns the metadata entry"""

    # Color JPG
    color_filename = f"{idx:04d}_{action_name}_color.jpg"
    color_filepath = os.path.join(DATASET_DIR, color_filename)
    cv2.imwrite(color_filepath, color_img)

    # Depth JPG 
    depth_filename = f"{idx:04d}_{action_name}_depth.png"
    depth_filepath = os.path.join(DATASET_DIR, depth_filename)
    cv2.imwrite(depth_filepath, depth_img)

    return {
            "id": idx,
            "action": action_name, 
            "color_path": color_filepath, 
            "depth_path": depth_filepath, 
            "yolo_detections": detections,           # List of boxes [x1, y1, x2, y2]
            "person_present": len(detections) > 0,   # Boolean flag 
            "person_distance_m": target_dist         # Explicit distance for the VLM  
        }

# === FLASK ROUTES ===
def generate_frames():
    """Background loop to capture frames from RealSense"""
    global latest_color, latest_depth
    while True:
        try:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not color_frame or not depth_frame: continue
            
            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())
                        
            with img_lock:
                latest_color = color_img.copy()
                latest_depth  = depth_img.copy()

            _, buffer = cv2.imencode('.jpg', color_img)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Generate Frames Error: {e}")
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template_string("""
    <html>
    <head>
        <title>Smart Recorder</title>
        <style>
            body { font-family: monospace; text-align: center; background: #222; color: #fff; }
            img { width: 80%; border: 3px solid #444; border-radius: 10px; }
            .controls { background: #333; padding: 20px; display: inline-block; border-radius: 10px; margin-top: 10px;}
            .key { display: inline-block; padding: 10px 20px; margin: 5px; background: #555; border-radius: 5px; }
            .busy { color: #ff0; }
        </style>
    </head>
    <body>
        <h1>📹 AI Data Recorder</h1>
        <img src="/video_feed" />
        <div class="controls">
            <div class="keys">
                <span class="key">W</span><span class="key">A</span><span class="key">S</span><span class="key">D</span> Move <br>
                <span class="key">I</span> Sit <span class="key">U</span> Stand <br>
            </div>
            <br>
            <div>Press <b>Q</b> to Save & Quit</div>
            <div id="status">Ready...</div>
        </div>
        <script>
            let isProcessing = false;
            document.addEventListener('keydown', function(event) {
                if (event.repeat || isProcessing) return;
                const key = event.key.toLowerCase();
                const validKeys = ['w','a','s','d','u','p','q'];
                if (validKeys.includes(key)) {
                    isProcessing = true;
                    const st = document.getElementById('status');
                    st.innerText = "Processing...";
                    st.classList.add("busy");
                    fetch('/control?key=' + key)
                        .then(r => r.text()).then(t => { 
                            st.innerText = t; st.classList.remove("busy");
                            if (key === 'q') alert("Saved!");
                        })
                        .finally(() => { isProcessing = false; });
                }
            });
        </script>
    </body>
    </html>
    """)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/control')
def control():
    global frame_idx
    key = request.args.get('key')
    
    # Handle Quit
    if key == 'q':
        print("Saving dataset...")
        with open(os.path.join(DATASET_DIR, "log.json"), "w") as f:
            json.dump(dataset_log, f, indent=4)
        return "Dataset Saved. Exit via Ctrl+C in terminal."

    # Handle Movement & Recording
    if key in ACTION_MAP:
        log_label, api_route = ACTION_MAP[key]
        print(f"Action: {log_label}")

        # Capture Both Frames 
        curr_color = None
        curr_depth = None

        with img_lock:
            if latest_color is not None and latest_depth is not None:
                curr_color = latest_color.copy()
                curr_depth = latest_depth.copy()
        
        if curr_color is None: return "Error: No Camera Frame"

        # Run YOLO 
        results = yolo_model(curr_color, verbose=False, conf=0.5)

        person_detections = []
        person_detected = False
        min_dist_m = -1.0 # default if no person detected 

        for r in results:
            for box in r.boxes:
                if (int(box.cls[0]) == 0):
                    person_detected = True 
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
                    person_detections.append([x1, y1, x2, y2])
                    
                    # center of the box 
                    cx, cy = int((x1+x2)/2), int((y1+y2)/2)

                    # ensure coords are within frame
                    h, w = curr_depth.shape
                    cx = max(0, min(cx, w-1))
                    cy = max(0, min(cy, h-1))

                    # read depth at center (mm)
                    dist_mm = curr_depth[cy, cx]

                    # if 0 (invalid), try a small area average 
                    if dist_mm == 0:
                        roi = curr_depth[max(0, cy-5):min(h, cy+5), max(0, cx-5):min(w, cx+5)]
                        valid = roi[roi > 0]
                        if len(valid) > 0: dist_mm = np.mean(valid)

                    dist_m = dist_mm / 1000.0
                    
                    # Keep the closest person's distance
                    if min_dist_m < 0 or (dist_m > 0 and dist_m < min_dist_m):
                        min_dist_m = dist_m

        # Send Command 
        try:
            requests.post(f"{CONTROL_SERVICE_URL}/{api_route}", json={}, timeout=0.1)
        except requests.exceptions.ReadTimeout:
            # This is GOOD. It means the command was sent, and we didn't wait for the robot to finish.
            pass 
        except Exception as e:
            print(f"Failed to contact control service: {e}")
            return "Robot Connection Failed"

        # Save Image + Label
        entry = save_data(curr_color, curr_depth, log_label, frame_idx, person_detections, min_dist_m)
        dataset_log.append(entry)
        frame_idx += 1
        
        if frame_idx % 5 == 0:
            with open(os.path.join(DATASET_DIR, "log.json"), "w") as f:
                json.dump(dataset_log, f, indent=4)

        status = "PERSON SEEN" if person_detected else "NO PERSON"
        dist_str = f"{min_dist_m:.2f}m" if min_dist_m > 0 else "N/A"

        return f"REC: {log_label} | {status} | {dist_str} (FRAME: {frame_idx})"
    
    return "Ignored"

# === MAIN ===
if __name__ == "__main__":
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    
    load_existing_dataset()

    pipeline = init_camera()
    yolo_model = YOLO('yolov8n.pt') # Load model once at startup
    
    try:
        print("Starting Web Server. Go to http://<ROBOT_IP>:5000")
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    finally:
        if pipeline: pipeline.stop()