import time
import json
import os
import cv2
import base64
import random
import numpy as np
import pyrealsense2 as rs
from openai import OpenAI
import requests
import sys
from ultralytics import YOLO 
from flask import Flask, Response
import threading

# === CONFIG ===
VLM_API_URL = "http://10.208.2.89:8000/v1"
# VLM_API_URL = "http://10.208.2.205:8000/v1" # qwen3-4b
VLM_API_KEY = "EMPTY"
VLM_MODEL = "/home/nlp/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct-FP8"
# VLM_MODEL = "/home/nlp/.cache/modelscope/hub/models/Qwen/Qwen3-VL-4B-Instruct"
DATASET_DIR = "dataset"
CONTROL_SERVICE_URL = "http://localhost:5051/"

app = Flask(__name__)
output_frame = None
frame_lock = threading.Lock()

def init_camera():
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30) 
        pipeline.start(config)

        align = rs.align(rs.stream.color)
        print("Camera Connected (Color + Depth)")
        return pipeline, align
    except Exception as e:
        print(f"Camera Failed: {e}")
        return None, None
    
def send_command(api_route, payload={}):
    try: 
        # requests.post(f"{CONTROL_SERVICE_URL}/{api_route}", json=payload, timeout=0.1)
        response = requests.post(f"{CONTROL_SERVICE_URL}/{api_route}", json=payload, timeout=3.5)
        
        if response.status_code == 200:
            print("Robot finished moving.")
    # except requests.exceptions.ReadTimeout:
    #     # This is GOOD. It means the command was sent, and we didn't wait for the robot to finish.
    #     pass 
    except Exception as e:
        print(f"Failed to contact control service: {e}")
        return "Robot Connection Failed"

def encode_image_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def encode_cv2_image(img):
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

def load_full_dataset():
    """
    Loads the entire dataset into memory grouped by action.
    Returns a Dictionary: {'FORWARD': [entry1, entry2...], 'LEFT': [...]}
    """
    log_path = os.path.join(DATASET_DIR, "log.json")
    if not os.path.exists(log_path):
        print("No dataset found.")
        return {}

    with open(log_path, "r") as f:
        data = json.load(f)
    
    dataset_by_action = {}
    for entry in data:
        action = entry["action"]
        # Verify file exists before adding
        img_path = entry.get("color_path")
        if img_path and os.path.exists(img_path):
            if action not in dataset_by_action: dataset_by_action[action] = []
            dataset_by_action[action].append(entry)
    
    print(f"Loaded {len(data)} total records.")
    return dataset_by_action

def pick_random_examples(dataset_by_action, n_shots=1):
    """
    Picks n_shots random examples for each available action.
    """
    teaching_set = []
    for _, entries in dataset_by_action.items():
        if entries:
            # Ensure we don't try to pick more samples than we have
            count = min(len(entries), n_shots)
            samples = random.sample(entries, count)
            teaching_set.extend(samples)
    return teaching_set

def get_vlm_decision(client, current_img, examples, dep_img=None, memory=None):
    """
    Constructs the Few-Shot Prompt
    """
    messages_content = []
    
    # 1. System Instruction
    messages_content.append({
        "type": "text", 
        "text": "You are a curious, friendly robot dog that is navigate the world."
                "If you go off course try turning LEFT or RIGHT is 90 degrees and SLIGHT LEFT or SLIGHT RIGHT is 30 degrees"
                "Use the depth image to determine if you are walking too close to the wall and correct your drift as needed."
                "Try to navigate mostly with turning and limit backing up as much as possible"
    })
    
    # 2. Add Teaching Examples
    for ex in examples:
        img_path = ex.get("color_path")
        dep_path = ex.get("depth_path")
        
        if img_path and os.path.exists(img_path):
            img_b64_ex = encode_image_file(img_path)
            messages_content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64_ex}"}
            })
            messages_content.append({"type": "text", "text": f"Action: {ex['action']}"})
        
        if dep_path and os.path.exists(dep_path):
            dep_b64_ex = encode_image_file(dep_path)
            messages_content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{dep_b64_ex}"}
            })
            messages_content.append({"type": "text", "text": f"Action: {ex['action']}"})

    
    # 3. Add Current Reality
    b64_curr = encode_cv2_image(current_img)
    messages_content.append({
        "type": "image_url", 
        "image_url": {"url": f"data:image/jpeg;base64,{b64_curr}"}
    })
    messages_content.append({"type": "text", "text": "Action:"})

    valid_choices = [
        "FORWARD", "BACKWARD", "LEFT", "RIGHT", "SLIGHT LEFT", "SLIGHT RIGHT"
    ]

    # 4. Memories 
    messages_content.append({"type": "text", "text": f"To help with your decision, here were your last 4 actions {memory[-4:]}"})

    # 5. Call API
    try:
        response = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{"role": "user", "content": messages_content}],
            temperature=0.1, 
            max_tokens=10,
            extra_body={"guided_choice": valid_choices}
        )
        return response.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"VLM Error: {e}")
        return "STOP"

def generate_frames():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                continue 

            flag, encodedImage = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')
        time.sleep(0.033) # Cap stream at ~30fps

def is_oscillating(history):
    """
    Detects if the last 4 moves form an A-B-A-B pattern.
    Example: Forward, Backward, Forward, Backward
    """
    if len(history) < 4:
        return False
    
    # Check if its not just Forward-Forward-Forward-Forward
    return (history[-1] == history[-3] and 
            history[-2] == history[-4] and 
            history[-1] != history[-2])

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/")
def index():
    return "<h1>Robot VLM Brain Live Feed</h1><img src='/video_feed'>"

def start_flask():
    # Run on port 5001 to avoid conflict with control_service (5051)
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)

# === MAIN LOOP ===
if __name__ == "__main__":
    # Flask Setup
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    print("Video Feed running at: http://<ROBOT_IP>:5001")

    # Hardware Setup
    pipeline, align = init_camera()
    if not pipeline: sys.exit(1)
    
    # AI Setup
    client = OpenAI(api_key=VLM_API_KEY, base_url=VLM_API_URL)
    yolo_model = YOLO("yolov8n.pt")
    memory = []
    yolo_timing = []
    vlm_timing = []
    
    print("Loading examples...")
    teaching_examples = load_full_dataset()
    
    if len(teaching_examples) == 0:
        print("Error: Dataset is empty. Record some moves first!")
        sys.exit(1)
    
    print("\n=== VLM AUTOPILOT STARTED ===")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            # A. Capture
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame: continue

            color_img = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())
            h, w = depth_img.shape

            # ====== yolo priority check ======
            # if YOLO sees a person, YOLO takes control 
            t0_yolo = time.perf_counter()

            results = yolo_model(color_img, verbose=False, conf=0.5)
            person_box = None
            max_area = 0 

            t1_yolo = time.perf_counter()
            yolo_latency = t1_yolo - t0_yolo
            yolo_timing.append(yolo_latency)

            # Create visual frame for web stream
            vis_img = color_img.copy()
            dep_img = depth_img.copy()

            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        area = (x2-x1) * (y2-y1)

                        if area > max_area:
                            max_area = area
                            person_box = (x1, y1, x2, y2)
                        
                        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(vis_img, "Person" , (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            with frame_lock:
                output_frame = vis_img
            
            if person_box:
                x1, y1, x2, y2 = map(int, person_box)
                box_cx = (x1 + x2) // 2
                box_cy = (y1 + y2) // 2

                # Ensure bounds and depth at person 
                box_cx = min(max(0, box_cx), w-1)
                box_cy = min(max(0, box_cy), h-1)

                dist_mm = depth_img[box_cy, box_cx]
                if dist_mm == 0: # Filter bad pixels
                     roi = depth_img[max(0,box_cy-5):min(h,box_cy+5), max(0,box_cx-5):min(w,box_cx+5)]
                     valid = roi[roi > 0]
                     if len(valid) > 0: dist_mm = np.mean(valid)
                     else: dist_mm = 9999

                # Tracking logic 
                if 0 < dist_mm < 1200:
                    print(f"YOLO: Reached Person ({dist_mm:.0f}mm). Greeting!")
                    send_command("action/hand_shake") # Auto-Greet!
                    time.sleep(0.5)
                    send_command("action/turn_180")
                else:
                    # PID Steering
                    img_cx = w / 2
                    error = (img_cx - box_cx) / w
                    turn_speed = error * 1.8 # Steering gain
                    
                    print(f"YOLO Tracking: Dist={dist_mm:.0f}mm Turn={turn_speed:.2f}")
                    send_command("move", {"vx": 0.5, "vy": 0.0, "yaw": turn_speed})
                
                print(f"[TIMING] YOLO: {yolo_latency:.4f}s | MODE: SERVO (Tracking Person)")
                # Fast Loop for smooth tracking (skip VLM)
                time.sleep(0.05)
                continue 

            # ====== VLM Control (No Person) ======
            cx, cy = w // 2, h // 2
            center_dist = depth_img[cy-10:cy+10, cx-10:cx+10]
            valid_dist = center_dist[center_dist > 0]
            dist_mm = 9999
            if len(valid_dist) > 0:
                dist_mm = np.mean(valid_dist)

            print(f"Dist: {dist_mm:.0f}mm", end=" | ")
            
            t0_vlm = time.perf_counter()
            curr_examples = pick_random_examples(teaching_examples, n_shots=5)
            decision = get_vlm_decision(client, color_img, curr_examples, dep_img=dep_img, memory=memory)
            t1_vlm = time.perf_counter()
            vlm_latency = t1_vlm - t0_vlm
            vlm_timing.append(vlm_latency)

            print(f"VLM: {decision} | Latency: {vlm_latency:.4f}s", end=" | ")

            if "FORWARD" in decision and 0 < dist_mm < 1000:
                print("SAFETY STOP: Too close to target (1m)!")
                send_command("action/turn_left")

            elif "FORWARD" in decision:
                send_command("action/forward")

            elif "BACKWARD" in decision:
                send_command("action/backward")

            elif "LEFT" in decision:
                send_command("action/turn_left")

            elif "RIGHT" in decision:
                send_command("action/turn_right")

            elif "SLIGHT LEFT" in decision:
                send_command("action/nudge_left")

            elif "SLIGHT RIGHT" in decision:
                send_command("action/nudge_right")
        
            time.sleep(1.0) # thinking time 
        
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        send_command("stop_movement")
        pipeline.stop()
        print("Average YOLO Latency: {:.4f}s".format(sum(yolo_timing)/len(yolo_timing) if yolo_timing else 0))
        print("Average VLM Latency: {:.4f}s".format(sum(vlm_timing)/len(vlm_timing) if vlm_timing else 0))