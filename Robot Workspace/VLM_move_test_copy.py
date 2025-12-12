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

# === CONFIG ===
VLM_API_URL = "http://10.208.2.89:8000/v1"
VLM_API_KEY = "EMPTY"
VLM_MODEL = "/home/nlp/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct-FP8"
DATASET_DIR = "dataset"
CONTROL_SERVICE_URL = "http://localhost:5051/"

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
        requests.post(f"{CONTROL_SERVICE_URL}/{api_route}", json=payload, timeout=0.1)
    except requests.exceptions.ReadTimeout:
        # This is GOOD. It means the command was sent, and we didn't wait for the robot to finish.
        pass 
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
    n_shots=1 -> One example per action (Fastest)
    n_shots=3 -> Three examples per action (Smarter, Slower)
    """
    teaching_set = []
    for _, entries in dataset_by_action.items():
        if entries:
            # Ensure we don't try to pick more samples than we have
            count = min(len(entries), n_shots)
            samples = random.sample(entries, count)
            teaching_set.extend(samples)
    return teaching_set

def get_vlm_decision(client, current_img, examples):
    """
    Constructs the Few-Shot Prompt
    """
    messages_content = []
    
    # 1. System Instruction
    messages_content.append({
        "type": "text", 
        "text": "You are a curious, friendly robot dog that approaches humans and avoid obstacles."
                "I will show you examples of what to do."
                "Walk towards humans. If close, HAND SHAKE to them. "
                "If a human is moving away, FOLLOW them."
    })
    
    # 2. Add Teaching Examples
    for ex in examples:
        img_path = ex.get("color_path", ex.get("image_path"))
        
        if img_path and os.path.exists(img_path):
            b64_ex = encode_image_file(img_path)
            messages_content.append({
                "type": "image_url", 
                "image_url": {"url": f"data:image/jpeg;base64,{b64_ex}"}
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
        "FORWARD", "BACKWARD", "LEFT", "RIGHT", "SIT", "HAND SHAKE", "FOLLOW"
    ]

    # 4. Call API
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

# === MAIN LOOP ===
if __name__ == "__main__":
    # 1. Hardware Setup
    pipeline, align = init_camera()
    if not pipeline: sys.exit(1)
    
    # 2. AI Setup
    client = OpenAI(api_key=VLM_API_KEY, base_url=VLM_API_URL)
    yolo_model = YOLO("yolov8n.pt")
    
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
            results = yolo_model(color_img, verbose=False, conf=0.5)
            person_box = None
            max_area = 0 

            for r in results:
                for box in r.boxes:
                    if int(box.cls[0]) == 0:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        area = (x2-x1) * (y2-y1)

                        if area > max_area:
                            max_area = area
                            person_box = (x1, y1, x2, y2)
            
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
                # Randomize greeting
                # if random.random() > 0.5: send_command("action/hand_shake")
                # else: send_command("action/heart")

                if 0 < dist_mm < 1200:
                    print(f"YOLO: Reached Person ({dist_mm:.0f}mm). Greeting!")
                    send_command("action/stop_movement")
                    time.sleep(0.5)
                    send_command("action/hand_shake") # Auto-Greet!
                    time.sleep(0.5)
                    send_command("action/backward")
                    time.sleep(0.5)
                    send_command("action/turn_left")
                    time.sleep(1.0)
                else:
                    # PID Steering
                    img_cx = w / 2
                    error = (img_cx - box_cx) / w
                    turn_speed = error * 1.8 # Steering gain
                    
                    print(f"YOLO Tracking: Dist={dist_mm:.0f}mm Turn={turn_speed:.2f}")
                    send_command("move", {"vx": 0.5, "vy": 0.0, "yaw": turn_speed})
                
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
            curr_examples = pick_random_examples(teaching_examples, n_shots=1)
            decision = get_vlm_decision(client, color_img, curr_examples)
            print(f"VLM: {decision}")

            if "FORWARD" in decision and 0 < dist_mm < 1000:
                print("SAFETY STOP: Too close to target (1m)! Backing up")
                send_command("action/backward")
            
            elif "FORWARD" in decision:
                send_command("action/forward")
            elif "BACKWARD" in decision:
                send_command("action/backward")
            elif "LEFT" in decision:
                send_command("action/turn_left")
            elif "RIGHT" in decision:
                send_command("action/turn_right")
            else:
                print("Unsure what to do...")
                send_command("action/sit_down")
        
            time.sleep(1.0) # thinking time 
        
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        send_command("stop_movement")
        pipeline.stop()