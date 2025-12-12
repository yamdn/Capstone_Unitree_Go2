import sys
import time
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO  
from flask import Flask, Response
import threading 
import requests # communicate with control_service.py
import base64
import re
from openai import OpenAI


app = Flask(__name__)
CONTROL_SERVICE_URL = "http://localhost:5051/action"

# 1. Initialize RealSense Camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 30) 
align = rs.align(rs.stream.color)     # align object to map depth pixels to color pixels (they're roughly 1 in apart, so this is necessary)
pipeline.start(config)

# 2. Load the AI Model (Nano version for speed)
yolo_model = YOLO('yolov8s.pt')
TARGET_CLASS_ID = 0  # yolo class '0' is person 

VLM_API_URL = "http://10.208.2.89:8000/v1"
VLM_API_KEY = "EMPTY"
VLM_MODEL = "/home/nlp/.cache/modelscope/hub/models/Qwen/Qwen3-VL-8B-Instruct-FP8"

# 3. Data Attributes 
latest_frame = None 
latest_depth = None
latest_detections = None
latest_yolo_results = []  # From Inference Thread (Safety)
latest_vlm_target = None  # From VLM Thread (Navigation Target: cx_ratio, cy_ratio)

# 4. Concurrency Tools 
stop_flag = threading.Event()
frame_lock = threading.Lock()
detect_lock = threading.Lock()

def encode_to_base64(img):
    """Convert a PIL image directly to base64 string."""
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

"""
    capture_frames Function: 
        - start the Real Sense camera on the robot
        - obtain the color and depth frames 
"""
def capture_frames():
    global latest_frame, latest_depth 

    try:
        while not stop_flag.is_set():
            frames = pipeline.wait_for_frames()

            aligned_frames = align.process(frames)
            depth_frames = aligned_frames.get_depth_frame()
            color_frames = aligned_frames.get_color_frame()

            if not color_frames or not depth_frames:
                continue 
            
            color_image = np.asanyarray(color_frames.get_data())
            depth_image = np.asanyarray(depth_frames.get_data())

            with frame_lock:
                latest_frame = color_image.copy()
                latest_depth = depth_image.copy()
        
    except Exception as e:
        print(f"Capture Frame Camera Error: {e}")
        time.sleep(0.5)

def inference_loop():
    global latest_detections
    while not stop_flag.is_set():
        yolo_frame = None

        with frame_lock:
            if latest_frame is not None:
                yolo_frame = latest_frame.copy()
        
        if yolo_frame is None:
            time.sleep(0.1)
            continue
        
        results = yolo_model(yolo_frame, verbose=False, conf=0.7)

        with detect_lock:
            latest_detections = results

def query_vlm():
    global latest_vlm_target 
    client = OpenAI(api_key=VLM_API_KEY, base_url=VLM_API_URL)

    while not stop_flag.is_set():
        vlm_frame = None

        with frame_lock:
            if latest_frame is not None:
                vlm_frame = latest_frame.copy()

        if vlm_frame is None:
            time.sleep(0.1)
            continue
            
        b64_img = encode_to_base64(vlm_frame)
        image_mime_type = "image/jpg"  # Adjust if using jpg/jpeg etc.
        image_url = f"data:{image_mime_type};base64,{b64_img}"

        prompt = (f"\rAnalyze the image and return a list of detected objects. For each detected object, output: "
                    "'label': object class name"
                    "'confidence': value between 0 and 1"
                    "'bbox': [x_min, y_min, x_max, y_max] normalized 0-1000"
                    "No natural language explanation. No extra text. If not found, say [0,0,0,0]")

        messages = [
            {  
                "role": "user",
                "content": [
                    {  
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },  
                    {"type": "text", "text": prompt},
                ],  
            }  
        ]  

        try:
            response = client.chat.completions.create(
                model=VLM_MODEL,
                messages=messages,
                temperature=0.1,
            )  

            content = response.choices[0].message.content
            content = content.replace("```json", "").replace("```", "").strip()
            print(f"(Raw) VLM Response: {content}\n")

            # Regex to find the pattern [y1, x1, y2, x2]
            match = re.search(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]", content)

            if match:
                y1, x1, y2, x2 = map(int, match.groups())
                if x1 != 0 or x2 != 0:
                    cx = (x1 + x2) / 2 / 1000.0 # Ratio 0.0-1.0
                    cy = (y1 + y2) / 2 / 1000.0
                    with detect_lock:
                        latest_vlm_target = (cx, cy)
                        print(f"VLM Target at ({cx:.2f}, {cy:.2f})")
                else:
                    with detect_lock: latest_vlm_target = None
            
            else:
                with detect_lock: latest_vlm_target = None

        except Exception as e:
            print(f"VLM Connection Error: {e}")
        
        time.sleep(0.1)

        
def generate_frames():
    try:
        while True:    
            output_frame = None
            current_depth = None
            current_results = None 

            # grab data safely
            with frame_lock:
                if latest_frame is not None:
                    output_frame = latest_frame.copy()
                    if latest_depth is not None:
                        current_depth = latest_depth.copy()
                
            with detect_lock:
                current_results = latest_detections

            # annotate frames 
            if output_frame is not None and current_results is not None and current_depth is not None:
                for r in current_results:
                    boxes = r.boxes
                    for box in boxes:
                        # Get Box Coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        label = r.names[int(box.cls[0])]
                        
                        # Calculate Center Point
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        # Get Distance from Aligned Depth Map
                        # Safety check: ensure point is within frame bounds
                        if 0 <= center_y < current_depth.shape[0] and 0 <= center_x < current_depth.shape[1]:
                            dist_mm = current_depth[center_y, center_x]
                            dist_m = dist_mm / 1000.0
                            
                            # Draw Box
                            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Draw Distance Label (This is your feedback!)
                            label_text = f"{label}: {dist_m:.2f}m"
                            cv2.putText(output_frame, label_text, (x1, y1 - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            # Draw center dot
                            cv2.circle(output_frame, (center_x, center_y), 5, (0, 0, 255), -1)
            
            if output_frame is not None:
                ret, buffer = cv2.imencode('.jpg', output_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            time.sleep(0.05) # Limit stream FPS to save bandwidth
                  
    except Exception as e:
        print(f"Error: {e}")

# def robot_action_loop():
#     robot_state = "standing"
#     last_seen_person_time = 0 

#     while not stop_flag.is_set():
#         person_detected = False

#         with detect_lock:
#             if latest_detections is not None:
#                 for r in latest_detections:
#                     if 0 in r.boxes.cls:
#                         person_detected = True
#                         last_seen_person_time = time.time()
#                         break

#         try:
#             if person_detected and robot_state == "standing":
#                 requests.post(f"{CONTROL_SERVICE_URL}/sit_down", json={})
#                 robot_state = "sitting"
            
#             elif not person_detected and robot_state == "sitting":
#                 # Only stand up if we haven't seen a person for 5 seconds (debounce)
#                 if (time.time() - last_seen_person_time) > 5.0:
#                     requests.post(f"{CONTROL_SERVICE_URL}/get_up", json={})
#                     robot_state = "standing"
        
#         except Exception as e:
#             print(f"Could not talk to Control Service: {e}")
        
#         time.sleep(0.5)


@app.route('/')
def index():
    return "<html><body><h1>Robot Vision</h1><img src='/video_feed'></body></html>"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("Starting Threads...")

    captureThread = threading.Thread(target=capture_frames, daemon=True)
    inferenceThread = threading.Thread(target=inference_loop, daemon=True)
    # actionThread = threading.Thread(target=robot_action_loop, daemon=True)
    vlmThread = threading.Thread(target=query_vlm, daemon=True)

    captureThread.start()
    inferenceThread.start()
    # actionThread.start()
    vlmThread.start()

    try:
        # Use port 5000
        print("Starting stream. Open http://<ROBOT_IP>:5000 in your browser.")
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Stopping...")
        stop_flag.set()
        captureThread.join(timeout=1)
        inferenceThread.join(timeout=1)
        # actionThread.join(timeout=1)
        vlmThread.join(timeout=1)
        pipeline.stop()