from openai import OpenAI
from collections import deque
import RealSenseCamera

import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import time
import base64
import requests # To talk to the 'Big Brain' (VLM)
from ultralytics import YOLO
import time

class ActionAgent():
    def __init__(self, vlm_url):
        # ===== Enable the RealSense Camera =====
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        align_to = rs.stream.color # map Depth pixels to Color pixel
        self.align = rs.align(align_to)

        # ===== Send the capture frames to remote server (AKA the VLM) =====
        self.vlm_url = vlm_url 
        self.latest_vlm_timestamp = 0
        self.latest_vlm_response = "Initializing..."

        # ===== Robot configurations ===== 
        self.yolo_model = YOLO('yolov8n.pt') # lightweight model - safety net + reflex
        self.SAFE_DISTANCE = 1500 # 1.5 meters
        self.EMER_STOP_DISTANCE = 800
        self.current_state = "EXPLORING"
        self.explore_target = None 

        # ===== Initalize Threads ======
        self.stop_flag = threading.Event() # clear signalling 
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        # self.detect_lock = threading.Lock() 

        self._captureThread = threading.Thread(target=self._exploring_loop, daemon=True)


    def start_thread(self):
        self._captureThread.start()

    def stop_thread(self):
        print("\nStopping Threads...")
        self.stop_flag.set() # set the stop_flag
        self._captureThread.join(timeout=1)
        print("Threads joined")

    """===== Encode captured images into Base64 before sending to the VLM ====="""
    def encode_image(self, img):
        _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        return base64.b64encode(buffer).decode('utf-8')
    
    def _exploring_loop(self):
        self.pipeline.start(self.config)
        
        try:
            while not self.stop_flag.is_set():
                if self.current_state == "EXPLORING":
                    img = None
                    
                    with self.frame_lock:
                        if self.latest_frame is not None:
                            img = self.latest_frame.copy()

                    if img is not None:
                        try: 
                            print("\n [VLM]: Scanning room for interesting objects...")
                            base64_img = self.encode_image(img)
                            payload = {"image": base64_img, "prompt": "Identify one object in the image to explore. No need to explain."}
                            response = requests.post(self.vlm_url, json=payload, timeout=5)

                            target = response.json().get("object", "").strip().lower()
                            if target and "person" not in target:
                                self.explore_target = target
                                print(f"[VLM]: Let's go look at the {target}")
                        except: pass
            time.sleep(0.5)
            # time.sleep(3) # what google recommends

        except Exception as e:
            print(f"Exploring Image error: {e}")

    """===== Given there are mutiple huamsn in the frame, choose the cloest human in the frame ======"""
    def get_closest_human(self, detections):
        closest_human = None
        max_area = 0
        
        if detections[0].boxes:
            for box in detections[0].boxes:
                label = self.yolo.name[int[box.cls[0]]]
                if label == "person":
                    w = box.xywh[0][2]
                    h = box.xywh[0][3]
                    area = w * h

                    # decide which bounding box is the largest (aka closest to the camera)
                    if area > max_area:
                        max_area = area 
                        closest_human = box
        
        return closest_human
    
    def get_object_center_distance(self, depth, box):
        x1, y1, x2, y1 = box.xyxy[0].cpu().numpy().astype(int)
        cx = (x1 + x2) // 2 

        # clamp 
        h, w = depth.shape 
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2) 

        crop = depth[y1:y2, x1:x2]
        if crop.size == 0: 
            return 9999, cx
        
        valid = crop[crop > 0]
        if valid.size > 0:
            dist = np.mead(valid)
        else:
            return 9999
        
        return dist, cx
        

    """===== Capture color and depth frames ====="""
    def capture_frames(self):
        self.pipeline.start(self.config)

        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        if not color_frame:
            return
            
        rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        with self.frame_lock:
            self.latest_frame = rgb.copy()
        
        return rgb, depth

    def run(self):
        print("Robot Active")

        try:
            while True:
                # Hardware update, get frames
                rgb, depth = self.capture_frames()

                # run perception -- always check for obstacles
                obstacle_mm = self.get_object_center_distance(depth)

                # check for objects/ people 
                results = self.yolo.predict(rgb, verbose=False, conf=0.5)
                closest_human_box = self.get_closest_human(results)

                # ===== Decision Hierarchy =====

                # +++++ Layer 1: COLLISION DETECTION 
                if obstacle_mm < self.EMER_STOP_DISTANCE:
                    self.current_state = "EMERGENCY_STOP"
                    print(f"\r[COLLISION]: BACKUP -- OBJECT(S) DETECTED TOO CLOSE ({obstacle_mm}mm)")
                
                # +++++ Layer 2: SOCIAL 
                elif closest_human_box is not None:
                    self.current_state = "PERSON DETECTED"
                    self.explore_target = None # forget pervious target (ie: couch), follow human
                    
                    dist_mm, cx = self.get_object_center_distance(depth, closest_human_box)
                    dist_error_m = (dist_mm - self.SAFE_DISTANCE) / 1000.0

                    if abs(dist_error_m) < 0.2:
                        print(f"\r[SOCIAL]: Person found, greeting! ({dist_mm/1000:.1f}m)", end="")
                    elif dist_error_m > 0:
                        print(f"\r[SOCIAL]: Person is walking away, following! ({dist_mm/1000:.1f}m)", end="")
                    else:
                        print(f"\r[SOCIAL]: Person TOO CLOSE, backing up! ({dist_mm/1000:.1f}m)", end="")

                # +++++ Layer 3: EXPLORE 
                elif self.explore_target is not None:
                    self.current_state = "EXPLORING"
                    target_box = None
                    
                    for box in results[0].boxes:
                        label = self.yolo.names[int(box.cls[0])]
                        if label in self.explore_target:
                            target_box = box 
                            break
                    
                    if target_box:
                        dist_mm, cx = self.get_object_center_distance(depth, target_box)

                        # move towards the target object
                        if dist_mm > self.SAFE_DISTANCE:
                            print(f"\r[EXPLORE]: Investigating {self.explore_target}, {dist_mm/1000:.1f}m away", end="")
                        
                        # sucessfully reached the object, back to exploring something new 
                        else:
                            print(f"\r[EXPLORE]: I found the {self.explore_target} object!", end="")
                            self.explore_target = None
                            target_box = None 
                            time.sleep(0.5) # look for a new object 
                            # i dont understand the target box
                    
                else:
                    self.current_state = "WANDERING"
                    print(f"\r[WANDER]: Scanning the room....", end="")    
                
                if closest_human_box is not None:
                     x1, y1, x2, y2 = closest_human_box.xyxy[0].cpu().numpy().astype(int)
                     cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 0, 255), 3) # RED BOX for Human
                
                cv2.imshow("Robot Brain", rgb)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

        finally:
            self.stop_event.set()
            self.pipeline.stop()
            cv2.destroyAllWindows()




