import pyrealsense2 as rs
import numpy as np
import cv2
import threading
import time
from ultralytics import YOLO  

class RealSenseCamera():
    def __init__(self, yolo_model):
        # ===== Data Attributes ===== 
        self.latest_frame = None
        self.latest_depth = None 
        self.latest_detections = [] 
        self.yolo_model = YOLO(yolo_model)

        # ===== Concurrency Tools ===== 
        self.stop_flag = threading.Event() # clear signalling 
        self.frame_lock = threading.Lock() # protects shared data
        self.detect_lock = threading.Lock()

        # ===== initalize the threads ===== 
        """ t1 = threading.Thread(target=capture_frames, daemon=True) """
        self._captureThread = threading.Thread(target=self._capture_frames, daemon=True)
        self._inferenceThread = threading.Thread(target=self._inference_loop, daemon=True)

    def start_thread(self):
        self._captureThread.start()
        self._inferenceThread.start()
    
    def stop_thread(self):
        print("\nStopping Threads...")
        self.stop_flag.set() # set the stop_flag
        self._captureThread.join(timeout=1)
        self._inferenceThread.join(timeout=1)
        print("Threads joined")

    def _capture_frames(self):
        # global latest_frame, latest_depth, stop_flag

        # ===== Enable the RealSense Camera =====
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # ===== Create an align object to map Depth pixels to Color pixels =====
        align_to = rs.stream.color
        align = rs.align(align_to)
        pipeline.start(config)

        try:
            while not self.stop_flag.is_set():
                frames = pipeline.wait_for_frames()

                # Align the depth frame to color frame
                aligned_frames = align.process(frames)
            
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                if not color_frame or not aligned_depth_frame:
                    continue
        
                # Convert to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(aligned_depth_frame.get_data())
        
                # Update the global frame variable safely
                with self.frame_lock:
                    self.latest_frame = color_image.copy()
                    self.latest_depth = depth_image.copy()
        
        except Exception as e:
            print(f"Camera error: {e}")
        finally:
            pipeline.stop()

    def _inference_loop(self):
        # global latest_frame, latest_depth, latest_detections, stop_flag

        while not self.stop_flag.is_set():
            # Get a snapshot of the current frame to run AI on
            frame_for_ai = None
            depth_for_ai = None

            with self.frame_lock:
                if self.latest_frame is not None and self.latest_depth is not None:
                    frame_for_ai = self.latest_frame.copy()
                    depth_for_ai = self.latest_depth.copy()
            
            # If no frame yet, wait
            if frame_for_ai is None:
                time.sleep(0.1)
                continue

            # Run YOLO Inference, verbose=False keeps the terminal clean
            results = self.yolo_model.predict(source=frame_for_ai, conf=0.50, save=False, verbose=False)
            
            # Process results into a simple list of data
            new_detections = []
            if len(results) > 0 and results[0].boxes:
                for box in results[0].boxes:
                    # 1. Get Box Coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = self.yolo_model.names[cls_id]

                    # 2. Calulate Distance -- Ensure coordinates are within image bounds
                    h, w = depth_for_ai.shape
                    x1, x2 = max(0, x1), min(w, x2)
                    y1, y2 = max(0, y1), min(h, y2)

                    # Extract depth region for the detected object 
                    object_depth = depth_for_ai[y1:y2, x1:x2]
                    distance_mm = 0

                    if object_depth.size > 0:
                        valid_depths = object_depth[object_depth > 0]

                        if valid_depths.size > 0:
                            distance_mm = np.mean(valid_depths)

                    new_detections.append((x1, y1, x2, y2, label, conf, distance_mm))

            # Update the global detections variable safely
            with self.detect_lock:
                self.latest_detections = new_detections
                
            time.sleep(0.01)

    def _annotate(self, img, detections):
        annotated_img = img.copy()
        closest_dist = 99999

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 1
        box_thickness = 2

        # Loop through detections
        for (x1, y1, x2, y2, label, conf, distance_mm) in detections:
            color = (0, 255, 0) # green safe

            if 0 < distance_mm < 1500:
                color = (0, 0, 255) # red warning 
                if distance_mm < closest_dist:
                    closest_dist = distance_mm            

            # Draw bounding box 
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, box_thickness)

            # Format text (Convert mm to meters)
            dist_m = distance_mm / 1000.0
            text = f"{label}: {conf:.2f}: {dist_m:.2f}m"

            # Draw label on box 
            cv2.putText(annotated_img, text, (x1, y1 - 5), font, font_scale, (255,255,255), font_thickness)
            print(f"{label}, {dist_m}", end="")

        if closest_dist < 1500:
            # using \r overwrites the line in terminal so it doesn't get spammed
            print(f"\rWARNING: Object detected within 1m! ({closest_dist/1000:.2f}m)   ", end=" ")
        else:
            print(f"\rSafe distance...", end="\n")

        return annotated_img
    
    def run(self):
        self.start_thread() # start the threads 

        try:
            while True:
                current_frame = None
                with self.frame_lock:
                    if self.latest_frame is not None:
                        current_frame = self.latest_frame.copy()
                
                if current_frame is not None:
                    # 2. Grab the latest known box locations
                    current_detections = []
                    with self.detect_lock:
                        current_detections = self.latest_detections
                    
                    # 3. Combine them immediately before showing
                    final_image = self._annotate(current_frame, current_detections)
                    cv2.imshow("Smooth YOLO Tracking", final_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            pass
        finally:
            # Allow threads time to see the stop flag
            self.stop_thread()
            cv2.destroyAllWindows()
            print("Program finished.")

# --- Execution Block ---
if __name__ == "__main__":
    camera_system = RealSenseCamera(yolo_model='yolov8m.pt')
    camera_system.run()





