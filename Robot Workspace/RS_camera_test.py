import pyrealsense2 as rs
import numpy as np
import cv2
import time

def take_realsense_photo():
    print("1. Starting RealSense Camera...")
    
    # Configure the camera pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Enable the Color stream (640x480 is standard, 30fps)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start the camera
    pipeline.start(config)
    
    try:
        print("2. Warming up camera (waiting for auto-exposure)...")
        # RealSense cameras start 'dark'. We must read ~30 frames to let it adjust.
        for i in range(30):
            pipeline.wait_for_frames()
            
        print("3. Capturing photo...")
        # Get the actual frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            print("Error: No color frame found!")
            return

        # Convert the raw RealSense data to a standard Image (numpy array)
        # We use numpy to turn the "buffer" of data into a matrix of pixels
        color_image = np.asanyarray(color_frame.get_data())

        # Save the image
        filename = "./uploads/realsense_img.jpg"
        cv2.imwrite(filename, color_image)
        print(f"Success! Saved to {filename}")

    finally:
        # Always stop the pipeline, or the camera might stay 'busy' and block other apps
        pipeline.stop()
        print("Camera stopped.")

if __name__ == "__main__":
    take_realsense_photo()