import cv2
import os
import sys
import time
from edge_impulse_linux.image import ImageImpulseRunner
from picamera2 import Picamera2

# --- CONFIGURATION ---
MODEL_FILE = "my_fomo_model.eim" 
CONFIDENCE_THRESHOLD = 0.50

# --- PERFORMANCE TUNING ---
# Set to True when driving! Disables the video window and drawing to save massive CPU.
HEADLESS_MODE = True 

def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(dir_path, MODEL_FILE)
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found.")
        sys.exit(1)

    print("Loading Edge Impulse model...")
    runner = ImageImpulseRunner(model_path)
    try:
        runner.init()
    except Exception as e:
        print(f"Failed to initialize model: {e}")
        sys.exit(1)

    print("Initializing Picamera2...")
    picam2 = Picamera2()
    
    # PERFORMANCE BOOST: Lowered capture resolution to reduce CPU resizing overhead
    config = picam2.create_preview_configuration(main={"size": (480, 360), "format": "BGR888"})
    picam2.configure(config)
    picam2.start()

    print(f"Starting camera feed. Headless Mode: {HEADLESS_MODE}")

    try:
        with runner:
            while True:
                start_time = time.time()
                
                bgr_frame = picam2.capture_array("main")
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                
                features, cropped_rgb_frame = runner.get_features_from_image(rgb_frame)

                frame_height, frame_width, _ = cropped_rgb_frame.shape
                camera_center_x = int(frame_width / 2)

                res = runner.classify(features)
                
                if not HEADLESS_MODE:
                    display_frame = cv2.cvtColor(cropped_rgb_frame, cv2.COLOR_RGB2BGR)
                
                if "bounding_boxes" in res["result"]:
                    for box in res["result"]["bounding_boxes"]:
                        if box['value'] > CONFIDENCE_THRESHOLD:
                            x = box['x']
                            y = box['y']
                            label = box['label']
                            
                            # --- AUTONOMOUS CONTROL LOGIC ---
                            error_x = x - camera_center_x
                            
                            # Only print if we found something to keep terminal fast
                            print(f"Target '{label}' | X: {x} | Steering Error: {error_x}")
                            
                            # --- VISUAL DEBUGGING (SKIPPED IF HEADLESS) ---
                            if not HEADLESS_MODE:
                                cv2.circle(display_frame, (x, y), 8, (0, 0, 255), -1)
                                cv2.putText(display_frame, f"{label} {box['value']:.2f}", (x - 20, y - 15), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Calculate FPS
                fps = 1.0 / (time.time() - start_time)
                
                if not HEADLESS_MODE:
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
                    cv2.imshow('Pi 4 Vehicle Tracker', display_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    # In headless mode, just print the FPS occasionally to monitor it
                    print(f"Current FPS: {fps:.1f}", end='\r')

    finally:
        picam2.stop()
        if not HEADLESS_MODE:
            cv2.destroyAllWindows()
        print("\nSystem shut down securely.")

if __name__ == "__main__":
    main()