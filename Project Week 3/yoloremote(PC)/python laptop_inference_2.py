import cv2
import threading
import torch
import time
import requests
from ultralytics import YOLO

# !!! IMPORTANT: Set your Pi's IP Address here !!!
PI_IP = "10.51.237.198:5000"
PI_STREAM_URL = f"http://{PI_IP}:5000/video_feed"
PI_COMMAND_URL = f"http://{PI_IP}:5000/command"

class VideoStream:
    """Thread to empty the OpenCV buffer and eliminate network lag."""
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False
        self.new_frame_available = False 

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if not grabbed:
                self.stop()
            else:
                self.frame = frame
                self.new_frame_available = True 

    def read(self):
        self.new_frame_available = False 
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

def send_command_to_pi(state):
    """Fires the command in the background so it doesn't freeze the video loop."""
    try:
        requests.get(f"{PI_COMMAND_URL}/{state}", timeout=0.5)
        print(f"Sent command to Pi: {state}")
    except:
        pass 

def main():
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Loading YOLO model...")
    model = YOLO('my_model_4.pt')
    model.to(device)

    print(f"Connecting to Pi stream at {PI_STREAM_URL}...")
    vs = VideoStream(PI_STREAM_URL).start()
    time.sleep(1.0) # Warmup buffer
    
    current_pi_state = "running"
    print("Stream connected! Press 'q' to quit.")

    # Set your maximum allowed box size here (0.10 = 10% of the screen)
    MAX_SIZE_THRESHOLD = 0.10

    while True:
        if not vs.new_frame_available:
            time.sleep(0.005)
            continue

        frame = vs.read()
        if frame is None:
            continue
        
        # BANDWIDTH HACK: The Pi is now sending a clean, single frame.
        # No cropping required! Just pass the raw frame directly.
        clean_frame = frame 
        
        # Calculate the total pixel area of the frame
        total_frame_area = clean_frame.shape[0] * clean_frame.shape[1]
        
        # Run YOLO Inference 
        # (Confidence raised to 0.70 now that the image is sharper)
        results = model(clean_frame, conf=0.50, imgsz=320, verbose=False)
        
        # --- YOLO DECISION LOGIC ---
        stop_detected = False
        
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = r.names[class_id]
                
                # Calculate the bounding box size
                box_w = float(box.xywh[0][2])
                box_h = float(box.xywh[0][3])
                box_area = box_w * box_h
                
                # Figure out what percentage of the screen this box covers
                area_percentage = box_area / total_frame_area
                
                # PRINT EVERYTHING IT SEES HERE
                print(f"{class_name} | Conf: {confidence:.2f} | Size: {area_percentage*100:.1f}%")
                
                # !!! IMPORTANT: Replace '0' with your actual target Class ID !!!
                if class_id == 0: 
                    # Only trigger a stop if the box is SMALLER than your threshold
                    if area_percentage < MAX_SIZE_THRESHOLD:
                        stop_detected = True
                        print(f"   -> ACTION: Valid size! Stopping robot.")
                    else:
                        print(f"   -> ACTION: Ignored! Shadow/Blur is too big.")

        # State Machine: Only send network command if state changes
        new_state = "stop" if stop_detected else "running"
        
        if new_state != current_pi_state:
            threading.Thread(target=send_command_to_pi, args=(new_state,), daemon=True).start()
            current_pi_state = new_state

        # Display Results
        for r in results:
            annotated_frame = r.plot()
            cv2.imshow("Laptop Brain - Low Latency", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            requests.get(f"{PI_COMMAND_URL}/stop") 
            break
        
    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()