import cv2
import threading
import torch
import time
import requests
from ultralytics import YOLO

# !!! IMPORTANT: Set your Pi's IP Address here (Without the :5000) !!!
PI_IP = "10.45.165.198"
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
    # Using your exact local path
    model = YOLO("my_model_6.pt")
    model.to(device)

    print(f"Connecting to Pi stream at {PI_STREAM_URL}...")
    vs = VideoStream(PI_STREAM_URL).start()
    time.sleep(1.0) # Warmup buffer
    
    current_pi_state = "running"
    print("Stream connected! Press 'q' to quit.")

    # Set your maximum allowed box size here (0.15 = 15% of the screen)
    MAX_SIZE_THRESHOLD = 0.15

    while True:
        if not vs.new_frame_available:
            time.sleep(0.005)
            continue

        frame = vs.read()
        if frame is None:
            continue
        
        # 1. THE DISPLAY FRAME
        display_frame = frame.copy() 
        
        # 2. THE GHOST FRAME
        yolo_frame = frame.copy()
        
        # 3. BLINDFOLD THE AI
        #cv2.rectangle(yolo_frame, (0, 0), (200, 75), (0, 0, 0), -1)
        yolo_frame[0:120, 0:200] = (0, 0, 0)
        
        total_frame_area = yolo_frame.shape[0] * yolo_frame.shape[1]
        
        # Run YOLO Inference 
        results = model(yolo_frame, conf=0.55, imgsz=320, verbose=False)
        
        # --- NEW: YOLO DECISION LOGIC ---
        detected_action = "running" # Default to normal line following
        
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = r.names[class_id]
                
                box_w = float(box.xywh[0][2])
                box_h = float(box.xywh[0][3])
                box_area = box_w * box_h
                
                area_percentage = box_area / total_frame_area
                
                print(f"{class_name} | Conf: {confidence:.2f} | Size: {area_percentage*100:.1f}%")
                
                # Check if the box is a valid size
                if area_percentage < MAX_SIZE_THRESHOLD:
                    name_lower = class_name.lower()
                    
                    # Map the YOLO string name to the required action
                    if name_lower == "recycle":
                        detected_action = "spin_360"
                    elif name_lower in ["danger", "stop"]:
                        detected_action = "stop_2s"
                    elif name_lower == "arrow left":
                        detected_action = "turn_left"
                    elif name_lower == "arrow right":
                        detected_action = "turn_right"
                    elif name_lower == "arrow up":
                        detected_action = "move_straight"

                    if detected_action != "running":
                         print(f"   -> ACTION TRIGGERED: {detected_action}")

        # State Machine Update
        new_state = detected_action
        
        if new_state != current_pi_state:
            threading.Thread(target=send_command_to_pi, args=(new_state,), daemon=True).start()
            current_pi_state = new_state

        # Display Results
        for r in results:
            annotated_frame = r.plot(img=display_frame)
            cv2.imshow("Laptop Brain - Low Latency", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            requests.get(f"{PI_COMMAND_URL}/stop") 
            break
        
    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()