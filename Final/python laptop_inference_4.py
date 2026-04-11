import cv2
import threading
import torch
import time
import requests
from ultralytics import YOLO

# !!! IMPORTANT: Set your Pi's IP Address here (Without the :5000) !!!
PI_IP = "10.51.237.198"
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

    print("Loading models...")
    # 1. Load your original symbol model
    symbol_model = YOLO("my_model_13.pt")
    symbol_model.to(device)
    
    # 2. Load your new facial recognition model
    # Replace with your actual face model path
    face_model = YOLO("my_face_model.pt") 
    face_model.to(device)
    
    # NEW: Load OpenCV's built-in generic face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print(f"Connecting to Pi stream at {PI_STREAM_URL}...")
    vs = VideoStream(PI_STREAM_URL).start()
    time.sleep(1.0) # Warmup buffer
    
    current_pi_state = "running"
    print("Stream connected! Press 'q' to quit.")

    MAX_SIZE_THRESHOLD = 0.15
    
    # State tracking for Face Authentication
    face_auth_mode = False
    auth_timeout_start = 0
    face_successfully_matched = False
    AUTH_DURATION = 10.0 # Mandatory wait time in seconds

    while True:
        if not vs.new_frame_available:
            time.sleep(0.005)
            continue

        frame = vs.read()
        if frame is None:
            continue
        
        display_frame = frame.copy() 

        
        # ==========================================
        # MODE 1: FACE AUTHENTICATION ACTIVE
        # ==========================================
        if face_auth_mode:
            if current_pi_state != "stop_for_auth":
                threading.Thread(target=send_command_to_pi, args=("stop_for_auth",), daemon=True).start()
                current_pi_state = "stop_for_auth"

            cv2.putText(display_frame, "AUTH MODE ACTIVE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Run YOLO Face Model Inference
            results = face_model(frame, conf=0.6, imgsz=320, verbose=False)
            
            yolo_found_authorized = False
            yolo_detected_any_face = False # NEW: Track if YOLO sees literally anything

            # 1. Scan for faces using YOLO
            for r in results:
                # If YOLO returns any bounding boxes, it detected a face/object
                if len(r.boxes) > 0:
                    yolo_detected_any_face = True 
                    
                for box in r.boxes:
                    class_name = r.names[int(box.cls[0])]
                    if class_name.lower() == "authorized_person": 
                        face_successfully_matched = True
                        yolo_found_authorized = True
                        cv2.putText(display_frame, "MATCHED - WAITING...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw YOLO bounding boxes for whatever YOLO found
            for r in results:
                display_frame = r.plot(img=display_frame)

            # --- UPDATED: FALLBACK FOR UNKNOWN FACES ---
            # ONLY run OpenCV if YOLO saw absolutely nothing
            if not yolo_detected_any_face:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=7, minSize=(40, 40))
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(display_frame, "UNKNOWN", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # 2. Check if the timeout has finished
            if time.time() - auth_timeout_start > AUTH_DURATION:
                if face_successfully_matched:
                    print("   -> AUTH COMPLETE. Resuming normal operations...")
                else:
                    print("   -> AUTH FAILED. Resuming anyway...")
                
                threading.Thread(target=send_command_to_pi, args=("running",), daemon=True).start()
                current_pi_state = "running"
                
                face_auth_mode = False
                face_successfully_matched = False

        # ==========================================
        # MODE 2: NORMAL SYMBOL DETECTION
        # ==========================================
        else:
            yolo_frame = frame.copy()
            yolo_frame[0:100, 0:180] = (0, 0, 0) # Blindfold
            total_frame_area = yolo_frame.shape[0] * yolo_frame.shape[1]
            
            results = symbol_model(yolo_frame, conf=0.6, imgsz=320, verbose=False)
            detected_action = "running" 
            
            for r in results:
                for box in r.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = r.names[class_id]
                    
                    box_w, box_h = float(box.xywh[0][2]), float(box.xywh[0][3])
                    area_percentage = (box_w * box_h) / total_frame_area
                    
                    if area_percentage < MAX_SIZE_THRESHOLD:
                        name_lower = class_name.lower()
                        
                        # --- ADDED: PRINT DETECTED SYMBOL TO TERMINAL ---
                        print(f"Symbol detected: {class_name}")
                        
                        # TRIGGER: Enter Auth Mode
                        if name_lower in ["fingerprint", "qr"]:
                            face_auth_mode = True
                            auth_timeout_start = time.time() 
                            print(f"   -> {name_lower.upper()} DETECTED. Switching to Face Auth...")
                            detected_action = "stop_for_auth" 
                            
                        # STANDARD MANEUVERS
                        elif name_lower == "recycle":
                            detected_action = "spin_360"
                        elif name_lower in ["danger", "stop"]:
                            detected_action = "stop_2s"
                        elif name_lower == "arrow left":
                            detected_action = "turn_left"
                        elif name_lower == "arrow right":
                            detected_action = "turn_right"
                        elif name_lower == "arrow up":
                            detected_action = "move_straight"

            # Send maneuver action if state changed and not entering auth mode
            if detected_action != current_pi_state and not face_auth_mode:
                threading.Thread(target=send_command_to_pi, args=(detected_action,), daemon=True).start()
                current_pi_state = detected_action

            # Draw symbol bounding boxes
            for r in results:
                display_frame = r.plot(img=display_frame)

        # Display Live Feed
        cv2.imshow("Laptop Brain - Low Latency", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            requests.get(f"{PI_COMMAND_URL}/stop") 
            break
        
    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()