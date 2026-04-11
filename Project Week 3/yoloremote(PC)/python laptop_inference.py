import cv2
import threading
import torch
from ultralytics import YOLO

# !!! IMPORTANT !!!
# Replace this with your Raspberry Pi's actual IP address
PI_STREAM_URL = "http://10.51.237.198:5000/video_feed"

class VideoStream:
    """
    Background thread that constantly pulls the latest frame from the stream.
    This prevents OpenCV's internal buffer from creating 'ghosting' lag.
    """
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        # Set buffer size to 1 to minimize latency at the OS level
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            grabbed, frame = self.cap.read()
            if not grabbed:
                self.stop()
            else:
                # Thread-safe update of the latest frame
                self.grabbed, self.frame = grabbed, frame

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

def main():
    # 1. Select Device (GPU is significantly faster than CPU)
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 2. Load the YOLO model
    print("Loading YOLO model...")
    # NOTE: For maximum speed, export 'my_model.pt' to OpenVINO or TensorRT
    model = YOLO('my_model_3.pt')
    model.to(device)

    # 3. Initialize the threaded stream
    print(f"Connecting to Pi stream at {PI_STREAM_URL}...")
    vs = VideoStream(PI_STREAM_URL).start()
    print("Stream connected! Press 'q' to quit.")
    
    while True:
        frame = vs.read()
        if frame is None:
            continue
        
        # 4. Optimized Inference
        # imgsz=320: Matches your Pi's 360px feed closely, reducing resize overhead.
        # stream=True: Uses a generator to handle frames more efficiently.
        # persist=True: Helps tracking if you use model.track()
        results = model(frame, conf=0.4, imgsz=320, verbose=False, stream=True)

        # 5. Process and Display
        for r in results:
            annotated_frame = r.plot()
            cv2.imshow("Laptop Brain - Low Latency", annotated_frame)

        # Break loop on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    vs.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()