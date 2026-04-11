import cv2
import time
from picamera2 import Picamera2
from ultralytics import YOLO

def main():
    # 1. Load the YOLOv5 Nano NCNN model
    print("Loading YOLOv5n NCNN model...")
    model = YOLO('yolov5nu_ncnn_model')
    
    # 2. Initialize Picamera2
    print("Initializing Picamera2...")
    picam2 = Picamera2()
    
    # CRITICAL FOR PI 4: Lower the resolution to 320x240 or 416x416. 
    # High resolutions will cause massive lag on the Pi 4 CPU.
    config = picam2.create_video_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    
    picam2.start()
    print("Camera started. Press 'q' to exit.")

    # FPS Calculation variables
    prev_time = 0

    try:
        while True:
            # Grab the current time to calculate FPS later
            current_time = time.time()

            # Capture frame and convert RGB to BGR for OpenCV
            frame_rgb = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # Run YOLOv5 inference. 
            # We lower the confidence slightly for the smaller model
            results = model(frame_bgr, conf=0.2)

            # Draw the boxes
            annotated_frame = results[0].plot()

            # Calculate and display the Frames Per Second (FPS)
            fps = 1 / (current_time - prev_time)
            prev_time = current_time
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Display the video window
            cv2.imshow("Raspberry Pi 4 - YOLOv5", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        print("Cleaning up...")
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()