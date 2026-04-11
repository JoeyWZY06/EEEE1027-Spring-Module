import cv2
from flask import Flask, Response
from picamera2 import Picamera2

app = Flask(__name__)

# Initialize the camera
print("Warming up camera...")
picam2 = Picamera2()
# IMPROVEMENT 1: Force BGR888 format right at the source to skip cv2.cvtColor
config = picam2.create_video_configuration(main={"size": (360, 360), "format": "BGR888"})
picam2.configure(config)
picam2.start()

def generate_frames():
    """Continuously capture frames and encode them for streaming."""
    while True:
        # Capture directly in BGR.
        frame_bgr = picam2.capture_array()
        
        # IMPROVEMENT 2: Compress the frame harder to speed up Wi-Fi transmission
        # Default is 95. 70 provides massive speed gains with minimal visual loss.
        success, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not success:
            continue
            
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in the MJPEG format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Route that serves the video stream."""
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # host='0.0.0.0' allows your laptop to connect to the Pi
    # threaded=True allows multiple connections without crashing
    print("Starting video stream on port 5000...")
    app.run(host='0.0.0.0', port=5000, threaded=True)