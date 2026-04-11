import cv2
import time
from flask import Flask, Response
from picamera2 import Picamera2

app = Flask(__name__)
picam2 = Picamera2()

# 1. Lower resolution = Lower latency
# 640x360 is a good "sweet spot" for SSH streaming
config = picam2.create_video_configuration(main={"size": (640, 360)})
picam2.configure(config)
picam2.start()

# 2. Fix the Yellow Tint
# Modes: 1=Auto, 2=Incandescent, 3=Tungsten, 4=Fluorescent, 5=Indoor, 6=Daylight
# '6' (Daylight) usually removes the indoor yellow 'warmth'
picam2.set_controls({"AwbMode": 6}) 

def generate_frames():
    while True:
        # 3. Pull frame directly from the camera buffer (fastest way)
        request = picam2.capture_request()
        frame = request.make_array("main")
        request.release() # Release immediately to keep latency low

        # 4. Correct the BGR/RGB swap
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 5. Lower JPEG quality to 40% to speed up transmission over SSH
        _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 40])
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # threaded=True allows the web server to be more responsive
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)