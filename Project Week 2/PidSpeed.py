import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from flask import Flask, Response
import threading

# ==========================================
# 1. GPIO & Motor Configuration
# ==========================================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

IN1, IN2, ENA = 26, 19, 20
IN3, IN4, ENB = 13, 6, 5

SERVO_PIN = 21
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)

GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)
pwm_a = GPIO.PWM(ENA, 400)
pwm_b = GPIO.PWM(ENB, 400)
pwm_a.start(0)
pwm_b.start(0)

# --- Tuned Constants ---
KP, KI, KD = 2.5, 0.0001, 0.1  # WARNING: KD/KI likely need retuning with dt added
BASE_SPEED = 100                
TURN_SPEED = 100
BLACK_THRESHOLD = 60 

# ==========================================
# 2. Motor & Servo Control
# ==========================================
def set_motor_speed(left_speed, right_speed):
    # Cap speeds between -100 and 100
    left_speed = max(min(left_speed, 100), -100)
    right_speed = max(min(right_speed, 100), -100)

    # Left Motor
    if left_speed >= 0:
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
    else:
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
    pwm_a.ChangeDutyCycle(abs(left_speed))

    # Right Motor
    if right_speed >= 0:
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)
    else:
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
    pwm_b.ChangeDutyCycle(abs(right_speed))

def set_angle(angle):
    duty = 1.65 + (angle / 18)
    GPIO.output(SERVO_PIN, True)
    servo.ChangeDutyCycle(duty)
    time.sleep(0.2) 
    GPIO.output(SERVO_PIN, False)
    servo.ChangeDutyCycle(0)

# ==========================================
# 3. Global State for Threading
# ==========================================
latest_frame_encoded = None
frame_lock = threading.Lock()
running = True

# ==========================================
# 4. Main Robot Control Thread
# ==========================================
def robot_control_loop():
    global latest_frame_encoded, running

    # Initialize Camera
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.set_controls({"AwbMode": 6}) # Daylight AWB
    picam2.start()

    last_error = 0
    integral = 0
    prev_time = time.time()
    
    try:
        while running:
            # Capture frame
            request = picam2.capture_request()
            frame = request.make_array("main")
            request.release()

            # Time and FPS Calculation
            current_time = time.time()
            dt = current_time - prev_time
            if dt == 0: dt = 0.001 # Prevent divide by zero
            fps = 1 / dt
            prev_time = current_time

            # 1. EARLY CROPPING (Saves CPU)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            height, width = frame_bgr.shape[:2]
            roi_y_start = int(height / 2.8)
            
            # Slice only the bottom portion of the image to process
            roi_bgr = frame_bgr[roi_y_start:height, :]
            gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Draw visual guide on the main frame
            cv2.rectangle(frame_bgr, (0, roi_y_start), (width, height), (255, 0, 0), 2)

            if contours:
                c = max(contours, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    error = cx - (width // 2)
                    
                    # 2. TIME-BASED PID
                    P = error
                    integral += error * dt
                    integral = max(min(integral, 1000), -1000) # Anti-windup
                    D = (error - last_error) / dt
                    last_error = error
                    
                    correction = (KP * P) + (KI * integral) + (KD * D)
                    
                    # Motor logic
                    turn_threshold = (width // 2) * 0.7
                    if abs(error) > turn_threshold:
                        # Sharp turn
                        if error > 0: set_motor_speed(TURN_SPEED, -TURN_SPEED)
                        else: set_motor_speed(-TURN_SPEED, TURN_SPEED)
                    else:
                        # Normal PID steering
                        set_motor_speed(BASE_SPEED + correction, BASE_SPEED - correction)
                    
                    cv2.circle(frame_bgr, (cx, roi_y_start + 20), 10, (0, 255, 0), -1)
            else:
                # 3. NON-BLOCKING LOST LINE LOGIC
                # Notice there is no time.sleep() here. It just sets the motors and immediately 
                # loops back to check the camera again.
                if last_error > 0: 
                    set_motor_speed(TURN_SPEED, -TURN_SPEED)
                else: 
                    set_motor_speed(-TURN_SPEED, TURN_SPEED)

            # Add FPS to image
            cv2.putText(frame_bgr, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # Create side-by-side view
            thresh_3_channel = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            
            # We must pad the thresh image back to full height to stack it horizontally 
            # with the full frame_bgr, or just stack it with the ROI.
            # Let's stack it with the ROI to keep the stream lightweight:
            combined_view = np.hstack((roi_bgr, thresh_3_channel))
            
            # Encode for Web 
            _, buffer = cv2.imencode('.jpg', combined_view, [cv2.IMWRITE_JPEG_QUALITY, 40])
            
            # Safely update the global frame for Flask
            with frame_lock:
                latest_frame_encoded = buffer.tobytes()

    except Exception as e:
        print(f"Robot thread error: {e}")
    finally:
        picam2.stop()
        set_motor_speed(0, 0)

# ==========================================
# 5. Flask Web Server
# ==========================================
app = Flask(__name__)

def generate_web_stream():
    global latest_frame_encoded
    while True:
        with frame_lock:
            if latest_frame_encoded is None:
                continue
            frame_data = latest_frame_encoded
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        
        # Small sleep to prevent maxing out the CPU on the web server thread
        time.sleep(0.03) 

@app.route('/')
def index():
    return Response(generate_web_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Start the robot control in a background thread
    robot_thread = threading.Thread(target=robot_control_loop, daemon=True)
    robot_thread.start()

    try:
        print("Server starting at http://<pi-ip>:5000")
        # Run Flask on the main thread
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        running = False
        robot_thread.join(timeout=2.0)
        set_motor_speed(0, 0)
        pwm_a.stop()
        pwm_b.stop()
        GPIO.cleanup()