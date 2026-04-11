import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
import threading
from picamera2 import Picamera2
from flask import Flask, Response

# ==========================================
# 1. Configuration 
# ==========================================
# Options: "yellow", "red", or "yellow_red" (to track either/both as secondary)
ACTIVE_SHORTCUT = "yellow_red" 
robot_state = "running" # Global state controlled by Laptop
is_maneuvering = False  # Prevents YOLO from spamming commands while moving
line_detected = False   # Shared flag to tell the maneuver thread if a line is visible

# ==========================================
# 2. GPIO & Motor Configuration
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
pwm_a = GPIO.PWM(ENA, 500)
pwm_b = GPIO.PWM(ENB, 500)
pwm_a.start(0)
pwm_b.start(0)

KP, KI, KD = 1.0, 0.001, 3.0
BASE_SPEED, TURN_SPEED = 27, 36
BLACK_THRESHOLD = 60 

# ==========================================
# 3. HSV Color Ranges
# ==========================================
COLOR_RANGES = {
    "black": [(0, 0, 0), (180, 255, BLACK_THRESHOLD)], 
    "red_1": [(0, 100, 100), (10, 255, 255)],    
    "red_2": [(160, 100, 100), (180, 255, 255)], 
    "yellow":[(20, 100, 100), (40, 255, 255)]
}

def get_color_mask(hsv_frame, color_name):
    if color_name == "red":
        mask1 = cv2.inRange(hsv_frame, np.array(COLOR_RANGES["red_1"][0]), np.array(COLOR_RANGES["red_1"][1]))
        mask2 = cv2.inRange(hsv_frame, np.array(COLOR_RANGES["red_2"][0]), np.array(COLOR_RANGES["red_2"][1]))
        return cv2.bitwise_or(mask1, mask2)
    else:
        return cv2.inRange(hsv_frame, np.array(COLOR_RANGES[color_name][0]), np.array(COLOR_RANGES[color_name][1]))

# ==========================================
# 4. Motor Control Functions
# ==========================================
def set_motor_speed(left_speed, right_speed):
    left_speed = max(min(left_speed, 100), -100)
    right_speed = max(min(right_speed, 100), -100)

    if left_speed >= 0:
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
    else:
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
    pwm_a.ChangeDutyCycle(abs(left_speed))

    if right_speed >= 0:
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)
    else:
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
    pwm_b.ChangeDutyCycle(abs(right_speed))

def turn_left(speed, duration):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)
    time.sleep(duration)
    
def turn_right(speed, duration):
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)
    time.sleep(duration)

def execute_maneuver(action):
    """Runs in the background so it doesn't freeze the camera feed!"""
    global is_maneuvering, robot_state, line_detected
    is_maneuvering = True
    
    if action == "stop_for_auth":
        set_motor_speed(0, 0)
        # Return early. DO NOT set is_maneuvering to False.
        # This keeps the Pi frozen until laptop overrides it.
        return 
        
    elif action == "stop_2s":
        set_motor_speed(0, 0)
        time.sleep(2.0)
        set_motor_speed(40, 40)
        time.sleep(0.6)
        
    elif action == "spin_360":
        turn_left(70, 2.0)
        set_motor_speed(35, 35)
        time.sleep(0.5) 
        
    elif action == "turn_left":
        # 1. Turn blindly for 0.4s to get off the current line
        turn_left(70, 0.25)
        # 2. Loop and wait until the camera thread sees a line again
        while not line_detected:
            time.sleep(0.01)
        # 3. Lock it in: bump forward slightly to align before PID takes over
        set_motor_speed(35, 35)
        time.sleep(0.2) 
        
    elif action == "turn_right":
        # 1. Turn blindly for 0.4s to get off the current line
        turn_right(70, 0.25)
        # 2. Loop and wait until the camera thread sees a line again
        while not line_detected:
            time.sleep(0.01)
        # 3. Lock it in: bump forward slightly to align before PID takes over
        set_motor_speed(35, 35)
        time.sleep(0.2)
        
    elif action == "move_straight":
        set_motor_speed(35, 35)
        time.sleep(0.6)

    # Resume normal line following
    is_maneuvering = False
    robot_state = "running"

# ==========================================
# 5. Flask & Dual-Thread Camera Setup
# ==========================================
app = Flask(__name__)
latest_jpeg = None
lock = threading.Lock() 

def run_camera_thread():
    global latest_jpeg, robot_state, is_maneuvering, line_detected
    
    print("Warming up camera...")
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (320, 320)})
    picam2.configure(config)
    picam2.set_controls({"AwbMode": 6})
    picam2.start()
    
    last_error = 0
    integral = 0
    prev_frame_time = 0
    
    while True:
        request = picam2.capture_request()
        frame = request.make_array("main")
        request.release()

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time else 0
        prev_frame_time = new_frame_time

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        height, width = frame_bgr.shape[:2]
        roi_y_start = int(height / 1.7)
        
        use_shortcut = False
        thresh = None
        contours = []

        if ACTIVE_SHORTCUT:
            if ACTIVE_SHORTCUT == "yellow_red":
                mask_y = get_color_mask(hsv, "yellow")
                mask_r = get_color_mask(hsv, "red")
                shortcut_mask = cv2.bitwise_or(mask_y, mask_r)
                display_text = "yellow+red"
            else:
                shortcut_mask = get_color_mask(hsv, ACTIVE_SHORTCUT)
                display_text = ACTIVE_SHORTCUT

            shortcut_roi = shortcut_mask[roi_y_start:height, :]
            shortcut_contours, _ = cv2.findContours(shortcut_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            if shortcut_contours:
                largest_shortcut = max(shortcut_contours, key=cv2.contourArea)
                if cv2.contourArea(largest_shortcut) > 200:
                    thresh = shortcut_mask
                    contours = shortcut_contours
                    use_shortcut = True
                    cv2.putText(frame_bgr, f"Tracking: {display_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        if not use_shortcut:
            thresh = get_color_mask(hsv, "black")
            roi = thresh[roi_y_start:height, :]
            contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Update the global flag so the maneuver thread knows a line is visible
            line_detected = len(contours) > 0
            
            cv2.putText(frame_bgr, "Tracking: black", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.rectangle(frame_bgr, (0, roi_y_start), (width, height), (255, 0, 0), 2)

        if is_maneuvering:
            cv2.putText(frame_bgr, f"ACTION: {robot_state}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        elif robot_state == "stop":
            set_motor_speed(0, 0)
        else:
            if contours:
                c = max(contours, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    error = cx - (width // 2)
                    
                    P = error
                    integral += error
                    D = error - last_error
                    last_error = error
                    correction = (KP * P) + (KI * integral) + (KD * D)
                    
                    turn_threshold = (width // 2) * 0.7
                    if abs(error) > turn_threshold:
                        if error > 0: set_motor_speed(TURN_SPEED, -TURN_SPEED)
                        else: set_motor_speed(-TURN_SPEED, TURN_SPEED)
                    else:
                        set_motor_speed(BASE_SPEED + correction, BASE_SPEED - correction)
                    
                    cv2.circle(frame_bgr, (cx, roi_y_start + 20), 10, (0, 255, 0), -1)
            else:
                if last_error > 0: 
                    set_motor_speed(TURN_SPEED, -TURN_SPEED)
                    turn_right(70, 0.2)
                else: 
                    set_motor_speed(-TURN_SPEED, TURN_SPEED)
                    turn_left(70, 0.2)

        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame_bgr, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        
        _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 55])
        
        with lock:
            latest_jpeg = buffer.tobytes()
            
        time.sleep(0.005)

def generate_frames():
    global latest_jpeg
    while True:
        with lock:
            if latest_jpeg is None:
                time.sleep(0.01)
                continue
            frame_bytes = latest_jpeg 
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/command/<action>')
def handle_command(action):
    global robot_state, is_maneuvering
    
    if action == "running":
        # Force override: if laptop commands running, break out of auth mode lock
        if robot_state == "stop_for_auth" or not is_maneuvering:
            is_maneuvering = False
            robot_state = "running"
    elif not is_maneuvering: 
        robot_state = action
        threading.Thread(target=execute_maneuver, args=(action,), daemon=True).start()
        
    return f"Command received: {action}"

@app.route('/')
def index():
    return "Robot Server Online. Video at /video_feed"

if __name__ == '__main__':
    try:
        cam_thread = threading.Thread(target=run_camera_thread, daemon=True)
        cam_thread.start()
        
        print("Server starting at http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutting down...")
        pwm_a.stop()
        pwm_b.stop()
        GPIO.cleanup()