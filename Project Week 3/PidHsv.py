import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from flask import Flask, Response

# ==========================================
# 1. Configuration (Change shortcut here!)
# ==========================================
# Options: "red", "green", "blue", "yellow", or None (to only follow black)
ACTIVE_SHORTCUT = "green"

# ==========================================
# 2. GPIO & Motor Configuration
# ==========================================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

IN1, IN2, ENA = 26, 19, 20
IN3, IN4, ENB = 13, 6, 5

SERVO_PIN = 21  # We are using GPIO 21
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)

GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)
pwm_a = GPIO.PWM(ENA, 1000)
pwm_b = GPIO.PWM(ENB, 1000)
pwm_a.start(0)
pwm_b.start(0)

# PID & Speed Constants
KP, KI, KD = 1.0, 0.001, 2.8
BASE_SPEED, TURN_SPEED = 60, 60
BLACK_THRESHOLD = 60 

# ==========================================
# 3. HSV Color Ranges
# ==========================================
# Format: (Hue 0-180, Saturation 0-255, Value 0-255)
COLOR_RANGES = {
    "black": [(0, 0, 0), (180, 255, BLACK_THRESHOLD)], 
    "red_1": [(0, 100, 100), (10, 255, 255)],   # Red lower bound (wrap around)
    "red_2": [(160, 100, 100), (180, 255, 255)], # Red upper bound
    "green": [(40, 100, 100), (75, 255, 255)],
    "blue":  [(100, 50, 50), (130, 255, 255)],
    "yellow":[(20, 100, 100), (40, 255, 255)]
}

def get_color_mask(hsv_frame, color_name):
    """Generates a binary mask for the specified color."""
    if color_name == "red":
        mask1 = cv2.inRange(hsv_frame, np.array(COLOR_RANGES["red_1"][0]), np.array(COLOR_RANGES["red_1"][1]))
        mask2 = cv2.inRange(hsv_frame, np.array(COLOR_RANGES["red_2"][0]), np.array(COLOR_RANGES["red_2"][1]))
        return cv2.bitwise_or(mask1, mask2)
    else:
        return cv2.inRange(hsv_frame, np.array(COLOR_RANGES[color_name][0]), np.array(COLOR_RANGES[color_name][1]))

# ==========================================
# 4. Motor and Servo Control Functions
# ==========================================
def set_motor_speed(left_speed, right_speed):
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

def set_angle(angle):
    duty =  1.65 + (angle / 18)
    GPIO.output(SERVO_PIN, True)
    servo.ChangeDutyCycle(duty)
    time.sleep(.2) 
    GPIO.output(SERVO_PIN, False)
    servo.ChangeDutyCycle(0)

# ==========================================
# 5. Flask & Camera Setup
# ==========================================
app = Flask(__name__)
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (320, 240)})
picam2.configure(config)
# Force AWB to Daylight to kill the yellow tint
picam2.set_controls({"AwbMode": 6})
picam2.start()

def generate_frames():
    last_error = 0
    integral = 0
    
    # FPS Variables
    prev_frame_time = 0
    new_frame_time = 0
    
    while True:
        # Capture frame
        request = picam2.capture_request()
        frame = request.make_array("main")
        request.release()

        # FPS Calculation
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time) if prev_frame_time else 0
        prev_frame_time = new_frame_time

        # Image Processing (HSV Conversion)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        
        height, width = frame_bgr.shape[:2]
        roi_y_start = int(height / 2.3)
        
        use_shortcut = False
        thresh = None
        contours = []

        # Step A: Try to find the shortcut line first
        if ACTIVE_SHORTCUT:
            shortcut_mask = get_color_mask(hsv, ACTIVE_SHORTCUT)
            shortcut_roi = shortcut_mask[roi_y_start:height, :]
            shortcut_contours, _ = cv2.findContours(shortcut_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check if we see a substantial line (filter out noise)
            if shortcut_contours:
                largest_shortcut = max(shortcut_contours, key=cv2.contourArea)
                if cv2.contourArea(largest_shortcut) > 200:
                    thresh = shortcut_mask
                    contours = shortcut_contours
                    use_shortcut = True
                    cv2.putText(frame_bgr, f"Tracking: {ACTIVE_SHORTCUT}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # Step B: Fallback to Black Main Line if no shortcut is found
        if not use_shortcut:
            thresh = get_color_mask(hsv, "black")
            roi = thresh[roi_y_start:height, :]
            contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.putText(frame_bgr, "Tracking: black", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Visual Guides
        cv2.rectangle(frame_bgr, (0, roi_y_start), (width, height), (255, 0, 0), 2)

        # Step C: PID Logic & Motor Control
        if contours:
            c = max(contours, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                error = cx - (width // 2)
                
                # PID Calculations
                P = error
                integral += error
                D = error - last_error
                last_error = error
                correction = (KP * P) + (KI * integral) + (KD * D)
                
                # Motor logic
                turn_threshold = (width // 2) * 0.7
                if abs(error) > turn_threshold:
                    if error > 0: set_motor_speed(TURN_SPEED, -TURN_SPEED)
                    else: set_motor_speed(-TURN_SPEED, TURN_SPEED)
                else:
                    set_motor_speed(BASE_SPEED + correction, BASE_SPEED - correction)
                
                cv2.circle(frame_bgr, (cx, roi_y_start + 20), 10, (0, 255, 0), -1)
        else:
            # Lost line logic
            if last_error > 0: 
                set_motor_speed(TURN_SPEED, -TURN_SPEED)
                turn_right(70, 0.1)
            else: 
                set_motor_speed(-TURN_SPEED, TURN_SPEED)
                turn_left(70, 0.1)

        # FPS counter
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame_bgr, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Create the side-by-side view for the web stream
        thresh_3_channel = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        combined_view = np.hstack((frame_bgr, thresh_3_channel))
        
        # Encode for Web
        _, buffer = cv2.imencode('.jpg', combined_view, [cv2.IMWRITE_JPEG_QUALITY, 35])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        print("Server starting at http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutting down...")
        pwm_a.stop()
        pwm_b.stop()
        GPIO.cleanup()