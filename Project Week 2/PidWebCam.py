import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from flask import Flask, Response

# ==========================================
# 1. GPIO & Motor Configuration
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
pwm_a = GPIO.PWM(ENA, 400)
pwm_b = GPIO.PWM(ENB, 400)
pwm_a.start(0)
pwm_b.start(0)


# PID & Speed Constants
KP, KI, KD = 2.3, 0.0001, 2.0
BASE_SPEED, TURN_SPEED = 100, 100
BLACK_THRESHOLD = 60 

# ==========================================
# 2. Motor Control and Servo Control
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


# Turn left function
def turn_left(speed, duration):
    # Left Motor Logic
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    
    # Right Motor Logic
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)
    
    # Set Speed
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)

    # Wait for the specified duration
    time.sleep(duration)
    
# Turn right function
def turn_right(speed, duration):
    # Left Motor Logic
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    
    # Right Motor Logic
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    
    # Set Speed
    pwm_a.ChangeDutyCycle(speed)
    pwm_b.ChangeDutyCycle(speed)

    # Wait for the specified duration
    time.sleep(duration)

# control servo angle (0,90,180)
def set_angle(angle):
 
    duty =  1.65 + (angle / 18)
    
    # Send the signal
    GPIO.output(SERVO_PIN, True)
    servo.ChangeDutyCycle(duty)
    
    # Wait for the servo to physically move there
    time.sleep(.2) 
    
    # Stop sending signal to prevent "jittering"
    GPIO.output(SERVO_PIN, False)
    servo.ChangeDutyCycle(0)

# ==========================================
# 3. Flask & Camera Setup
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
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Image Processing (PID Logic)
        # Convert RGB to BGR for OpenCV and then Gray for processing
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, BLACK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
        
        height, width = thresh.shape
        roi_y_start = int(height / 2.8)
        roi = thresh[roi_y_start:height, :] 
        contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Visual Guides
        cv2.rectangle(frame_bgr, (0, roi_y_start), (width, height), (255, 0, 0), 2)

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
            """
            set_motor_speed(0,0)
            time.sleep(0.1)
            if last_error > 0:
                set_angle(0)
                turn_right(50, 0.45)
            else:
                set_angle(180)
                turn_left(50, 0.45)
            set_angle(86)
            """
        # FPS counter
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame_bgr, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        # Create the side-by-side view for the web stream
        thresh_3_channel = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        combined_view = np.hstack((frame_bgr, thresh_3_channel))
        
        # Encode for Web (Lower quality = lower latency over SSH)
        _, buffer = cv2.imencode('.jpg', combined_view, [cv2.IMWRITE_JPEG_QUALITY, 35])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        print("Server starting at http://<pi-ip>:5000")
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pwm_a.stop()
        pwm_b.stop()
        GPIO.cleanup()