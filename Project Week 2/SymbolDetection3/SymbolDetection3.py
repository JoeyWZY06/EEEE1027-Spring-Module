# ==============================================================================
# 1. INDUSTRIAL-STRENGTH 'imp' MOCK FOR PYTHON 3.12+ & TENSORFLOW COMPATIBILITY
# ==============================================================================
import sys
import types
import importlib.util

if 'imp' not in sys.modules:55
    imp = types.ModuleType('imp')
    imp.C_EXTENSION = 3
    imp.PY_SOURCE = 1
    imp.PY_COMPILED = 2
    
    def find_module(name, path=None):
        spec = importlib.util.find_spec(name, path)
        if spec is None:
            raise ImportError(f"No module named {name!r}")
        
        if spec.origin and spec.origin.endswith('.so'):
            return (None, spec.origin, ('.so', 'rb', imp.C_EXTENSION))
        elif spec.origin and spec.origin.endswith('.py'):
            return (None, spec.origin, ('.py', 'r', imp.PY_SOURCE))
        elif spec.origin and spec.origin.endswith('.pyc'):
            return (None, spec.origin, ('.pyc', 'rb', imp.PY_COMPILED))
        else:
            return (None, spec.origin, ('', 'r', imp.PY_SOURCE))
            
    def load_module(name, file, pathname, description):
        spec = importlib.util.spec_from_file_location(name, pathname)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    imp.find_module = find_module
    imp.load_module = load_module
    sys.modules['imp'] = imp

# ==============================================================================
# 2. STANDARD IMPORTS & GLOBALS
# ==============================================================================
import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from tensorflow.lite.python.interpreter import Interpreter

# --- System States ---
ACTIVE_SHORTCUT = "green"
robot_state = "FOLLOWING"  
pause_start_time = 0
cv_arrow_active = False

# --- Timings & Thresholds ---
PAUSE_DURATION = 10.0      
REVERSE_DURATION = 0.5     
FORWARD_DURATION = 0.5     
ML_COOLDOWN_TIME = 4.0     

# Vision Trigger Thresholds
SATURATION_THRESHOLD = 70 
VALUE_THRESHOLD = 50      
MIN_COLOR_AREA = 1800

# ==============================================================================
# 3. GPIO & MOTOR SETUP
# ==============================================================================
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

IN1, IN2, ENA = 26, 19, 20
IN3, IN4, ENB = 13, 6, 5
SERVO_PIN = 21  

GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)
servo.start(0)

GPIO.setup([IN1, IN2, IN3, IN4, ENA, ENB], GPIO.OUT)
pwm_a = GPIO.PWM(ENA, 1000)
pwm_b = GPIO.PWM(ENB, 1000)
pwm_a.start(0)
pwm_b.start(0)

# PID Constants & Speed
KP, KI, KD = 1.0, 0.001, 2.8
BASE_SPEED, TURN_SPEED = 32, 70
REVERSE_SPEED = 40         
FORWARD_SPEED = 40         

BLACK_THRESHOLD = 60 
LINE_COLORS = {
    "black": [(0, 0, 0), (180, 255, BLACK_THRESHOLD)], 
    "green": [(40, 100, 100), (75, 255, 255)]
}

def set_motor_speed(left_speed, right_speed):
    left_speed = max(min(left_speed, 100), -100)
    right_speed = max(min(right_speed, 100), -100)

    if left_speed >= 0:
        GPIO.output(IN1, GPIO.LOW); GPIO.output(IN2, GPIO.HIGH)
    else:
        GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
    pwm_a.ChangeDutyCycle(abs(left_speed))

    if right_speed >= 0:
        GPIO.output(IN3, GPIO.LOW); GPIO.output(IN4, GPIO.HIGH)
    else:
        GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
    pwm_b.ChangeDutyCycle(abs(right_speed))

def stop_motors():
    set_motor_speed(0, 0)

# ==============================================================================
# 4. ML MODEL SETUP (TENSORFLOW LITE)
# ==============================================================================
print("Loading TFLite model...")
MODEL_PATH = "model_unquant.tflite" 
LABEL_PATH = "labels.txt"

LABELS = []
try:
    with open(LABEL_PATH, "r") as f:
        for line in f.readlines():
            line = line.strip()
            if " " in line and line.split(" ", 1)[0].isdigit():
                LABELS.append(line.split(" ", 1)[1])
            else:
                LABELS.append(line)
except FileNotFoundError:
    print(f"WARNING: {LABEL_PATH} not found.")
    LABELS = ["Idle", "Arrow"]

interpreter = Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def run_ml_inference(frame_bgr):
    input_shape = input_details[0]['shape'] 
    height, width = input_shape[1], input_shape[2]
    
    img_resized = cv2.resize(frame_bgr, (width, height))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    input_data = np.expand_dims(img_rgb, axis=0).astype(np.float32)
    input_data = (input_data / 127.5) - 1.0 
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predictions = np.squeeze(output_data)
    
    top_index = predictions.argmax()
    confidence = predictions[top_index]
    label = LABELS[top_index] if top_index < len(LABELS) else "Unknown"
    
    return label, confidence

# ==============================================================================
# 5. VISION & OPENCV FUNCTIONS
# ==============================================================================
def get_line_mask(hsv_frame, color_name):
    lower = np.array(LINE_COLORS[color_name][0])
    upper = np.array(LINE_COLORS[color_name][1])
    return cv2.inRange(hsv_frame, lower, upper)

def process_arrow_mask(mask, color_name, frame):
    results = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            box_center_x = x + (w // 2)
            box_center_y = y + (h // 2)

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                mass_center_x = int(M["m10"] / M["m00"])
                mass_center_y = int(M["m01"] / M["m00"])

                direction = "Unknown"
                if w > h:
                    direction = "Right" if mass_center_x > box_center_x else "Left"
                else:
                    direction = "Down" if mass_center_y > box_center_y else "Up"

                label = f"{color_name} {direction}"
                results.append(label)
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.circle(frame, (box_center_x, box_center_y), 4, (0, 0, 255), -1) 
                cv2.circle(frame, (mass_center_x, mass_center_y), 4, (255, 0, 0), -1) 
    return results

def detect_arrows(frame, hsv):
    color_ranges = {
        "Green":  [(40, 50, 50), (85, 255, 255)],
        "Blue":   [(100, 100, 50), (130, 255, 255)],
        "Orange": [(10, 100, 100), (25, 255, 255)]
    }
    red_lower1, red_upper1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    red_lower2, red_upper2 = np.array([160, 100, 100]), np.array([180, 255, 255])

    detected_symbols = []
    for color_name, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        detected_symbols.extend(process_arrow_mask(mask, color_name, frame))

    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    detected_symbols.extend(process_arrow_mask(mask_red, "Red", frame))

    return detected_symbols

# ==============================================================================
# 6. CAMERA INIT & MAIN EXECUTION LOOP
# ==============================================================================
if __name__ == '__main__':
    print("Initializing Camera...")
    picam2 = Picamera2()
    config = picam2.create_video_configuration(main={"size": (320, 240)})
    picam2.configure(config)
    picam2.start()
    picam2.set_controls({"AwbMode": 6})
    
    print("Starting System... Click video window and press 'q' to quit.")
    
    last_error = 0
    integral = 0
    prev_frame_time = 0
    ml_cooldown_until = 0  
    
    # Track previous ML output to prevent it from disappearing
    last_ml_label = "SLEEPING"
    last_ml_conf = 0.0

    try:
        while True:
            request = picam2.capture_request()
            frame_rgb = request.make_array("main")
            request.release() 

            current_time = time.time()
            fps = 1 / (current_time - prev_frame_time) if prev_frame_time else 0
            prev_frame_time = current_time

            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            hsv_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            annotated_frame = frame_bgr.copy()
            
            height, width = annotated_frame.shape[:2]
            
            # --- EXPANDED VISION AREA ---
            # Now scans the top 70% of the screen instead of 40%
            color_vision_y_end = int(height * 0.70) 
            roi_y_start = int(height / 2.3) # Keep line following at same height

            # =========================================================
            # THE LIGHTWEIGHT TRIGGER: SATURATION + BRIGHTNESS TRACKING
            # =========================================================
            vision_roi = frame_bgr[0:color_vision_y_end, :]
            hsv_vision = cv2.cvtColor(vision_roi, cv2.COLOR_BGR2HSV)
            
            h_chan, s_chan, v_chan = cv2.split(hsv_vision)
            
            blurred_s = cv2.GaussianBlur(s_chan, (5, 5), 0)
            blurred_v = cv2.GaussianBlur(v_chan, (5, 5), 0)
            
            _, sat_mask = cv2.threshold(blurred_s, SATURATION_THRESHOLD, 255, cv2.THRESH_BINARY)
            _, val_mask = cv2.threshold(blurred_v, VALUE_THRESHOLD, 255, cv2.THRESH_BINARY)
            
            trigger_mask = cv2.bitwise_and(sat_mask, val_mask)
            kernel = np.ones((5,5), np.uint8)
            trigger_mask = cv2.morphologyEx(trigger_mask, cv2.MORPH_OPEN, kernel)
            
            # --- DRAW CONTOURS ON THE MASK VIEW ---
            # Convert mask to color so we can draw green lines on it
            trigger_display = cv2.cvtColor(trigger_mask, cv2.COLOR_GRAY2BGR)
            color_contours, _ = cv2.findContours(trigger_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            color_blob_detected = False
            if color_contours:
                largest_color_contour = max(color_contours, key=cv2.contourArea)
                if cv2.contourArea(largest_color_contour) > MIN_COLOR_AREA:
                    color_blob_detected = True
                    
                    # Draw thick green outline around the detected color blob on the threshold window
                    cv2.drawContours(trigger_display, [largest_color_contour], -1, (0, 255, 0), 3)

                    M_color = cv2.moments(largest_color_contour)
                    if M_color["m00"] != 0:
                        cx = int(M_color['m10'] / M_color['m00'])
                        cy = int(M_color['m01'] / M_color['m00'])
                        cv2.circle(annotated_frame, (cx, cy), 15, (255, 0, 255), 2)
                        cv2.putText(annotated_frame, "COLOR DETECTED", (cx + 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

            # ---------------------------------------------------------
            # STATE MACHINE
            # ---------------------------------------------------------
            if robot_state == "FOLLOWING":
                cv2.putText(annotated_frame, "STATE: FOLLOWING", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # A. Run ML ONLY if Color is Detected
                if color_blob_detected and current_time > ml_cooldown_until:
                    
                    # RUN TENSORFLOW
                    ml_label, confidence = run_ml_inference(frame_bgr)
                    
                    last_ml_label = ml_label
                    last_ml_conf = confidence
                    
                    is_idle = any(word in ml_label.lower() for word in ["idle", "background", "none"])

                    if not is_idle and confidence > 0.70:
                        robot_state = "PAUSED"
                        pause_start_time = current_time
                        cv_arrow_active = "arrow" in ml_label.lower()
                        print(f"[!] Target Acquired: {ml_label} ({int(confidence*100)}%)")
                
                elif current_time > ml_cooldown_until and not color_blob_detected:
                    last_ml_label = "SLEEPING"
                    last_ml_conf = 0.0

                # B. PID Line Following Logic
                use_shortcut = False
                contours = []

                if ACTIVE_SHORTCUT:
                    shortcut_mask = get_line_mask(hsv_full, ACTIVE_SHORTCUT)
                    shortcut_roi = shortcut_mask[roi_y_start:height, :]
                    shortcut_contours, _ = cv2.findContours(shortcut_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if shortcut_contours:
                        largest_shortcut = max(shortcut_contours, key=cv2.contourArea)
                        if cv2.contourArea(largest_shortcut) > 200:
                            contours = shortcut_contours
                            use_shortcut = True
                            cv2.putText(annotated_frame, f"Tracking: {ACTIVE_SHORTCUT}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if not use_shortcut:
                    full_black_mask = get_line_mask(hsv_full, "black")
                    roi = full_black_mask[roi_y_start:height, :]
                    contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.putText(annotated_frame, "Tracking: black", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                cv2.rectangle(annotated_frame, (0, roi_y_start), (width, height), (255, 0, 0), 2)

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
                        
                        cv2.circle(annotated_frame, (cx, roi_y_start + 20), 10, (0, 255, 0), -1)
                else:
                    if last_error > 0: 
                        set_motor_speed(TURN_SPEED, -TURN_SPEED)
                        time.sleep(0.05)
                    else: 
                        set_motor_speed(-TURN_SPEED, TURN_SPEED)
                        time.sleep(0.05)

            elif robot_state == "PAUSED":
                elapsed_time = current_time - pause_start_time
                # NOTICE: I completely removed the code here that was overwriting last_ml_label!
                
                # 1. Reverse Phase
                if elapsed_time < REVERSE_DURATION:
                    set_motor_speed(-REVERSE_SPEED, -REVERSE_SPEED)
                    cv2.putText(annotated_frame, "STATE: REVERSING", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                
                # 2. Stopped & Scanning Phase
                elif elapsed_time < PAUSE_DURATION:
                    stop_motors()
                    cv2.putText(annotated_frame, f"STATE: PAUSED ({int(PAUSE_DURATION - elapsed_time)}s)", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
                    if cv_arrow_active:
                        detected_arrows = detect_arrows(annotated_frame, hsv_full)
                        if detected_arrows:
                            cv2.putText(annotated_frame, f"CV ARROW: {detected_arrows[0]}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # 3. Drive Forward Phase
                elif elapsed_time < (PAUSE_DURATION + FORWARD_DURATION):
                    set_motor_speed(FORWARD_SPEED, FORWARD_SPEED)
                    cv2.putText(annotated_frame, "STATE: CROSSING", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                
                # 4. Resume Following
                else:
                    print("[!] Maneuver complete. Resuming.")
                    robot_state = "FOLLOWING"
                    cv_arrow_active = False
                    ml_cooldown_until = current_time + ML_COOLDOWN_TIME

            # --- DISPLAY RENDER ---
            cv2.rectangle(annotated_frame, (0, 0), (width, 40), (0, 0, 0), -1)
            cv2.putText(annotated_frame, f"FPS: {int(fps)}", (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            
            # Format ML Label Banner
            text_color = (0, 165, 255) if "SLEEP" not in last_ml_label else (150, 150, 150)
            if current_time <= ml_cooldown_until:
                status_str = f"ML: COOLDOWN ({int(ml_cooldown_until - current_time)}s)"
            else:
                status_str = f"ML: {last_ml_label} ({int(last_ml_conf*100)}%)" if last_ml_conf > 0 else f"ML: {last_ml_label}"
            
            cv2.putText(annotated_frame, status_str, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

            cv2.imshow('Robot Vision Core', annotated_frame)
            cv2.imshow('Color Trigger Mask', trigger_display) # Now displays a colored contour!

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("\nShutting down cleanly...")
        stop_motors()
        pwm_a.stop()
        pwm_b.stop()
        GPIO.cleanup()
        picam2.stop()
        picam2.close()
        cv2.destroyAllWindows()
        print("Done.")