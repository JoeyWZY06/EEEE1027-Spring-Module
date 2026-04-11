import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from flask import Flask, Response, render_template_string, jsonify

# ==========================================
# 1. Configuration & Global State
# ==========================================
ACTIVE_SHORTCUT = "green"

# --- NEW: Expanded State Machine Variables ---
robot_state = "FOLLOWING"  
align_start_time = 0
pause_start_time = 0
cooldown_start_time = 0

current_ml_prediction = "Scanning..." # Stores what the browser tells Python

# ==========================================
# 2. GPIO & Motor Configuration (UNCHANGED)
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
pwm_a = GPIO.PWM(ENA, 1000)
pwm_b = GPIO.PWM(ENB, 1000)
pwm_a.start(0)
pwm_b.start(0)

# PID & Speed Constants
KP, KI, KD = 1.0, 0.001, 2.8
BASE_SPEED, TURN_SPEED = 32, 70
BLACK_THRESHOLD = 60 

SATURATION_THRESHOLD = 60 

COLOR_RANGES = {
    "black": [(0, 0, 0), (180, 255, BLACK_THRESHOLD)], 
    "red_1": [(0, 100, 100), (10, 255, 255)],
    "red_2": [(160, 100, 100), (180, 255, 255)],
    "green": [(40, 100, 100), (75, 255, 255)],
    "blue":  [(100, 50, 50), (130, 255, 255)],
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
# 3. Motor Control Functions (UNCHANGED)
# ==========================================
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

def turn_left(speed, duration):
    GPIO.output(IN1, GPIO.HIGH); GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW); GPIO.output(IN4, GPIO.HIGH)
    pwm_a.ChangeDutyCycle(speed); pwm_b.ChangeDutyCycle(speed)
    time.sleep(duration)
    
def turn_right(speed, duration):
    GPIO.output(IN1, GPIO.LOW); GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH); GPIO.output(IN4, GPIO.LOW)
    pwm_a.ChangeDutyCycle(speed); pwm_b.ChangeDutyCycle(speed)
    time.sleep(duration)

def stop_motors():
    set_motor_speed(0, 0)

# ==========================================
# 4. OpenCV Arrow Detection Logic (UNCHANGED)
# ==========================================
def process_mask(mask, color_name, frame):
    results = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) > 1200:
            x, y, w, h = cv2.boundingRect(cnt)
            box_center_x = x + (w // 2); box_center_y = y + (h // 2)

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                mass_center_x = int(M["m10"] / M["m00"])
                mass_center_y = int(M["m01"] / M["m00"])

                direction = "Unknown"
                if w > h: direction = "Right" if mass_center_x > box_center_x else "Left"
                else: direction = "Down" if mass_center_y > box_center_y else "Up"

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
        detected_symbols.extend(process_mask(mask, color_name, frame))

    mask_red1 = cv2.inRange(hsv, red_lower1, red_upper1)
    mask_red2 = cv2.inRange(hsv, red_lower2, red_upper2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    detected_symbols.extend(process_mask(mask_red, "Red", frame))

    return detected_symbols

# ==========================================
# 5. Embedded HTML (ML Pipeline - UPDATED API)
# ==========================================
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Robot Control Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.3.1/dist/tf.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@0.8/dist/teachablemachine-image.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; background-color: #222; color: white; margin-top: 20px; }
        #video-container { border: 4px solid #444; display: inline-block; border-radius: 8px; overflow: hidden; margin-bottom: 20px;}
        #status-banner { font-size: 24px; font-weight: bold; height: 35px; }
        #label-container { display: flex; flex-direction: column; align-items: center; }
        .prediction { margin: 5px; padding: 10px; background-color: #333; border-radius: 5px; width: 300px; }
    </style>
</head>
<body>
    <h2>Live Vision & ML Inference</h2>
    <div id="status-banner" style="color: yellow;">Loading ML Model...</div>
    
    <div id="video-container">
        <img id="videoStream" src="/video_feed" height="240" crossorigin="anonymous" />
    </div>
    
    <div id="label-container">Initializing...</div>

    <script>
        const URL = "/"; 
        let model, labelContainer, statusBanner, maxPredictions;
        const imgElement = document.getElementById("videoStream");

        async function init() {
            const modelURL = URL + "model.json";
            const metadataURL = URL + "metadata.json";

            try {
                model = await tmImage.load(modelURL, metadataURL);
                maxPredictions = model.getTotalClasses();
                labelContainer = document.getElementById("label-container");
                statusBanner = document.getElementById("status-banner");
                
                setInterval(processFrame, 150);
            } catch (error) {
                statusBanner.innerHTML = "Error loading Teachable Machine model!";
                statusBanner.style.color = "red";
                console.error(error);
            }
        }

        async function processFrame() {
            if (!imgElement.complete || imgElement.naturalWidth === 0) return;

            try {
                const predictions = await model.predict(imgElement);
                let bestPrediction = predictions[0];
                
                for (let i = 1; i < maxPredictions; i++) {
                    if (predictions[i].probability > bestPrediction.probability) {
                        bestPrediction = predictions[i];
                    }
                }

                // Update UI Elements
                labelContainer.innerHTML = "";
                for (let i = 0; i < maxPredictions; i++) {
                    let div = document.createElement("div");
                    div.className = "prediction";
                    const prob = Math.round(predictions[i].probability * 100);
                    div.innerHTML = predictions[i].className + ": " + prob + "%";
                    
                    if (predictions[i].className === bestPrediction.className) {
                        div.style.color = "#ffaa00";
                        div.style.border = "1px solid #ffaa00";
                    }
                    labelContainer.appendChild(div);
                }

                // --- NEW: JUST SEND PREDICTION TO PYTHON ---
                const confidenceThreshold = 0.85;
                const className = bestPrediction.className.toLowerCase();

                if (className !== "idle" && !className.includes("background") && bestPrediction.probability > confidenceThreshold) {
                    statusBanner.innerHTML = "High Confidence Detection!";
                    statusBanner.style.color = "#00ff00";
                    // Send to Python
                    fetch('/update_prediction/' + encodeURIComponent(bestPrediction.className));
                } else {
                    statusBanner.innerHTML = "Scanning...";
                    statusBanner.style.color = "#ffaa00";
                }

            } catch (e) {
                console.log("Error processing frame", e);
            }
        }
        init();
    </script>
</body>
</html>
"""

# ==========================================
# 6. Flask & Camera Setup
# ==========================================
print("Initializing Camera...")
app = Flask(__name__, static_folder='.', static_url_path='/')
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (320, 240)})
picam2.configure(config)
picam2.set_controls({"AwbMode": 6})
picam2.start()

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

# --- NEW: Updates Python variables so the video feed can display the ML results ---
@app.route('/update_prediction/<label>')
def update_prediction(label):
    global current_ml_prediction
    current_ml_prediction = label
    return jsonify({"status": "ok"})

# ==========================================
# 7. MAIN LOGIC LOOP (Combined Dynamic Parking & PID)
# ==========================================
def generate_frames():
    global robot_state, align_start_time, pause_start_time, cooldown_start_time, current_ml_prediction
    
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
        hsv_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        height, width = frame_bgr.shape[:2]
        
        # --- SPLIT VISION ZONES ---
        roi_y_start = int(height / 2.3)
        vision_roi = frame_bgr[0:roi_y_start, :]
        vision_h = vision_roi.shape[0]

        # ---------------------------------------------------------
        # A. SATURATION CHECK (Look for object in top half)
        # ---------------------------------------------------------
        hsv_vision = cv2.cvtColor(vision_roi, cv2.COLOR_BGR2HSV)
        _, s_chan, _ = cv2.split(hsv_vision)
        blurred_s = cv2.GaussianBlur(s_chan, (5, 5), 0)
        
        _, sat_mask = cv2.threshold(blurred_s, SATURATION_THRESHOLD, 255, cv2.THRESH_BINARY)
        kernel = np.ones((5,5), np.uint8)
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel)
        
        M_sat = cv2.moments(sat_mask)
        color_area = M_sat['m00'] / 255.0

        """"
        cv2.imshow("1. Saturation Color Mask", sat_mask)
        cv2.imshow("2. Main Camera View", frame_bgr)
        cv2.waitKey(1)
        """
        
        # ---------------------------------------------------------
        # B. STATE MACHINE (Dynamic Parking)
        # ---------------------------------------------------------
        current_time = time.time()

        if robot_state == "FOLLOWING":
            # Start parking if we see a saturated blob (Scaled down threshold for 320x240 resolution)
            if color_area > 800: 
                robot_state = "ALIGNING"
                align_start_time = current_time

        elif robot_state == "ALIGNING":
            if current_time - align_start_time > 3.5:
                # Timeout safeguard: if it takes too long, it was a glitch.
                robot_state = "COOLDOWN"
                cooldown_start_time = current_time
            elif color_area > 400:
                # Calculate the vertical center (cy) of the color blob
                cy = int(M_sat['m01'] / M_sat['m00'])
                
                # The "Perfect Parking" zone (Matches your exact boundaries)
                top_boundary = vision_h * 0.35
                bottom_boundary = vision_h * 0.65
                
                if cy > bottom_boundary:
                    set_motor_speed(-35, -35) # Reverse
                    cv2.putText(frame_bgr, "ALIGNING: REVERSING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                elif cy < top_boundary:
                    set_motor_speed(35, 35)   # Creep Forward
                    cv2.putText(frame_bgr, "ALIGNING: CREEPING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
                else:
                    # Target is perfectly centered!
                    stop_motors()
                    robot_state = "PAUSED"
                    pause_start_time = current_time
                    current_ml_prediction = "Scanning..." 
            else:
                robot_state = "COOLDOWN"
                cooldown_start_time = current_time

        elif robot_state == "PAUSED":
            stop_motors()
            time_left = 6.0 - (current_time - pause_start_time)
            
            # Show TM Prediction on the camera feed
            cv2.putText(frame_bgr, f"TM SAYS: {current_ml_prediction}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame_bgr, f"PAUSED: {time_left:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

            if time_left <= 0:
                robot_state = "COOLDOWN"
                cooldown_start_time = current_time

        elif robot_state == "COOLDOWN":
            cv2.putText(frame_bgr, "COOLDOWN: IGNORING SYMBOLS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255), 2)
            if current_time - cooldown_start_time > 2.0:
                robot_state = "FOLLOWING"
                last_error = 0 # Reset PID to prevent jumping

        # ---------------------------------------------------------
        # C. PID LINE FOLLOWING (Runs only when not Parking/Paused)
        # ---------------------------------------------------------
        if robot_state in ["FOLLOWING", "COOLDOWN"]:
            use_shortcut = False
            contours = []

            # Step A: Find Shortcut
            if ACTIVE_SHORTCUT:
                shortcut_mask = get_color_mask(hsv_full, ACTIVE_SHORTCUT)
                shortcut_roi = shortcut_mask[roi_y_start:height, :]
                shortcut_contours, _ = cv2.findContours(shortcut_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                if shortcut_contours:
                    largest_shortcut = max(shortcut_contours, key=cv2.contourArea)
                    if cv2.contourArea(largest_shortcut) > 200:
                        contours = shortcut_contours
                        use_shortcut = True
                        cv2.putText(frame_bgr, f"Tracking: {ACTIVE_SHORTCUT}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Step B: Fallback to Black Main Line
            if not use_shortcut:
                thresh = get_color_mask(hsv_full, "black")
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
                    turn_right(70, 0.1)
                else: 
                    set_motor_speed(-TURN_SPEED, TURN_SPEED)
                    turn_left(70, 0.1)

        # FPS counter 
        cv2.putText(frame_bgr, f"FPS: {int(fps)}", (width - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            
        # Encode for Web 
        _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 40])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    try:
        print("Starting unified robot server at http://0.0.0.0:5000")
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\nShutting down safely...")
    finally:
        stop_motors()
        pwm_a.stop()
        pwm_b.stop()
        GPIO.cleanup()
        picam2.stop()
        picam2.close()