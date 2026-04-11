import time
import cv2
import numpy as np
import RPi.GPIO as GPIO
from picamera2 import Picamera2
from flask import Flask, Response, render_template_string, jsonify

# ==========================================
# 1. Configuration & Global State
# ==========================================
ACTIVE_SHORTCUT = False # Turned off since you are only using black tape

# State Machine Variables
robot_state = "FOLLOWING"  # States: "FOLLOWING", "COASTING", "PAUSED", "CROSSING"
coast_start_time = 0
pause_start_time = 0
crossing_start_time = 0

PAUSE_DURATION = 10.0      # Total time to remain stopped and scan
FORWARD_DURATION = 0.3     # How long to drive straight over the symbol after pausing
COAST_DURATION = 0.21      # DELAY: How long to keep driving after detecting a symbol before stopping

# OpenCV Arrow Flag
cv_arrow_active = False

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
pwm_a = GPIO.PWM(ENA, 1000)
pwm_b = GPIO.PWM(ENB, 1000)
pwm_a.start(0)
pwm_b.start(0)

# PID & Speed Constants
KP, KI, KD = 1.0, 0.001, 2.8
BASE_SPEED, TURN_SPEED = 31, 70
FORWARD_SPEED = 40         
BLACK_THRESHOLD = 60 

SATURATION_THRESHOLD = 300 

COLOR_RANGES = {
    "red_1": [(0, 100, 100), (10, 255, 255)],
    "red_2": [(160, 100, 100), (180, 255, 255)],
    "green": [(40, 100, 100), (75, 255, 255)],
    "blue":  [(100, 50, 50), (130, 255, 255)],
    "yellow":[(20, 100, 100), (40, 255, 255)],
    "orange":[(10, 100, 100), (25, 255, 255)]
}

# ==========================================
# 3. Motor Control Functions
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
# 4. RECTANGLE EXTRACTION & PERSPECTIVE WARP
# ==========================================
def extract_and_warp_panel(frame):
    """Finds the 4-sided sign, crops it, and flattens it to a perfect 150x150 square."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Sort contours by area to check the largest ones first
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True) # 0.02 is the threshold for estimating shape
        
        # If it has 4 points and is large enough to be a sign
        if len(approx) == 4 and cv2.contourArea(cnt) > 2000:
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype="float32")
            
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)] # Top-left
            rect[2] = pts[np.argmax(s)] # Bottom-right
            
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)] # Top-right
            rect[3] = pts[np.argmax(diff)] # Bottom-left
            
            # Map the tilted points to a perfect 150x150 straight square
            dst = np.array([
                [0, 0], [149, 0], [149, 149], [0, 149]
            ], dtype="float32")
            
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(frame, M, (150, 150))
            return warped, approx
            
    return None, None

# ==========================================
# 5. OpenCV Arrow Detection Logic
# ==========================================
def process_mask(mask, color_name, warped_frame):
    results = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Area threshold lowered to 100 because the warped image is only 150x150 pixels total
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            
            roi = mask[y:y+h, x:x+w]
            direction = "Unknown"
            
            # --- LEFT / RIGHT ARROWS ---
            if w > h:
                mid_x = w // 2
                left_half = roi[:, :mid_x]
                right_half = roi[:, mid_x:]
                
                left_mass = cv2.countNonZero(left_half)
                right_mass = cv2.countNonZero(right_half)
                
                direction = "Right" if right_mass > left_mass else "Left"
                
                cv2.line(warped_frame, (x + mid_x, y), (x + mid_x, y + h), (255, 0, 0), 2)
                cv2.putText(warped_frame, f"L:{left_mass} R:{right_mass}", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # --- UP / DOWN ARROWS ---
            else:
                mid_y = h // 2
                top_half = roi[:mid_y, :]
                bottom_half = roi[mid_y:, :]
                
                top_mass = cv2.countNonZero(top_half)
                bottom_mass = cv2.countNonZero(bottom_half)
                
                direction = "Down" if bottom_mass > top_mass else "Up"
                
                cv2.line(warped_frame, (x, y + mid_y), (x + w, y + mid_y), (255, 0, 0), 2)
                cv2.putText(warped_frame, f"T:{top_mass} B:{bottom_mass}", (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            label = f"{color_name} {direction}"
            results.append(label)
            
            cv2.rectangle(warped_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(warped_frame, label, (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
    return results

def detect_arrows(warped_frame, hsv_warped):
    detected_symbols = []

    # Detect Green, Blue, Orange
    for color_name in ["green", "blue", "orange"]:
        lower, upper = COLOR_RANGES[color_name]
        mask = cv2.inRange(hsv_warped, np.array(lower), np.array(upper))
        detected_symbols.extend(process_mask(mask, color_name.capitalize(), warped_frame))

    # Detect Red (Handles the wrap-around in the HSV spectrum)
    mask_red1 = cv2.inRange(hsv_warped, np.array(COLOR_RANGES["red_1"][0]), np.array(COLOR_RANGES["red_1"][1]))
    mask_red2 = cv2.inRange(hsv_warped, np.array(COLOR_RANGES["red_2"][0]), np.array(COLOR_RANGES["red_2"][1]))
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    detected_symbols.extend(process_mask(mask_red, "Red", warped_frame))

    return detected_symbols

# ==========================================
# 6. Embedded HTML (ML Pipeline)
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
        
        let isPaused = false; 

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

                // TRIGGER LOGIC
                const confidenceThreshold = 0.50;
                const className = bestPrediction.className.toLowerCase();

                if (className !== "idle" && bestPrediction.probability > confidenceThreshold) {
                    if (!isPaused) {
                        isPaused = true;
                        statusBanner.innerHTML = "ML TRIGGERED OpenCV MATH";
                        statusBanner.style.color = "#ff4444";
                        
                        await fetch('/pause_robot/' + encodeURIComponent(bestPrediction.className));

                        setTimeout(() => { 
                            isPaused = false; 
                            statusBanner.innerHTML = "Line Following Resumed";
                            statusBanner.style.color = "#00ff00";
                        }, 15000); 
                    }
                } else if (!isPaused) {
                    statusBanner.innerHTML = "Following Line... (ML Active)";
                    statusBanner.style.color = "#00ff00";
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
# 7. Flask & Camera Setup
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

@app.route('/pause_robot/<symbol_class>')
def pause_robot(symbol_class):
    global cv_arrow_active, robot_state, coast_start_time
    
    if robot_state == "FOLLOWING":
        robot_state = "COASTING"
        coast_start_time = time.time()
        print(f"\n[!] ML Triggered Stop Maneuver for: {symbol_class}")

    if "arrow" in symbol_class.lower():
        cv_arrow_active = True
        print("[!] OpenCV Geometric Analysis Activated.")
            
    return jsonify({"status": "acknowledged"})

def generate_frames():
    global robot_state, coast_start_time, pause_start_time, crossing_start_time, cv_arrow_active
    
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
        height, width = frame_bgr.shape[:2]
        roi_y_start = int(height / 2.3)
        vision_roi = frame_bgr[0:roi_y_start, :]
        
        current_time = time.time()
        
        # ---------------------------------------------------------
        # STATE MACHINE LOGIC
        # ---------------------------------------------------------
        if robot_state == "COASTING":
            if current_time - coast_start_time < COAST_DURATION:
                set_motor_speed(BASE_SPEED, BASE_SPEED)
                cv2.putText(frame_bgr, "STATE: DELAYING STOP...", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            else:
                stop_motors()
                robot_state = "PAUSED"
                pause_start_time = current_time
        
        elif robot_state == "PAUSED":
            elapsed_time = current_time - pause_start_time
            
            if elapsed_time >= PAUSE_DURATION:
                robot_state = "CROSSING"
                crossing_start_time = current_time
            else:
                stop_motors() 
                cv2.putText(frame_bgr, f"STATE: PAUSED ({int(PAUSE_DURATION - elapsed_time)}s)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                
                # --- APPLY THE WARP AND ANALYZE WHILE PARKED ---
                warped_panel, panel_approx = extract_and_warp_panel(vision_roi)
                
                if warped_panel is not None:
                    # Draw a yellow box around the detected sign on the main camera feed
                    cv2.drawContours(vision_roi, [panel_approx], -1, (0, 255, 255), 2)
                    
                    # RUN ARROW MATH ON THE FLATTENED SQUARE
                    if cv_arrow_active:
                        hsv_warped = cv2.cvtColor(warped_panel, cv2.COLOR_BGR2HSV)
                        detect_arrows(warped_panel, hsv_warped)
                    
                    # Overlay the flattened square in the top-right corner of the stream
                    # so you (and the Teachable Machine) can see the perfectly corrected image
                    frame_bgr[0:150, width-150:width] = warped_panel
                    
        elif robot_state == "CROSSING":
            if current_time - crossing_start_time < FORWARD_DURATION:
                set_motor_speed(FORWARD_SPEED, FORWARD_SPEED) 
                cv2.putText(frame_bgr, "STATE: CROSSING SYMBOL", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,165,0), 2)
            else:
                robot_state = "FOLLOWING"
                cv_arrow_active = False

        elif robot_state == "FOLLOWING":
            # 1. Color Logic Detection (The Stop Trigger)
            hsv_vision = cv2.cvtColor(vision_roi, cv2.COLOR_BGR2HSV)
            _, s_chan, _ = cv2.split(hsv_vision)
            blurred_s = cv2.GaussianBlur(s_chan, (5, 5), 5)
            
            _, sat_mask = cv2.threshold(blurred_s, SATURATION_THRESHOLD, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5,5), np.uint8)
            sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel)
            
            M_sat = cv2.moments(sat_mask)
            color_area = M_sat['m00'] / 255.0 
            
            if color_area > 1200:
                robot_state = "COASTING"
                coast_start_time = current_time
                print("[!] Color Logic Triggered. Coasting closer before halt.")
            else:
                cv2.putText(frame_bgr, "STATE: FOLLOWING", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                
                # Pure Black Line Following
                hsv_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
                lower_black = np.array([0, 0, 0])
                upper_black = np.array([180, 255, BLACK_THRESHOLD])
                thresh = cv2.inRange(hsv_full, lower_black, upper_black)
                
                roi = thresh[roi_y_start:height, :]
                contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.putText(frame_bgr, "Tracking: black", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.rectangle(frame_bgr, (0, roi_y_start), (width, height), (255, 0, 0), 2)

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

        cv2.putText(frame_bgr, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            
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