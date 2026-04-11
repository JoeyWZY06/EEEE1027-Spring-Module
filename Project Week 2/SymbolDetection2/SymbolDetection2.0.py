import cv2
import numpy as np
import time
from flask import Flask, Response, render_template_string, jsonify
from picamera2 import Picamera2

app = Flask(__name__, static_folder='.', static_url_path='/')

# --- 1. Camera Setup ---
print("Initializing Camera...")
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (320, 240)})
picam2.configure(config)
picam2.start()
picam2.set_controls({"AwbMode": 6})

# --- Global State for the Toggle Logic ---
current_cv_status = "Idle"
cv_active = False  # OpenCV is OFF by default

# --- 2. API Endpoints to Toggle OpenCV ---
@app.route('/set_cv/<state>')
def set_cv(state):
    global cv_active, current_cv_status
    if state == "on":
        cv_active = True
    else:
        cv_active = False
        current_cv_status = "Idle"
    return jsonify({"status": "ok"})

@app.route('/status')
def get_status():
    global current_cv_status
    return jsonify({"status": current_cv_status})

# --- 3. THE EMBEDDED HTML WEBPAGE ---
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Efficient AI/CV Pipeline</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.3.1/dist/tf.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@0.8/dist/teachablemachine-image.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; background-color: #222; color: white; margin-top: 50px; }
        #video-container { border: 4px solid #444; display: inline-block; border-radius: 8px; overflow: hidden; }
        #status-banner { margin-top: 20px; font-size: 24px; font-weight: bold; height: 35px; }
        #label-container { margin-top: 10px; font-size: 20px; display: flex; flex-direction: column; align-items: center; }
        .prediction { margin: 5px; padding: 10px; background-color: #333; border-radius: 5px; width: 300px; }
    </style>
</head>
<body>

    <h2>Pipeline: ML Primary -> OpenCV Extractor</h2>
    
    <div id="video-container">
        <img id="videoStream" src="/video_feed" width="320" height="240" crossorigin="anonymous" />
    </div>

    <div id="status-banner" style="color: yellow;">Loading System...</div>
    <div id="label-container">Initializing Model...</div>

    <script>
        const URL = "/";
        let model, labelContainer, statusBanner, maxPredictions;
        const imgElement = document.getElementById("videoStream");
        
        let currentMode = "ML"; // State Machine starts in ML
        let cvMissCount = 0;    // Used to prevent flickering when switching back

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
                document.getElementById("status-banner").innerHTML = "Error loading model!";
                document.getElementById("status-banner").style.color = "red";
                console.error("Model load error:", error);
            }
        }

        async function processFrame() {
            if (!imgElement.complete || imgElement.naturalWidth === 0) return;

            try {
                if (currentMode === "ML") {
                    // --- 1. RUN MACHINE LEARNING ---
                    const predictions = await model.predict(imgElement);
                    
                    let bestPrediction = predictions[0];
                    for (let i = 1; i < maxPredictions; i++) {
                        if (predictions[i].probability > bestPrediction.probability) {
                            bestPrediction = predictions[i];
                        }
                    }

                    statusBanner.innerHTML = "Primary System (Teachable Machine)";
                    statusBanner.style.color = "#ffaa00";

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

                    // TRIGGER: If ML detects an Arrow with >90% confidence, switch to OpenCV
                    const confidenceThreshold = 0.90; 

                    // NOTE: Ensure your Teachable Machine class name contains "arrow"
                    if (bestPrediction.className.toLowerCase().includes("arrow") && bestPrediction.probability > confidenceThreshold) {
                        currentMode = "CV";
                        cvMissCount = 0;
                        await fetch('/set_cv/on'); // Signal Python to start math
                    }

                } else if (currentMode === "CV") {
                    // --- 2. RUN OPENCV (ML IS PAUSED) ---
                    const response = await fetch('/status');
                    const data = await response.json();

                    statusBanner.innerHTML = "Arrow Found -> OpenCV Active";
                    statusBanner.style.color = "#00ff00";

                    if (data.status !== "No Arrow" && data.status !== "Idle") {
                        cvMissCount = 0; // Reset misses since we see an arrow
                        labelContainer.innerHTML = `<div class='prediction' style='color: #00ff00; border: 2px solid #00ff00;'>
                            CV Detected: ${data.status}
                        </div>`;
                    } else {
                        // Debouncer: Wait 5 frames to ensure the arrow is actually gone
                        cvMissCount++;
                        if (cvMissCount > 3) {
                            currentMode = "ML";
                            await fetch('/set_cv/off'); // Signal Python to stop math
                        } else {
                            labelContainer.innerHTML = `<div class='prediction' style='color: #ff4444;'>
                                Locating Arrow Geometry...
                            </div>`;
                        }
                    }
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

# --- 4. OpenCV Logic ---
def process_mask(mask, color_name, frame):
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
                """cv2.circle(frame, (box_center_x, box_center_y), 4, (0, 0, 255), -1) 
                cv2.circle(frame, (mass_center_x, mass_center_y), 4, (255, 0, 0), -1)
                """
                
    return results

def detect_arrows(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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

    return frame, detected_symbols

# --- 5. Flask Streaming ---
@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

def generate_frames():
    global current_cv_status, cv_active
    while True:
        request = picam2.capture_request()
        frame_rgb = request.make_array("main")
        request.release() 

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # PIPELINE SWITCH: Only run OpenCV math if the JS triggered it
        if cv_active:
            annotated_frame, detected_symbols = detect_arrows(frame_bgr)
            if detected_symbols:
                current_cv_status = detected_symbols[0]
            else:
                current_cv_status = "No Arrow"
        else:
            # OpenCV is off. Just send the raw camera frame.
            annotated_frame = frame_bgr
            current_cv_status = "Idle"
            
        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 6. SAFE EXECUTION BLOCK ---
if __name__ == '__main__':
    try:
        print("Starting Flask server on http://0.0.0.0:5000/")
        print("Press Ctrl+C in the terminal to quit gracefully.")
        app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)
    finally:
        print("\nShutting down camera cleanly...")
        picam2.stop()
        picam2.close()
        print("Done.")