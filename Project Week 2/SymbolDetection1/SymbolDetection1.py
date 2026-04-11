import cv2
from flask import Flask, Response, render_template_string
from picamera2 import Picamera2

# 1. Initialize Flask to serve static files from the current directory ('.')
app = Flask(__name__, static_folder='.', static_url_path='/')
picam2 = Picamera2()

# 2. Start the Camera (320x240 for fast streaming)
config = picam2.create_video_configuration(main={"size": (320, 240)})
picam2.configure(config)
picam2.start()
picam2.set_controls({"AwbMode": 6})

# 3. THE EMBEDDED HTML WEBPAGE
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Browser AI Symbol Detection</title>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.3.1/dist/tf.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@0.8/dist/teachablemachine-image.min.js"></script>
    
    <style>
        body { font-family: Arial, sans-serif; text-align: center; background-color: #222; color: white; margin-top: 50px; }
        #video-container { border: 4px solid #444; display: inline-block; border-radius: 8px; overflow: hidden; }
        #label-container { margin-top: 20px; font-size: 24px; font-weight: bold; }
        .prediction { margin: 5px; padding: 10px; background-color: #333; border-radius: 5px; }
    </style>
</head>
<body>

    <h2>Live Symbol Detection</h2>
    
    <div id="video-container">
        <img id="videoStream" src="/video_feed" width="320" height="240" crossorigin="anonymous" />
    </div>

    <div id="label-container">Loading AI Model...</div>

    <script>
        // Because we told Flask the static folder is the root directory, 
        // the browser will look for the model files right next to SymbolDetectionTest.py
        const URL = "/";

        let model, labelContainer, maxPredictions;
        const imgElement = document.getElementById("videoStream");

        async function init() {
            const modelURL = URL + "model.json";
            const metadataURL = URL + "metadata.json";

            model = await tmImage.load(modelURL, metadataURL);
            maxPredictions = model.getTotalClasses();

            labelContainer = document.getElementById("label-container");
            labelContainer.innerHTML = ""; 
            for (let i = 0; i < maxPredictions; i++) {
                let div = document.createElement("div");
                div.className = "prediction";
                labelContainer.appendChild(div);
            }

            setInterval(predict, 100);
        }

        async function predict() {
            if (!imgElement.complete || imgElement.naturalWidth === 0) return;

            const predictions = await model.predict(imgElement);
            
            for (let i = 0; i < maxPredictions; i++) {
                const classPrediction =
                    predictions[i].className + ": " + Math.round(predictions[i].probability * 100) + "%";
                labelContainer.childNodes[i].innerHTML = classPrediction;
                
                if (predictions[i].probability > 0.75) {
                    labelContainer.childNodes[i].style.color = "#00ff00";
                } else {
                    labelContainer.childNodes[i].style.color = "#ffffff";
                }
            }
        }

        init();
    </script>
</body>
</html>
"""

# 4. SERVER ROUTES=
@app.route('/')
def index():
    # Serve the HTML string directly instead of looking for a file
    return render_template_string(HTML_PAGE)

def generate_frames():
    while True:
        request = picam2.capture_request()
        frame_rgb = request.make_array("main")
        request.release() 

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 50])
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)