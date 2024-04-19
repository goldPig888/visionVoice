from flask import Flask, render_template, request, Response, jsonify
from flask_bootstrap import Bootstrap
import threading
import requests
from yolo import VideoStreaming
import os
import time

application = Flask(__name__)
Bootstrap(application)
VIDEO = None  # Initialize VIDEO as None to be set after loading

def setup_model():
    """Set up model: check if file exists, download if not, and initialize VIDEO."""
    weights_path = 'models/yolov3.weights'
    weights_url = 'https://storage.googleapis.com/vision-voice-may/yolov3.weights'
    
    # Check if the weights file already exists
    if not os.path.exists(weights_path):
        response = requests.get(weights_url, stream=True)
        if response.status_code == 200:
            with open(weights_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Model weights downloaded successfully.")
        else:
            print(f"Failed to download model weights: Status code {response.status_code}")
            return
    
    # Initialize VIDEO after confirming weights are ready
    global VIDEO
    VIDEO = VideoStreaming()
    print("Video streaming initialized.")


# Start the model setup in a separate thread to not block the Flask app initialization
thread = threading.Thread(target=setup_model)
thread.start()


@application.route("/")
def home():
    TITLE = "Object Detection"
    # Check if VIDEO is initialized to update title accordingly
    return render_template("index.html", TITLE=TITLE)


@application.route("/video_feed")
def video_feed():
    """Video streaming route."""
    if VIDEO is None:
        return jsonify({"error": "Model not ready"}), 503
    return Response(VIDEO.show(), mimetype="multipart/x-mixed-replace; boundary=frame")


@application.route("/request_preview_switch", methods=['GET', 'POST'])
def request_preview_switch():
    if VIDEO is None:
        return jsonify({"error": "Camera not initialized"}), 503
    old_preview = VIDEO.preview
    VIDEO.preview = not VIDEO.preview
    return jsonify({"new_preview_state": VIDEO.preview, "old_preview_state": old_preview})



@application.route("/request_flipH_switch", methods=['GET', 'POST'])
def request_flipH_switch():
    if VIDEO is None:
        return jsonify({"error": "Camera not initialized"}), 503
    VIDEO.flipH = not VIDEO.flipH
    return jsonify({"new_flipH_state": VIDEO.flipH})



@application.route("/request_model_switch", methods=['GET', 'POST'])
def request_model_switch():
    if VIDEO is None:
        return jsonify({"error": "Camera not initialized"}), 503
    VIDEO.detect = not VIDEO.detect
    return jsonify({"new_detection_state": VIDEO.detect})

@application.route("/detection_data")
def detection_data():
    if VIDEO is None or not VIDEO.detect:
        time.sleep(5)
        return jsonify({"error": "Detection not active or camera not initialized"}), 503
    data = VIDEO.get_latest_detections()
    return jsonify(data)

def periodic_task():
    while True:
        if VIDEO and VIDEO.model_loaded:
            try:
                response = requests.get('http://127.0.0.1:5000/detection_data')
                print("Periodic update:", response.json())
            except Exception as e:
                print("Failed to trigger periodic update:", str(e))
                time.sleep(5)
            time.sleep(.3)
        else:
            time.sleep(5)

task_thread = None
@application.route('/start_periodic_task')
def start_periodic_task():
    global task_thread
    if not task_thread or not task_thread.is_alive():
        task_thread = threading.Thread(target=periodic_task)
        task_thread.daemon = True
        task_thread.start()
        return jsonify({'status': 'Periodic task started'}), 200
    else:
        return jsonify({'status': 'Periodic task is already running'}), 200


if __name__ == "__main__":
    application.run(debug=False, host='0.0.0.0', port=5000)
