from flask import Flask, render_template, request, Response, jsonify
from flask_bootstrap import Bootstrap
from flask_socketio import SocketIO, emit
import threading
import requests
from yolo import VideoStreaming
import os
import time
from flask import jsonify

application = Flask(__name__)
Bootstrap(application)
VIDEO = None  # Initialize VIDEO as None to be set after loading
socketio = SocketIO(application)
VIDEO = None  # Initialize VIDEO as None to be set after loading

def setup_model():
    """Set up model: check if file exists, download if not, and initialize VIDEO."""
    global VIDEO
    weights_path = 'models/yolov3.weights'
    weights_url = 'https://storage.googleapis.com/vision-voice-may/yolov3.weights'
    
    if not os.path.exists(weights_path):
        response = requests.get(weights_url, stream=True)
        if response.status_code == 200:
            with open(weights_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Model weights downloaded successfully.")
    VIDEO = VideoStreaming()
    print("Video streaming initialized.")

thread = threading.Thread(target=setup_model)
thread.start()

@application.route("/")
def home():
    TITLE = "Object Detection"
    return render_template("index.html", TITLE=TITLE)

@socketio.on('connect')
def handle_connect():
    print("Client connected")

@application.route("/video_feed")
def video_feed():
    if VIDEO is None or not VIDEO.VIDEO.isOpened():
        return "Camera not available", 503  # Simple text response for debugging
    return Response(VIDEO.show(), mimetype="multipart/x-mixed-replace; boundary=frame")

@application.route("/request_preview_switch", methods=['GET', 'POST'])
def request_preview_switch():
    if VIDEO is None:
        return jsonify({"error": "Camera not initialized"}), 503
    VIDEO.preview = not VIDEO.preview
    return jsonify({"new_preview_state": VIDEO.preview})

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
        return jsonify({"error": "Detection not active or camera not initialized"}), 503
    return jsonify(VIDEO.get_latest_detections())

def periodic_task():
    while True:
        if VIDEO and VIDEO.model_loaded:
            try:
                response = requests.get('http://127.0.0.1:5000/detection_data')
                print("Periodic update:", response.json())
            except Exception as e:
                print("Failed to trigger periodic update:", str(e))
                time.sleep(5)
            time.sleep(1)

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


task_thread = threading.Thread(target=periodic_task)
task_thread.daemon = True
task_thread.start()

if __name__ == "__main__":
    socketio.run(application, debug=True, host='0.0.0.0', port=5000)