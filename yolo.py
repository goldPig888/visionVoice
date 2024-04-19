import os
import time
import numpy as np
import cv2
import cv2
import numpy as np
import os

class ObjectDetection:
    def __init__(self):
        print("Current working directory:", os.getcwd())
        project_path = os.path.abspath(os.getcwd())
        models_path = os.path.join(project_path, "models")
        config_path = os.path.join(models_path, "yolov3.cfg")
        weights_path = os.path.join(models_path, "yolov3.weights")
        names_path = os.path.join(models_path, "coco.names")

        if not os.path.exists(config_path) or not os.path.exists(weights_path):
            raise FileNotFoundError("Model files not found. Check the paths!")
        self.MODEL = cv2.dnn.readNet(weights_path, config_path)

        if not os.path.exists(names_path):
            raise FileNotFoundError("COCO names file not found.")
        with open(names_path, "r") as f:
            self.CLASSES = [line.strip() for line in f.readlines()]

        self.OUTPUT_LAYERS = [self.MODEL.getLayerNames()[i - 1] for i in self.MODEL.getUnconnectedOutLayers()]
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))
        self.COLORS /= (np.sum(self.COLORS**2, axis=1)**0.5/255)[np.newaxis].T
        self.detections = []  # List to store current frame detections

    def detectObj(self, snap):
        self.detections.clear()  # Clear previous frame detections
        height, width, _ = snap.shape
        blob = cv2.dnn.blobFromImage(snap, 1/255, (416, 416), swapRB=True, crop=False)
        self.MODEL.setInput(blob)
        outs = self.MODEL.forward(self.OUTPUT_LAYERS)

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.7:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = self.CLASSES[class_ids[i]]
            center_x, center_y = int(x + w / 2), int(y + h / 2)
            position = self.classify_position(center_x, center_y, width, height)
            self.detections.append({
                "label": label,
                "position": position
            })
            label = self.CLASSES[class_ids[i]]
            color = self.COLORS[i]
            cv2.rectangle(snap, (x, y), (x + w, y + h), color, 2)
            cv2.putText(snap, f"{label}: {confidences[i]:.2f}", (x, y - 5), font, 2, color, 2)

        return snap
    
    def classify_position(self, center_x, center_y, image_width, image_height):
        horizontal = "right" if center_x > image_width / 2 else "left"
        vertical = "above" if center_y < image_height / 2 else "below"
        return {"horizontal": horizontal, "vertical": vertical}

    def get_latest_detections(self):
        return self.detections
        
class VideoStreaming:
    def __init__(self):
        self.model_loaded = False  # Initially false, will be true when model is loaded
        self.VIDEO = cv2.VideoCapture(0)
        self.MODEL = ObjectDetection()
        self._preview = True
        self._flipH = False
        self._detect = False

        if self.MODEL:  # Assuming MODEL initialization is successful
            self.model_loaded = True
    @property
    def preview(self):
        return self._preview

    @preview.setter
    def preview(self, value):
        self._preview = bool(value)

    @property
    def flipH(self):
        return self._flipH

    @flipH.setter
    def flipH(self, value):
        self._flipH = bool(value)

    @property
    def detect(self):
        return self._detect

    @detect.setter
    def detect(self, value):
        self._detect = bool(value)

    def show(self):
        if not self.VIDEO.isOpened():
            print("Failed to open camera.")
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + b'Camera not available' + b'\r\n'
            return

        while True:
            ret, snap = self.VIDEO.read()
            if not ret:
                print("Failed to capture frame from camera.")
                break

            if self.flipH:
                snap = cv2.flip(snap, 1)

            if self._preview:
                if self.detect:
                    snap = self.MODEL.detectObj(snap)
            else:
                snap = np.zeros((int(self.VIDEO.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.VIDEO.get(cv2.CAP_PROP_FRAME_WIDTH))), np.uint8)
                label = "camera disabled"
                H, W = snap.shape
                font = cv2.FONT_HERSHEY_PLAIN
                color = (255, 255, 255)
                cv2.putText(snap, label, (W//2 - 100, H//2), font, 2, color, 2)

            frame = cv2.imencode(".jpg", snap)[1].tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.01)

    def get_latest_detections(self):
        # Fetch latest detections from the ObjectDetection instance
        return self.MODEL.get_latest_detections()


