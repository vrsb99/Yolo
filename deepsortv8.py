import cv2
import numpy as np
import sys
import glob
import time
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO


class YoloDetector():
    
    def __init__(self, model_name):
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using {self.device} device")
        
    def load_model(self, model_name):
        
        if model_name:
            model = YOLO(model_name)
        else:
            model = YOLO("yolov8x.pt")
            
        return model
    
    def score_frame(self, frame):
        
        self.model.to(self.device)
        downscale_factor = 2
        width = int(frame.shape[1] / downscale_factor)
        height = int(frame.shape[0] / downscale_factor)
        frame = cv2.resize(frame, (width, height))
        
        results = self.model(frame)
        
        return results
    
    def class_to_labels(self, x):
        return self.classes[int(x)]
    
    def plot_boxes(self, results, frame, height, width, confidence=0.3):

        for results in results:

            detections = []
            x_shape, y_shape = width, height

            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r

                if score >= confidence:
                    
                    if self.class_to_labels(class_id) in ["person"]:
                        
                        x1 = int(x1)
                        y1 = int(y1)
                        x2 = int(x2)
                        y2 = int(y2)

                        tlwh = np.asarray([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)
                        confidence = float(score)
                        feature = self.class_to_labels(class_id)

                        detections.append([tlwh, confidence, feature])

            return frame, detections

    
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = YoloDetector(model_name=None)

object_tracker = DeepSort(
    max_age = 5,
    n_init = 2,
    nms_max_overlap = 1.0,
    max_cosine_distance=0.2,
    nn_budget = None,
    override_track_class = None,
    embedder = 'mobilenet',
    half = True,
    bgr = True,
    embedder_gpu = True,
    embedder_model_name=None,
    embedder_wts=None,
    polygon = False,
    today = None
)

while cap.isOpened():
    success, img = cap.read()
    start = time.perf_counter()
    
    results = detector.score_frame(img)
    img, detections = detector.plot_boxes(results, img, height=img.shape[0], width=img.shape[1], confidence=0.5)
    tracks = object_tracker.update_tracks(detections, frame=img)
    

    for track in tracks:
        
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_tlbr()
        
        bbox = ltrb
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])), (0, 0, 255), 2)
        cv2.putText(img, f"ID: {track_id} ", (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        
        
    end = time.perf_counter()
    total_time = end - start
    fps = 1/total_time
        
    cv2.putText(img, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
    cv2.imshow('img', img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
cap.release()
cv2.destroyAllWindows()