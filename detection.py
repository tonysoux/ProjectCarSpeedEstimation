import cv2
import numpy as np
from ultralytics import YOLO
import json



def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    return cap

def select_ROI(frame):
    r = cv2.selectROI(frame)
    #stocker la roi dans un json
    with open('roi.json', 'w') as f:
        json.dump(r, f)
    return r

def load_roi(frame): #fonction pour ouvrir le fichier roi.json et charger la roi si elle existe sinon on la selectionne
    try:
        with open('roi.json', 'r') as f:
            r = json.load(f)
    except:
        r = select_ROI(frame)
    return r

# Detection et tracking des voitures avec DeepSORT
def detection(frame, model):
    # Detection
    results = model(frame)
    return results

def main(video_path):
    cap = load_video(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Load ROI
        r = load_roi(frame)
        model = YOLO('yolov8s.pt')
        # Crop frame with ROI
        frame = frame[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        # Detection and tracking
        results = detection(frame, model)
        for i in range(len(results.xyxy)):
            x1, y1, x2, y2, conf, cls = results.xyxy[i]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, str(cls), (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = 'data/sample_video.mp4'
    main(video_path)
