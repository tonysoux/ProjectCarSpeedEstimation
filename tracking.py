import cv2
import numpy as np
from ultralytics import YOLO
import json
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog
import os
from datetime import datetime
import paho.mqtt.client as mqtt

# Charger le modèle YOLO
model = YOLO('yolov8x.pt')

# Dimensions de l'autoroute en mode plan
road_length = 200  # Longueur de l’autoroute en mètres
road_width = 25    # Largeur de l’autoroute en mètres

car_position_plan = {}
car_speeds = {}  # Vitesse des voitures en km/h
speed_plot_history = {}
vehicle_alert_ids = []  # ID des véhicules en excès de vitesse

SPEED_LIMIT = 130  # km/h

mqtt_client = mqtt.Client()
mqtt_client.connect("localhost", 1883, 60)  

MQTT_TOPIC_SPEED = "vehicle/speed"
MQTT_TOPIC_ALERT = "vehicle/alert"
MQTT_TOPIC_COUNT = "vehicle/count"



def select_video_file():
    """Ouvre une boîte de dialogue pour sélectionner une vidéo et retourne un chemin relatif à partir de `data`."""
    root = Tk()
    root.withdraw()

    # Répertoire de base pour la sélection des vidéos
    data_dir = os.path.join(os.getcwd(), "data")

    # Sélectionner la vidéo
    video_path = filedialog.askopenfilename(title="Select Video File", initialdir=data_dir)

    if not video_path:
        print("Aucune vidéo sélectionnée.")
        return None

    # Convertir en chemin relatif si possible
    try:
        relative_path = os.path.relpath(video_path, data_dir)
        video_path = os.path.join("data", relative_path).replace("\\", "/")
    except ValueError:
        print("Le fichier sélectionné n'est pas dans `data`.")
        return None

    print(f"Selected video: {video_path}")
    return video_path




def select_roi(frame):
    """Permet à l'utilisateur de sélectionner une région d'intérêt (ROI) sur une image."""
    roi_polygon = []

    def draw_polygon(event, x, y, flags, param):
        """Fonction de callback pour dessiner un polygone avec la souris."""
        if event == cv2.EVENT_LBUTTONDOWN:  # Ajouter un point
            roi_polygon.append((x, y))
    while True:

        cv2.namedWindow("Select ROI")
        cv2.setMouseCallback("Select ROI", draw_polygon)

    
        temp_frame = frame.copy()
        # Dessiner le polygone
        if len(roi_polygon) > 1:
            cv2.polylines(temp_frame, [np.array(roi_polygon)], isClosed=False, color=(0, 255, 0), thickness=2)
        for point in roi_polygon:
            cv2.circle(temp_frame, point, 5, (0, 0, 255), -1)

        cv2.imshow("Select ROI", temp_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and len(roi_polygon) > 2:  # Sauvegarder la ROI
            cv2.destroyWindow("Select ROI")
            return roi_polygon
        elif key == ord('q'):  # Quitter sans sauvegarder
            roi_polygon.clear()
            cv2.destroyWindow("Select ROI")



def calculate_speed(obj_id, obj_coords_plan, fps):
    """Calcule la vitesse d'un véhicule en fonction de sa position."""
    if obj_id in car_position_plan:
        old_coords = car_position_plan[obj_id]
        distance_meters = np.linalg.norm(np.array(obj_coords_plan) - np.array(old_coords))
        speed_m_per_s = (distance_meters * fps) / 5  # Intervalle ajusté pour éviter les fluctuations
        speed_kmh = speed_m_per_s * 3.6
        car_speeds[obj_id] = speed_kmh
        return smooth_speed(obj_id, speed_kmh)
    return 0


def smooth_speed(vehicle_id, new_speed):
    """Lissage des vitesses en utilisant une moyenne glissante."""
    if vehicle_id not in speed_plot_history:
        speed_plot_history[vehicle_id] = []
    speed_plot_history[vehicle_id].append(new_speed)

    # Garder seulement les 5 dernières mesures pour le lissage
    if len(speed_plot_history[vehicle_id]) > 5:
        speed_plot_history[vehicle_id].pop(0)

    return np.mean(speed_plot_history[vehicle_id])


def save_roi(video_path, roi_polygon):
    """Sauvegarde la ROI dans un fichier JSON avec le chemin de la vidéo comme clé."""
    data = {}

    # Charger les ROIs existantes si le fichier existe
    try:
        with open('roi_data.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        pass

    # Ajouter la key si elle n'existe pas
    if video_path not in data:
        data[video_path] = {}
    
    # Mettre à jour ou ajouter la ROI pour cette vidéo
    data[video_path]["roi_polygon"] = roi_polygon
    road_length=int(input("Veuillez définir la longueur de l'autoroute en mode plan."))
    data[video_path]["road_length"] = road_length   # Ajouter la longueur de la route
    road_width=int(input("Veuillez définir la largeur de l'autoroute en mode plan."))
    data[video_path]["road_width"] = road_width     # Ajouter la largeur de la route


    # Sauvegarder les nouvelles données
    with open('roi_data.json', 'w') as f:
        json.dump(data, f)

    return data[video_path]

def load_roi(video_path):
    """Charge la ROI associée à une vidéo si elle existe."""
    try:
        with open('roi_data.json', 'r') as f:
            data = json.load(f)
        if video_path in data:
            return data[video_path]
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    return None
def define_autoute_polygon(road_length, road_width):
    """Définit les dimensions de l'autoroute en mode plan."""
    autoroute_polygon = [[0, 0], [road_width-1, 0], [road_width-1, road_length-1], [0, road_length-1]]

    return autoroute_polygon

def log_speed_violation(vehicle_id, speed):
    """Enregistre une infraction dans un fichier JSON."""
    global vehicle_alert_ids
    infraction_data = {
        "vehicle_id": int(vehicle_id),
        "speed": round(speed, 2),
        "timestamp": datetime.now().isoformat()
    }
    try:
        with open("infractions.json", "r") as file:
            infractions = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        infractions = []

    infractions.append(infraction_data)
    vehicle_alert_ids.append(vehicle_id)

    with open("infractions.json", "w") as file:
        json.dump(infractions, file, indent=4)
    print(f"🚨 Infraction enregistrée : Véhicule {vehicle_id} à {speed} km/h")

    # Publier une alerte via MQTT
    alert_payload = json.dumps(infraction_data)
    result=mqtt_client.publish(MQTT_TOPIC_ALERT, alert_payload)
    result.wait_for_publish()
    print(f"MQTT Payload publié sur {MQTT_TOPIC_ALERT}: {alert_payload}")

def send_speed_data():
    """Envoie les vitesses des véhicules via MQTT."""
    speed_payload = json.dumps({str(vehicle_id): round(speed, 2) for vehicle_id, speed in car_speeds.items()})
    result=mqtt_client.publish(MQTT_TOPIC_SPEED, speed_payload)
    result.wait_for_publish()
    print(f"MQTT Payload publié sur {MQTT_TOPIC_SPEED}: {speed_payload}")


def process_video(video_path):
    """Traitement principal de la vidéo pour la détection et le suivi de véhicules."""
    global roi_polygon

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Impossible de lire la vidéo.")
        return
    frame = cv2.resize(frame, (960, 540))
    roi_data = load_roi(video_path)

    if roi_data:
        roi_polygon = np.array(roi_data["roi_polygon"], dtype=np.int32)
    else:
        print("Aucune ROI définie pour cette vidéo. Veuillez en sélectionner une.")
        roi_polygon = select_roi(frame)
        save_roi(video_path, roi_polygon)
    
    roi_data = load_roi(video_path)
    
    roi_polygon = np.array(roi_data["roi_polygon"], dtype=np.int32)
    road_length = roi_data["road_length"]
    road_width = roi_data["road_width"]


    autoroute_polygon = define_autoute_polygon(road_length, road_width)
    # Matrice de transformation pour perspective
    matrix = cv2.getPerspectiveTransform(np.array(roi_polygon, np.float32), np.array(autoroute_polygon, np.float32))

    autoroute_plan = np.ones((road_length, road_width, 3), dtype=np.uint8) * 255
    detected_objects_in_plan = []

    counter = 0  # Compteur de frames

    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (960, 540))
        if not ret:
            break

        fps = cap.get(cv2.CAP_PROP_FPS)
        counter += 1
        
        if counter % 5 != 0:
            continue

        overlay = frame.copy()
        cv2.polylines(overlay, [np.array(roi_polygon, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
        frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

        # Découpe de la sous-image contenant la ROI
        x, y, w, h = cv2.boundingRect(roi_polygon)
        roi_frame = frame[y:y+h, x:x+w]

        # Détection et suivi des véhicules
        results = model.track(source=roi_frame, tracker="bytetrack.yaml", persist=True, stream=True, conf=0.3)
        autoroute_plan.fill(255)
        detected_objects_in_plan.clear()

        for result in results:
            if result.boxes.id is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
            ids = result.boxes.id.cpu().numpy()  # IDs des objets suivis

            for box, obj_id in zip(boxes, ids):
                x1, y_top, x2, y_bottom = box
                x1 += x
                x2 += x
                y_top += y
                y_bottom += y
                x_center = (x1 + x2) / 2

                if cv2.pointPolygonTest(np.array(roi_polygon, np.int32), (x_center, y_bottom), False) >= 0:
                    cv2.rectangle(frame, (int(x1), int(y_top)), (int(x2), int(y_bottom)), (255, 0, 0), 2)
                    cv2.putText(frame, f"ID: {int(obj_id)}", (int(x1), int(y_top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    obj_coords = np.array([[x_center, y_bottom]], dtype=np.float32)
                    obj_coords_plan = cv2.perspectiveTransform(obj_coords[None, :, :], matrix)[0][0]

                    speed_kmh = calculate_speed(obj_id, obj_coords_plan, fps)
                    if speed_kmh > SPEED_LIMIT:
                        if obj_id not in vehicle_alert_ids:
                            log_speed_violation(obj_id, speed_kmh)
                        cv2.putText(frame, f"Vehicle : {int(obj_id)} ALERTE: {int(speed_kmh)} km/h", (int(x1), int(y_top) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.rectangle(frame, (int(x1), int(y_top)), (int(x2), int(y_bottom)), (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, f"Speed: {int(speed_kmh)} km/h", (int(x1), int(y_bottom) + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.rectangle(frame, (int(x1), int(y_top)), (int(x2), int(y_bottom)), (255, 0, 0), 2)
                    car_position_plan[obj_id] = obj_coords_plan
                    detected_objects_in_plan.append(obj_coords_plan)

        send_speed_data()
        cv2.imshow('frame', frame)
        
        
        """ plt.imshow(cv2.cvtColor(autoroute_plan, cv2.COLOR_BGR2RGB))
        if detected_objects_in_plan:
            plt.scatter(
                [pt[0] for pt in detected_objects_in_plan],
                [pt[1] for pt in detected_objects_in_plan],
                c='red', s=10
            )
        plt.plot([12, 12], [0, road_length], c='black', linewidth=1)
        plt.plot([7, 7], [0, road_length], c='black', linestyle='--', linewidth=1)
        plt.plot([17, 17], [0, road_length], c='black', linestyle='--', linewidth=1)
        plt.title("Plan View")
        plt.xlim(0, road_width)
        plt.ylim(road_length, 0)
        plt.pause(0.01)
        plt.clf() """

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = select_video_file()
    if video_path:
        process_video(video_path)
    else:
        print("No video selected.")  