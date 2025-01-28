import cv2
import numpy as np
from ultralytics import YOLO
import json
import matplotlib.pyplot as plt

# Charger le modèle YOLO
model = YOLO('yolov8m.pt')

# Chemin de la vidéo
video_path = 'data/vehicles.mp4'

# Variables globales pour la sélection du polygone
roi_polygon = []
drawing = False
road_length=150 # Longueur de l’autoroute en mètres
road_width=25 # Largeur de l’autoroute en mètres

# Dimensions de la ROI en mode plan
autoroute_polygon = [[0, 0], [road_width-1, 0], [road_width-1, road_length-1], [0, road_length-1]]
autoroute_plan = np.ones((road_length, road_width, 3), dtype=np.uint8)*255 
detected_objects_in_plan = []  # Coordonnées des objets détectés dans le plan
car_position_plan = {}
car_speeds = {}  # Vitesse des voitures en km/h
speed_plot_history = {}

def draw_polygon(event, x, y, flags, param):
    """Fonction de callback pour dessiner un polygone avec la souris."""
    global roi_polygon, drawing

    if event == cv2.EVENT_LBUTTONDOWN:  # Commencer à dessiner
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:  # Fin du point
        drawing = False
        roi_polygon.append((x, y))  # Ajouter un dernier point
    elif event == cv2.EVENT_RBUTTONDOWN:  # Supprimer tous les points si clic droit
        roi_polygon.clear()
    
    

def select_roi(video_path):
    """Permet de sélectionner une région d’intérêt (ROI) sur la première frame de la vidéo."""
    global roi_polygon

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print("Impossible de lire la vidéo.")
        return None
    
    # Charger la ROI à partir du fichier JSON
    try:
        with open('roi_polygon.json', 'r') as f:
            roi_polygon = json.load(f)
            return np.array(roi_polygon)
    except:
        roi_polygon = []
        cv2.namedWindow("Select ROI")
        cv2.setMouseCallback("Select ROI", draw_polygon)

        while True:
            temp_frame = frame.copy()
            temp_frame = cv2.resize(temp_frame, (960, 540))
            # Dessiner le polygone pendant la sélection
            if len(roi_polygon) > 1:
                cv2.polylines(temp_frame, [np.array(roi_polygon)], isClosed=False, color=(0, 255, 0), thickness=2)
            for point in roi_polygon:
                cv2.circle(temp_frame, point, 5, (0, 0, 255), -1)

            cv2.imshow("Select ROI", temp_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quitter sans enregistrer
                roi_polygon = []
            elif key == ord('s'):  # Sauvegarder la ROI
                # Enregistrer la ROI dans un fichier JSON
                with open('roi_polygon.json', 'w') as f:
                    json.dump(roi_polygon, f)
                break

        cv2.destroyWindow("Select ROI")
        cap.release()

        if len(roi_polygon) > 2:
            return np.array(roi_polygon)  # Retourner la ROI comme un polygone
        else:
            print("Région d’intérêt non définie.")
            return None
        
def calculate_speeds(car_position_plan, car_speeds):
    global speed_plot_history
    
    if obj_id in car_position_plan:
        old_x, old_y = car_position_plan[obj_id]
        dx = abs (obj_coords_plan[0] - old_x)
        dy = abs (obj_coords_plan[1] - old_y)
        distance_meters = dy
        speed_m_per_s = distance_meters * fps/3
        speed_kmh = speed_m_per_s * 3.6
        car_speeds[obj_id] = speed_kmh
        if obj_id not in speed_plot_history:
            speed_plot_history[obj_id] = []
        speed_plot_history[obj_id].append(speed_kmh)
        return speed_kmh
    return 0

# Utiliser l'outil de sélection ROI
roi_polygon = select_roi(video_path)
if roi_polygon is None:
    print("Aucune région d’intérêt définie. Arrêt.")
    exit()

# Lire la vidéo et appliquer le tracking
cap = cv2.VideoCapture(video_path)
counter = 0 # Compteur de frames
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (960, 540))
    if not ret:
        break
    fps = cap.get(cv2.CAP_PROP_FPS)
    counter += 1

    if counter % 3 != 0: 
        continue

    # Dessiner la région d’intérêt sur l’image
    overlay = frame.copy()
    cv2.polylines(overlay, [np.array(roi_polygon, np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
    frame = cv2.addWeighted(overlay, 0.8, frame, 0.2, 0)

    # Obtenez les limites du rectangle englobant la ROI
    x, y, w, h = cv2.boundingRect(roi_polygon)

    # Découpez la sous-image contenant la ROI
    roi_frame = frame[y:y+h, x:x+w]

    # Effectuer le tracking sur la frame
    results = model.track(source=roi_frame, tracker="bytetrack.yaml", persist=True, stream=True)
    matrix = cv2.getPerspectiveTransform(np.array(roi_polygon, np.float32), np.array(autoroute_polygon, np.float32))
    
    autoroute_plan.fill(255)  # Effacer le plan de l’autoroute
    detected_objects_in_plan.clear()

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        ids = result.boxes.id.cpu().numpy()     # IDs des objets suivis

        for box, obj_id in zip(boxes, ids):
            x1, y_top, x2, y_bottom = box
            x1 += x
            x2 += x
            y_top += y
            y_bottom += y
            x_center = (x1 + x2) / 2
            
            

            # Vérifier si le centre du rectangle est dans la ROI
            if cv2.pointPolygonTest(np.array(roi_polygon, np.int32), (x_center, y_bottom), False) >= 0:
                # Dessiner la boîte englobante et l’ID de l’objet dans la ROI
                cv2.rectangle(frame, (int(x1), int(y_top)), (int(x2), int(y_bottom)), (255, 0, 0), 2)
                cv2.circle(frame, (int(x_center), int(y_bottom)), 5, (0, 255, 0), -1)
                cv2.putText(frame, f"ID: {int(obj_id)}", (int(x1), int(y_top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                # Convertir les coordonnées de l’objet dans le plan de l’autoroute
                obj_coords = np.array([[x_center, y_bottom]], dtype=np.float32)
                obj_coords_plan = cv2.perspectiveTransform(obj_coords[None, :, :], matrix)[0][0]

                # Calculer la vitesse si une position précédente existe
                speed_kmh = calculate_speeds(car_position_plan, car_speeds)
                if speed_kmh > 0:
                    # Ajout de la vitesse des voitures sur la frame
                    cv2.putText(frame, f"Speed: {int(speed_kmh)} km/h", (int(x1), int(y_bottom) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Mettre à jour les positions actuelles
                car_position_plan[obj_id] = obj_coords_plan
                detected_objects_in_plan.append(obj_coords_plan)

                

    # Afficher la vidéo avec la détection
    cv2.imshow('frame', frame)

    # Afficher le plan de l’autoroute
    plt.imshow(cv2.cvtColor(autoroute_plan, cv2.COLOR_BGR2RGB))
    if detected_objects_in_plan:
        plt.scatter(
            [pt[0] for pt in detected_objects_in_plan],
            [pt[1] for pt in detected_objects_in_plan],
            c='red', s=10
        )
    plt.plot([12,12], [0, road_length], c='black',linewidth=1)
    plt.plot([7,7], [0, road_length], c='black', linestyle='--', linewidth=1)
    plt.plot([17,17], [0, road_length], c='black', linestyle='--', linewidth=1)
    plt.title("Vue en mode plan")
    plt.xlim(0, road_width)
    plt.ylim(road_length, 0)
    plt.pause(0.01)  # Pause pour actualiser l’affichage
    plt.clf()  # Effacer le graphique pour la prochaine frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
