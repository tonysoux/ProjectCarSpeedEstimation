import cv2

# 🔴 Remplace par l'URL du flux vidéo en direct
stream_url = "https://filmsgieat.viewsurf.com/aprr48/media_1739811000.mp4"

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("⚠️ Impossible d'ouvrir le flux vidéo")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Problème de lecture du flux")
        break

    cv2.imshow("Webcam Autoroute", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Quitter avec 'q'
        break

cap.release()
cv2.destroyAllWindows()
