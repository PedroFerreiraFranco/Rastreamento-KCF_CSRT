from ultralytics import YOLO
import cv2

MODEL_PATH = "yolov8n.pt"
VIDEO_PATH = "video4.mp4"
TARGET_CLASS = "person"

def open_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Erro ao abrir o vídeo: {video_path}")
    return cap

def read_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None
    return frame

def detect_target(frame, model, target_class):
    results = model(frame)
    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if model.names[int(class_id)] == target_class and score > 0.5:
            return int(x1), int(y1), int(x2 - x1), int(y2 - y1)
    return None

def initialize_tracker(tracker_type, frame, bbox):
    tracker = tracker_type()
    tracker.init(frame, bbox)
    return tracker

def draw_bounding_box(frame, bbox, color, text=None):
    x, y, w, h = map(int, bbox)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    if text:
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    model = YOLO(MODEL_PATH)

    try:
        cap = open_video(VIDEO_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    frame = read_frame(cap)
    if frame is None:
        print("Não foi possível ler o primeiro frame do vídeo!")
        cap.release()
        return

    bbox = detect_target(frame, model, TARGET_CLASS)
    if bbox is None:
        print(f"Objeto da classe '{TARGET_CLASS}' não detectado no primeiro frame!")
        cap.release()
        return

    tracker = initialize_tracker(cv2.TrackerKCF_create, frame, bbox)

    while True:
        frame = read_frame(cap)
        if frame is None:
            break

        ret, bbox = tracker.update(frame)
        if ret:
            draw_bounding_box(frame, bbox, (0, 255, 0))
        else:
            cv2.putText(frame, "Rastreamento falhou!", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        cv2.imshow("KCF Tracker", frame)

        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
