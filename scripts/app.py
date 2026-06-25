import os
import sys
import cv2
from flask import Flask, render_template, Response, jsonify
from src.realtime import FatiguePredictor


def get_resource_path(relative_path):
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", relative_path)


MODEL_PATH = get_resource_path("models/heuristic_model.pkl")
TASK_PATH = get_resource_path("face_landmarker.task")

app = Flask(
    __name__,
    template_folder=get_resource_path("templates"),
    static_folder=get_resource_path("static"),
)
predictor = FatiguePredictor(
    model_path=MODEL_PATH,
    task_path=TASK_PATH,
    window_size=75,
)

current_system_state = {
    "status": "Initializing",
    "fatigue_prob": 0.0,
    "perclos": 0.0,
    "reason": "Waiting for monitoring data",
}


def generate_frames():
    global current_system_state
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open the camera.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        annotated_frame, status, prob = predictor.process_frame(frame)
        current_system_state = {
            "status": predictor.current_status,
            "fatigue_prob": predictor.fatigue_prob,
            "perclos": round(predictor.perclos, 2),
            "reason": predictor.status_reason,
        }

        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

    cap.release()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/status")
def get_status():
    return jsonify(current_system_state)


if __name__ == "__main__":
    print("Open: http://127.0.0.1:5000")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        threaded=True,
        use_reloader=False,
    )
