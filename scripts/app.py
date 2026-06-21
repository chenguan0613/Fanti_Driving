import cv2
from flask import Flask, render_template, Response, jsonify
from src.realtime import FatiguePredictor

MODEL_PATH = "./models/fatigue_model.pkl"

app = Flask(__name__, template_folder="../templates")
predictor = FatiguePredictor(
    model_path=MODEL_PATH,
    window_size=150,
)


def generate_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open the camera.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        annotated_frame, _, _ = predictor.process_frame(frame)
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
    return jsonify(
        {
            "status": predictor.current_status,
            "fatigue_prob": predictor.fatigue_prob,
            "perclos": round(predictor.perclos, 2),
        }
    )


if __name__ == "__main__":
    print("Open: http://127.0.0.1:5000")
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
        threaded=True,
        use_reloader=False,
    )
