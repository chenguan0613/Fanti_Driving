import cv2
from flask import Flask, render_template, Response, jsonify
from src.realtime.predictor import FatiguePredictor

app = Flask(__name__)
# 全局实例化预测器，避免每次请求重复加载模型
predictor = FatiguePredictor()


def generate_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open the camera！")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 1. 喂给 AI 引擎处理
        annotated_frame, status, prob = predictor.process_frame(frame)

        # 2. 压缩为 JPEG 字节流
        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        frame_bytes = buffer.tobytes()

        # 3. 推送给前端
        yield (
            b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


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
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)
