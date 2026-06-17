import cv2
import time
from src.preprocessing.frame_extractor import FrameExtractor
from src.realtime.buffer import SlidingWindowBuffer
from src.features.window_agg import WindowAggregator


# python -m src.realtime.pipeline
def main():
    # ================= 1. 实例化核心组件 =================
    extractor = FrameExtractor()
    window_buffer = SlidingWindowBuffer(window_size=90)
    aggregator = WindowAggregator()
    # ================= 2. 启动摄像头流 =================
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open the camera")
        return
    prev_time = 0
    # ================= 3. 主循环 =================
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Unable to capture video, exiting now.")
            break
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1024, 768))
        current_time = time.time()
        # calculate FPS
        fps = 1 / (current_time - prev_time) if prev_time > 0 else 0
        prev_time = current_time
        # --- 阶段 A：单帧感知 ---
        feature = extractor.extract(frame, timestamp=current_time)
        hud_color = (0, 255, 0)
        if feature.face_detected:
            # --- 阶段 B：压入短期记忆 ---
            window_buffer.push(feature)

            # if feature.raw_landmarks:
            #     h, w, _ = frame.shape  # 获取画面的宽和高
            #     for lm in feature.raw_landmarks:
            #         # MediaPipe 给出的是 0~1 的比例，我们要乘上宽高变成像素坐标
            #         x, y = int(lm.x * w), int(lm.y * h)
            #         # 在脸上画半径为 1 的小实心绿点
            #         cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # --- 阶段 C：记忆满载，触发时序思考 ---
            if window_buffer.is_ready():
                recent_frames = window_buffer.get_window()
                stats = aggregator.aggregate(recent_frames)
                # 提取关键指标
                perclos = stats.get("perclos", 0.0)
                ear_mean = stats.get("ear_mean", 0.0)
                mar_max = stats.get("mar_max", 0.0)
                pitch_std = stats.get("pitch_std", 0.0)
                # [核心预警逻辑演示] 如果 PERCLOS 超过 20%，字体变红报警
                if perclos > 0.20:
                    hud_color = (0, 0, 255)  # 红色
                    cv2.putText(
                        frame,
                        "WARNING: FATIGUE DETECTED!",
                        (20, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3,
                    )
                # 将数据渲染到视频画面 (HUD)
                cv2.putText(
                    frame,
                    f"PERCLOS: {perclos*100:.1f}%",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    hud_color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"EAR Mean: {ear_mean:.2f}",
                    (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    hud_color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"MAR Max (Yawn): {mar_max:.2f}",
                    (350, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    hud_color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"Nodding (Pitch Std): {pitch_std:.1f}",
                    (350, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    hud_color,
                    2,
                )

            else:
                # 缓冲区还没满（刚启动的前几秒）
                loading_progress = int(
                    (window_buffer.current_size / window_buffer.window_size) * 100
                )
                cv2.putText(
                    frame,
                    f"Buffering... {loading_progress}%",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
        else:
            # 没检测到人脸，清空记忆防止数据污染
            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )
            window_buffer.clear()
        # 显示系统运行 FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )
        cv2.imshow("FANTI_DRIVING - Realtime Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Monitoring has ended.")


if __name__ == "__main__":
    main()
