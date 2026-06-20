import cv2
import time
import numpy as np
import joblib
import traceback
import pandas as pd
from dataclasses import asdict
from src.preprocessing.frame_extractor import FrameExtractor
from src.realtime.buffer import SlidingWindowBuffer
from src.features.window_agg import WindowAggregator


class FatiguePredictor:

    def __init__(
        # self, model_path="models/fatigue_model.pkl", task_path="face_landmarker.task"
        self,
        model_path="models/heuristic_model.pkl",
        task_path="face_landmarker.task",
    ):
        print("[INFO] 正在唤醒 FANTI_DRIVING AI 核心...")

        # ==========================================
        # 1. 加载模型和特征名单
        # ==========================================
        loaded_obj = joblib.load(model_path)
        if isinstance(loaded_obj, dict) and "model" in loaded_obj:
            self.model = loaded_obj["model"]
        else:
            self.model = loaded_obj

        if isinstance(loaded_obj, dict) and "feature_names" in loaded_obj:
            self.active_features = list(loaded_obj["feature_names"])
            print(f"[INFO] 从模型字典读取特征名单 ({len(self.active_features)} 维)。")
        elif hasattr(self.model, "feature_names_in_"):
            self.active_features = list(self.model.feature_names_in_)
            print(f"[INFO] 从模型内部读取特征名单 ({len(self.active_features)} 维)。")
        else:
            raise ValueError(
                "无法获取特征名单，请确保模型是通过 chen_train.py 保存的。"
            )

        # ==========================================
        # 3. 基础组件与状态变量初始化
        # ==========================================
        self.extractor = FrameExtractor(task_path)
        self.buffer = SlidingWindowBuffer(window_size=90)

        self.current_status = "Initializing"
        self.fatigue_prob = 0.0
        self.perclos = 0.0
        self.fps = 0
        self.prev_time = 0

        self.is_baseline_ready = False
        self.baseline_history = []
        self.baseline_stats = {}

        self.prev_ear_norm = 0.0
        self.prev_pitch_norm = 0.0

    def process_frame(self, frame):
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (1024, 768))

        # 计算运行帧率
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time) if self.prev_time > 0 else 0
        self.prev_time = current_time

        # 提取基础特征点
        feature = self.extractor.extract(frame, timestamp=current_time)

        # 无条件推入滑动窗口，依靠后续的容错率兜底，彻底杜绝因为丢1帧而卡死
        self.buffer.push(feature)

        if self.buffer.is_ready():
            raw_stats = WindowAggregator.aggregate(self.buffer.get_window())

            # 容错机制：只要最近 90 帧里有一半时间能看到脸，就继续干活
            if raw_stats and raw_stats.face_missing_ratio < 0.50:
                self.perclos = raw_stats.perclos
                self._think(raw_stats)
            else:
                self.current_status = "No Face Detected"
        else:
            self.current_status = "Buffering"

        annotated_frame = self._draw_hud(frame, feature)
        return annotated_frame, self.current_status, self.fatigue_prob

    def _think(self, raw_stats):
        """核心推理大脑 (被 try-except 终极护城河保护)"""
        try:
            stats_dict = asdict(raw_stats)

            # --- 阶段 1：建立绝对清醒基线 (冷启动校准) ---
            if not self.is_baseline_ready:
                self.baseline_history.append(stats_dict)
                self.current_status = "Calibrating"
                if len(self.baseline_history) >= 30:
                    df_base = pd.DataFrame(self.baseline_history)
                    for feat in ["ear_mean", "mar_max", "pitch_std", "yaw_std"]:
                        self.baseline_stats[feat] = df_base[feat].mean()
                    self.is_baseline_ready = True
                    print("\n[INFO] 🟢 个人基线校准完成，开始实时监控！\n")
                return

            # --- 阶段 2：实时计算增强特征 ---
            enhanced_features = stats_dict.copy()

            # 1. 归一化 (消除天生眼裂大小差异)
            for feat in ["ear_mean", "mar_max", "pitch_std", "yaw_std"]:
                base_val = self.baseline_stats[feat]
                curr_val = stats_dict[feat]
                norm_val = (curr_val - base_val) / (base_val + 1e-6)
                enhanced_features[f"{feat}_norm"] = norm_val

            # 2. 一阶差分速度计算
            curr_ear_norm = enhanced_features["ear_mean_norm"]
            curr_pitch_norm = enhanced_features["pitch_std_norm"]

            enhanced_features["ear_velocity"] = curr_ear_norm - self.prev_ear_norm
            enhanced_features["pitch_velocity"] = curr_pitch_norm - self.prev_pitch_norm

            self.prev_ear_norm = curr_ear_norm
            self.prev_pitch_norm = curr_pitch_norm

            # 3. 交叉特征
            enhanced_features["fatigue_index"] = self.perclos * abs(curr_ear_norm)

            # --- 阶段 3：执行 AI 预测 ---
            # 严格使用探测到的特征名单进行数据组装
            final_input = {
                col: enhanced_features.get(col, 0.0) for col in self.active_features
            }
            df_input = pd.DataFrame([final_input])

            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(df_input)[0]
                self.fatigue_prob = round(probabilities[1] * 100, 1)
            else:
                decision = self.model.decision_function(df_input)[0]
                prob = 1.0 / (1.0 + np.exp(-decision))
                self.fatigue_prob = round(prob * 100, 1)

            if self.fatigue_prob > 70.0:
                self.current_status = "FATIGUE WARNING"
            else:
                self.current_status = "Safe"

        except Exception as e:
            # 如果算力崩溃，拦截报错并输出到终端，保证视频流不断开
            print(f"\n[ERROR] 💥 AI 推理时发生崩溃，已被系统拦截: {str(e)}")
            traceback.print_exc()
            print("--------------------------------------------------\n")
            self.current_status = "Error: Check Terminal"

    def _draw_hud(self, frame, feature):
        display_frame = frame.copy()

        # 状态指示灯颜色逻辑
        if self.current_status == "FATIGUE WARNING":
            hud_color = (0, 0, 255)
            cv2.putText(
                display_frame,
                "WARNING: FATIGUE DETECTED!",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )
        elif self.current_status == "Safe":
            hud_color = (0, 255, 0)
        elif self.current_status == "Calibrating":
            hud_color = (255, 255, 0)
            cv2.putText(
                display_frame,
                "CALIBRATING BASELINE...",
                (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                hud_color,
                3,
            )
        else:
            hud_color = (0, 255, 255)

        # 核心数据面板
        if self.current_status in ["Safe", "FATIGUE WARNING"]:
            cv2.putText(
                display_frame,
                f"PERCLOS: {self.perclos*100:.1f}%",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                hud_color,
                2,
            )
            cv2.putText(
                display_frame,
                f"AI Prob: {self.fatigue_prob}%",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                hud_color,
                2,
            )

        # FPS 显示
        cv2.putText(
            display_frame,
            f"FPS: {self.fps:.1f}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        return display_frame

    def close(self):
        self.extractor.close()
