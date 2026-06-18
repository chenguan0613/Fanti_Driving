import pandas as pd
from pathlib import Path


def enhance_features(input_path: str, output_path: str):
    print(f"[INFO] 正在读取原始特征文件: {input_path}")
    df = pd.read_csv(input_path)

    # 确保数据中存在 subject_id 和 label，这是跨视频基线计算的前提
    if "subject_id" not in df.columns or "label" not in df.columns:
        raise ValueError(
            "数据集中缺少 'subject_id' 或 'label' 列，无法进行严谨的基线归一化！"
        )

    # 需要被归一化的绝对特征列表
    features_to_norm = ["ear_mean", "mar_max", "pitch_std", "yaw_std", "blink_rate"]

    print("[INFO] 正在提取受试者的绝对清醒基线 (label=0)...")

    # 1. 过滤出所有真正处于清醒状态的数据 (label == 0)
    awake_df = df[df["label"] == 0]

    # 2. 按受试者 (subject_id) 分组，计算他们在清醒状态下的特征绝对平均值
    # 这将生成一个索引为 subject_id 的对照表 (Lookup Table)
    baseline_table = awake_df.groupby("subject_id")[features_to_norm].mean()

    print("[INFO] 正在进行全量数据的向量化归一化...")
    for feat in features_to_norm:
        # 3. 将每个受试者的清醒基线值，映射 (map) 回完整的表格中
        baseline_col = df["subject_id"].map(baseline_table[feat])

        # [防雷机制] 如果某个极其特殊的人只有疲劳视频，没有 label=0 的数据，
        # baseline_col 就会出现 NaN。我们就退而求其次，用他所有视频的均值兜底。
        if baseline_col.isnull().any():
            fallback_mean = df.groupby("subject_id")[feat].mean()
            baseline_col = baseline_col.fillna(df["subject_id"].map(fallback_mean))

        # 4. 执行真正的归一化：(当前值 - 绝对清醒基线) / 绝对清醒基线
        df[f"{feat}_norm"] = (df[feat] - baseline_col) / (baseline_col + 1e-6)

    # ==========================================
    # 时序动态特征 (一阶差分 Velocity)
    # ==========================================
    print("[INFO] 正在计算动态一阶差分特征 (Velocity)...")

    # 注意：计算差分时，必须按 video_id 分组！绝对不能按 subject_id！
    # 否则一个人的清醒视频结尾，会去减他疲劳视频的开头，造成巨大的时序错乱！
    df = df.sort_values(by=["video_id"]).reset_index(drop=True)
    df["ear_velocity"] = df.groupby("video_id")["ear_mean_norm"].diff().fillna(0.0)
    df["pitch_velocity"] = df.groupby("video_id")["pitch_std_norm"].diff().fillna(0.0)

    # ==========================================
    # 高阶特征交叉 (Feature Crossing)
    # ==========================================
    print("[INFO] 正在生成交叉组合特征...")
    df["fatigue_index"] = df["perclos"] * df["ear_mean_norm"].abs()

    # 5. 清理不必要的中间列，保存新的增强数据集
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] 严谨的特征增强完毕！数据已保存至: {output_path}")


if __name__ == "__main__":
    INPUT_CSV = "src/dataset/training_features.csv"
    OUTPUT_CSV = "src/dataset/training_features_enhanced.csv"

    if Path(INPUT_CSV).exists():
        enhance_features(INPUT_CSV, OUTPUT_CSV)
    else:
        print(f"[ERROR] 找不到输入文件 {INPUT_CSV}，请检查路径！")
