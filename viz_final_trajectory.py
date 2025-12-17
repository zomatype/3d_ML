import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# ▼▼▼ 設定 ▼▼▼
X_RANGE = (-1000.0, 1200.0)
Z_RANGE = (-600.0, 1000.0)
TIME_STEPS = 20
MODEL_PATH = "models/model_final_complete.keras" # ランダムFT後のモデル
# ▲▲▲▲▲▲▲▲▲▲▲▲▲

def get_coords_logic5(heatmap, x_range, z_range):
    """ Logic 5: Swap (Width=X, Height=Z) """
    h, w = heatmap.shape
    threshold = np.max(heatmap) * 0.2
    heatmap_thresh = np.where(heatmap > threshold, heatmap, 0)
    total_mass = np.sum(heatmap_thresh)
    
    if total_mass <= 1e-6:
        y_idx, x_idx = np.unravel_index(np.argmax(heatmap), (h, w))
        center_w, center_h = x_idx, y_idx
    else:
        x_grid = np.arange(w)
        y_grid = np.arange(h)
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
        center_w = np.sum(X_mesh * heatmap_thresh) / total_mass
        center_h = np.sum(Y_mesh * heatmap_thresh) / total_mass
    
    norm_w = center_w / (w - 1)
    norm_h = center_h / (h - 1)
    
    # Swap Logic
    pred_x = x_range[0] + norm_w * (x_range[1] - x_range[0])
    pred_z = z_range[0] + norm_h * (z_range[1] - z_range[0])
    return pred_x, pred_z

def main():
    if not os.path.exists(MODEL_PATH):
        print("❌ モデルなし")
        return
    model = tf.keras.models.load_model(MODEL_PATH)

    print("データ読み込み中...")
    try:
        input_df = pd.read_csv("inputs/input2.csv")
        result_df = pd.read_csv("inputs/result2.csv")
    except:
        return
    input_df = input_df.fillna(-120.0)
    rssi_cols = [c for c in input_df.columns if "rssi" in c]
    X_raw = input_df[rssi_cols].values.astype(np.float32)

    parts = ["Head", "Heart", "Rshoulder", "Lshoulder", "Rhip", "Lhip"]
    target_cols = []
    for part in parts:
        target_cols.extend([f"{part}_X", f"{part}_Z"]) 
    y_true_coords = result_df[target_cols].values.astype(np.float32)

    # 動きが見える区間を可視化 (例えば中盤の300フレーム)
    start_frame = 300
    viz_len = 400 
    
    if start_frame + viz_len > len(X_raw): start_frame = 0

    X_seq = []
    y_true_viz = []
    for i in range(start_frame, start_frame + viz_len):
        X_seq.append(X_raw[i : i + TIME_STEPS])
        y_true_viz.append(y_true_coords[i + TIME_STEPS])
    X_seq = np.array(X_seq)
    y_true_viz = np.array(y_true_viz)

    print("予測中...")
    preds = model.predict(X_seq, verbose=0)

    # 描画 (HeadとRight Handあたりをプロット)
    target_parts_idx = [0, 2, 4] # Head, Rshoulder, Rhip
    
    plt.figure(figsize=(15, 5))
    
    for i, p_idx in enumerate(target_parts_idx):
        part_name = parts[p_idx]
        traj_pred_x, traj_pred_z = [], []
        traj_true_x, traj_true_z = [], []
        
        for k in range(len(preds)):
            px, pz = get_coords_logic5(preds[k, :, :, p_idx], X_RANGE, Z_RANGE)
            tx, tz = y_true_viz[k, p_idx*2], y_true_viz[k, p_idx*2+1]
            
            traj_pred_x.append(px)
            traj_pred_z.append(pz)
            traj_true_x.append(tx)
            traj_true_z.append(tz)
            
        # サブプロット (X-Z平面)
        plt.subplot(1, 3, i+1)
        plt.plot(traj_true_z, traj_true_x, label='True', color='blue', alpha=0.6, linewidth=2)
        plt.plot(traj_pred_z, traj_pred_x, label='Pred', color='red', linestyle='--', linewidth=2)
        plt.title(f"Trajectory: {part_name}")
        plt.xlabel("Z (Width) [mm]")
        plt.ylabel("X (Length) [mm]")
        plt.legend()
        plt.grid(True)
        plt.axis('equal')

    plt.tight_layout()
    plt.savefig("thesis_trajectory.png")
    print("\n✅ 'thesis_trajectory.png' を保存しました。")
    print("論文に載せられるクオリティになっているはずです！")

if __name__ == "__main__":
    main()