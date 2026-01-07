import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os
import io

# ▼▼▼ 設定 ▼▼▼
# 【ここにIDを指定】
NAME = "oka"  # 先ほどと同じIDを指定

# パスの自動生成
INPUT_FILE_PATH = f"inputs/input_{NAME}.csv"
RESULT_FILE_PATH = f"inputs/result_{NAME}.csv"
MODEL_PATH = f"models/model_finetuned_{NAME}.keras"  # ファインチューニング済みモデルを使う場合
# MODEL_PATH = "models/model_base.keras"  # ベースモデルを使う場合

# 範囲設定
X_RANGE_CALC = (-2000.0, 2000.0)
Z_RANGE_CALC = (-1000.0, 1000.0)
DISPLAY_X = (-1200, 1200)
DISPLAY_Z = (-800, 800)
BED_X = (-990, 990)
BED_Z = (-450, 450)

TIME_STEPS = 20 

START_OFFSET = 0      # 何フレーム目から始めるか (通常 0)
SKIP_STEP = 600      # 何フレームごとに表示するか (1000フレーム飛ばし)
GIF_DURATION = 500    # 1コマの表示時間 (ミリ秒) -> 500ms = 0.5秒
# ▲▲▲▲▲▲▲▲▲▲▲▲▲

def get_coords_logic5(heatmap, x_range, z_range):
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
    
    pred_x = x_range[0] + norm_w * (x_range[1] - x_range[0])
    pred_z = z_range[0] + norm_h * (z_range[1] - z_range[0])
    return pred_x, pred_z

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ モデルが見つかりません: {MODEL_PATH}")
        return

    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"データ読み込み中 (ID: {NAME})...")
    try:
        input_df = pd.read_csv(INPUT_FILE_PATH) 
        result_df = pd.read_csv(RESULT_FILE_PATH)
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        return
    
    input_df = input_df.fillna(-120.0)
    rssi_cols = [c for c in input_df.columns if "rssi" in c]
    X_raw = input_df[rssi_cols].values.astype(np.float32)
    
    print(f"📄 データ総行数: {len(X_raw)}")

    # 正規化の適用 (Global or SubjectSpecific)
    mean_path = "models/train_mean.npy"
    std_path = "models/train_std.npy"
    
    if os.path.exists(mean_path) and os.path.exists(std_path):
        print("✅ 正規化統計量を適用します (Global)。")
        mean = np.load(mean_path)
        std = np.load(std_path)
        X_raw = (X_raw - mean) / std
    else:
        print("⚠️ 正規化ファイルが見つかりません。SubjectSpecificで代用します。")
        mean = np.mean(X_raw, axis=0)
        std = np.std(X_raw, axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        X_raw = (X_raw - mean) / std

    full_parts_names = ["Head", "Heart", "Rshoulder", "Lshoulder", "Rhip", "Lhip"]
    target_cols = []
    for part in full_parts_names:
        target_cols.extend([f"{part}_X", f"{part}_Z"]) 
    y_true_coords = result_df[target_cols].values.astype(np.float32)

    # --- ダイジェスト用インデックス作成 ---
    # 0から始めて、データの最後まで SKIP_STEP 刻みでインデックスを作成
    # ただし、予測には TIME_STEPS 分の過去データが必要なので、それを考慮してループ上限を決める
    max_idx = len(X_raw) - TIME_STEPS
    target_indices = range(START_OFFSET, max_idx, SKIP_STEP)
    
    print(f"🎬 ダイジェスト作成モード: {len(target_indices)} 枚のフレームを生成します (間隔: {SKIP_STEP})")
    # -----------------------

    # データ切り出し
    X_seq = []
    y_true_viz = []
    frame_ids = []
    
    for i in target_indices:
        X_seq.append(X_raw[i : i + TIME_STEPS])
        y_true_viz.append(y_true_coords[i + TIME_STEPS])
        frame_ids.append(i + TIME_STEPS) # 表示用時刻 (入力の末尾時刻)
        
    X_seq = np.array(X_seq)
    y_true_viz = np.array(y_true_viz)

    print("一括予測中...")
    preds = model.predict(X_seq, verbose=1)

    frames = []
    print("画像を生成中...")
    
    bones = [
        (0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (1, 4), (1, 5), (2, 3), (4, 5)
    ]

    for k in range(len(preds)):
        heatmap_sum = np.sum(preds[k], axis=-1)
        heatmap_norm = (heatmap_sum - heatmap_sum.min()) / (heatmap_sum.max() - heatmap_sum.min() + 1e-6)
        
        plt.figure(figsize=(8, 10))
        
        plt.imshow(heatmap_norm, cmap='magma', origin='upper', aspect='equal', interpolation='bilinear',
                   extent=[X_RANGE_CALC[0], X_RANGE_CALC[1], Z_RANGE_CALC[1], Z_RANGE_CALC[0]], alpha=0.6)
        
        bx = [BED_X[0], BED_X[1], BED_X[1], BED_X[0], BED_X[0]]
        bz = [BED_Z[0], BED_Z[0], BED_Z[1], BED_Z[1], BED_Z[0]]
        plt.plot(bx, bz, color='cyan', linestyle='-', linewidth=2, label='Bed Frame')
        
        px_list, pz_list = [], []
        tx_list, tz_list = [], []
        
        for j in range(6):
            px, pz = get_coords_logic5(preds[k, :, :, j], X_RANGE_CALC, Z_RANGE_CALC)
            px_list.append(px)
            pz_list.append(pz)
            tx_list.append(y_true_viz[k, j*2])
            tz_list.append(y_true_viz[k, j*2+1])

        # Ground Truth
        for i, (b_start, b_end) in enumerate(bones):
            label = "Ground Truth" if i == 0 else ""
            plt.plot([tx_list[b_start], tx_list[b_end]], [tz_list[b_start], tz_list[b_end]], 
                     c='blue', linewidth=3, alpha=0.7, linestyle='--', label=label)
        plt.scatter(tx_list, tz_list, c='blue', s=80, edgecolors='white', zorder=4, alpha=0.8)

        # Prediction
        for i, (b_start, b_end) in enumerate(bones):
            label = "Prediction" if i == 0 else ""
            plt.plot([px_list[b_start], px_list[b_end]], [pz_list[b_start], pz_list[b_end]], 
                     c='lime', linewidth=2, alpha=1.0, label=label)
        plt.scatter(px_list, pz_list, c='white', s=100, edgecolors='black', zorder=5)
        
        # タイトルにフレーム番号を表示
        current_frame = frame_ids[k]
        plt.title(f"Frame {current_frame} (Total: {len(X_raw)})", fontsize=14)
        plt.xlabel("X (Long) [mm]")
        plt.ylabel("Z (Short) [mm]")
        plt.legend(loc='upper right')
        plt.xlim(DISPLAY_X[0], DISPLAY_X[1])
        plt.ylim(DISPLAY_Z[0], DISPLAY_Z[1])
        plt.grid(True, color='white', alpha=0.2)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=80)
        plt.close()
        buf.seek(0)
        frames.append(Image.open(buf))
        
        if (k+1) % 10 == 0:
            print(f" ... {k+1}/{len(preds)} 枚生成完了")

    print("GIF保存中...")
    save_filename = f"images/digest_view_{NAME}.gif"
    
    # loop=0 で無限ループ、durationで1コマの時間を指定
    frames[0].save(save_filename, save_all=True, append_images=frames[1:], duration=GIF_DURATION, loop=0)
    print(f"✅ 作成完了: {save_filename}")

if __name__ == "__main__":
    main()