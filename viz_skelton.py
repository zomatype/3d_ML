import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os
import io

# ▼▼▼ 設定 ▼▼▼
# 範囲設定 (学習時と合わせる)
X_RANGE_CALC = (-2000.0, 2000.0)
Z_RANGE_CALC = (-1000.0, 1000.0)

# 表示範囲 (ズーム) - ベッド全体が見える範囲
DISPLAY_X = (-1200, 1200)
DISPLAY_Z = (-800, 800)

# ベッド枠 (原点中心)
BED_X = (-990, 990)
BED_Z = (-450, 450)

TIME_STEPS = 20

# ★★★ 注意: Fine-Tuning後に保存したモデルを指定してください ★★★
MODEL_PATH = "models/model_base.keras" 
# ※もし保存していなければ "models/model_final_complete.keras"

VIZ_FRAMES = 100 
START_FRAME = 300
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
        print("  先に fine_tune_and_eval.py でモデルを保存(save)してください。")
        return
    model = tf.keras.models.load_model(MODEL_PATH)

    print("データ読み込み中...")
    try:
        input_df = pd.read_csv("inputs/input2.csv") 
        result_df = pd.read_csv("inputs/result2.csv")
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        return
    
    # NaN埋め
    input_df = input_df.fillna(-120.0)
    
    rssi_cols = [c for c in input_df.columns if "rssi" in c]
    X_raw = input_df[rssi_cols].values.astype(np.float32)

    # 正規化の適用
    mean_path = "models/train_mean.npy"
    std_path = "models/train_std.npy"
    
    if os.path.exists(mean_path) and os.path.exists(std_path):
        print("✅ 正規化統計量を適用します。")
        mean = np.load(mean_path)
        std = np.load(std_path)
        X_raw = (X_raw - mean) / std
    else:
        print("⚠️ 正規化ファイルが見つかりません。学習条件と一致しているか確認してください。")

    full_parts_names = ["Head", "Heart", "Rshoulder", "Lshoulder", "Rhip", "Lhip"]
    target_cols = []
    for part in full_parts_names:
        target_cols.extend([f"{part}_X", f"{part}_Z"]) 
    y_true_coords = result_df[target_cols].values.astype(np.float32)

    # データ切り出し
    X_seq = []
    y_true_viz = []
    
    safe_start = min(START_FRAME, len(X_raw) - TIME_STEPS - VIZ_FRAMES)
    
    for i in range(safe_start, safe_start + VIZ_FRAMES):
        X_seq.append(X_raw[i : i + TIME_STEPS])
        y_true_viz.append(y_true_coords[i + TIME_STEPS])
    X_seq = np.array(X_seq)
    y_true_viz = np.array(y_true_viz)

    print("予測中...")
    preds = model.predict(X_seq, verbose=0)

    # ▼▼▼ 追加: ヒートマップの元気度チェック ▼▼▼
    max_val = np.max(preds)
    min_val = np.min(preds)
    avg_val = np.mean(preds)
    print(f"=== ヒートマップ状態チェック ===")
    print(f"Max: {max_val:.5f}")
    print(f"Min: {min_val:.5f}")
    print(f"Avg: {avg_val:.5f}")
    
    if max_val < 0.01:
        print("⚠️ 警告: ヒートマップの値が小さすぎます！モデルが何も検知していません。")
        print("   -> 正規化ミス、または学習不足の可能性があります。")
    else:
        print("✅ ヒートマップには反応があります。")
    
    frames = []
    print("画像を生成中...")
    
    bones = [
        (0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (1, 4), (1, 5), (2, 3), (4, 5)
    ]

    for k in range(len(preds)):
        heatmap_sum = np.sum(preds[k], axis=-1)
        heatmap_norm = (heatmap_sum - heatmap_sum.min()) / (heatmap_sum.max() - heatmap_sum.min() + 1e-6)
        
        plt.figure(figsize=(8, 10))
        
        # 背景ヒートマップ
        # extent=[Xmin, Xmax, Zmax, Zmin] (origin=upper, Z軸反転)
        plt.imshow(heatmap_norm, cmap='magma', origin='upper', aspect='equal', interpolation='bilinear',
                   extent=[X_RANGE_CALC[0], X_RANGE_CALC[1], Z_RANGE_CALC[1], Z_RANGE_CALC[0]], alpha=0.6)
        
        # ベッド枠描画
        bx = [BED_X[0], BED_X[1], BED_X[1], BED_X[0], BED_X[0]]
        bz = [BED_Z[0], BED_Z[0], BED_Z[1], BED_Z[1], BED_Z[0]]
        plt.plot(bx, bz, color='cyan', linestyle='-', linewidth=2, label='Bed Frame')
        
        # 骨格座標のリスト化
        px_list, pz_list = [], []
        tx_list, tz_list = [], []
        
        for j in range(6):
            # 予測
            px, pz = get_coords_logic5(preds[k, :, :, j], X_RANGE_CALC, Z_RANGE_CALC)
            px_list.append(px)
            pz_list.append(pz)
            # 正解 (Ground Truth)
            tx_list.append(y_true_viz[k, j*2])
            tz_list.append(y_true_viz[k, j*2+1])

        # --- 描画ループ ---
        
        # 1. 正解データ (Blue) - 先に描画
        for i, (b_start, b_end) in enumerate(bones):
            label = "Ground Truth" if i == 0 else "" # 凡例用
            plt.plot([tx_list[b_start], tx_list[b_end]], [tz_list[b_start], tz_list[b_end]], 
                     c='blue', linewidth=3, alpha=0.7, linestyle='--', label=label)
            
        # 正解の関節点
        plt.scatter(tx_list, tz_list, c='blue', s=80, edgecolors='white', zorder=4, alpha=0.8)

        # 2. 予測データ (Green/White) - 上に描画
        for i, (b_start, b_end) in enumerate(bones):
            label = "Prediction" if i == 0 else ""
            plt.plot([px_list[b_start], px_list[b_end]], [pz_list[b_start], pz_list[b_end]], 
                     c='lime', linewidth=2, alpha=1.0, label=label)
            
        # 予測の関節点
        plt.scatter(px_list, pz_list, c='white', s=100, edgecolors='black', zorder=5)
        
        # 凡例とタイトル
        plt.title(f"Comparison Frame {safe_start + k}", fontsize=14)
        plt.xlabel("X (Long) [mm]")
        plt.ylabel("Z (Short) [mm]")
        
        # 凡例を表示 (重複削除のため工夫済み)
        plt.legend(loc='upper right')

        plt.xlim(DISPLAY_X[0], DISPLAY_X[1])
        plt.ylim(DISPLAY_Z[0], DISPLAY_Z[1]) # Top->Bottom
        
        plt.grid(True, color='white', alpha=0.2)

        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=80)
        plt.close()
        buf.seek(0)
        frames.append(Image.open(buf))

    print("GIF保存中...")
    frames[0].save("skeleton_compare_v4.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
    print("✅ 作成完了: skeleton_compare_v4.gif")
    print("  - 青い破線・青い丸 : 正解データ (Ground Truth)")
    print("  - 緑の実線・白い丸 : 予測データ (Prediction)")

if __name__ == "__main__":
    main()