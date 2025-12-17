import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import os
import io

# ▼▼▼ 設定 ▼▼▼
# 全体の範囲（ヒートマップ生成に必要）
X_RANGE_FULL = (-1000.0, 1200.0)
Z_RANGE_FULL = (-600.0, 1000.0)
MAP_SIZE = (64, 32)
TIME_STEPS = 20
MODEL_PATH = "models/model_final_complete.keras"

# 動画の設定
VIZ_FRAMES = 60 
START_FRAME = 350

# ★★★ ズーム設定 (ここを変えました) ★★★
# 指定範囲: X=500~800, Z=-200~150
# 少し余白(マージン)を持たせて見やすくします
ZOOM_X = (450, 850)    # 横軸 (Bed Length)
ZOOM_Z = (-250, 200)   # 縦軸 (Bed Width) 注意: 上がマイナス
# ▲▲▲▲▲▲▲▲▲▲▲▲▲

def main():
    if not os.path.exists(MODEL_PATH):
        print("❌ モデルがありません")
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

    parts = ["Head", "Heart", "R.Sho", "L.Sho", "R.Hip", "L.Hip"]
    full_parts_names = ["Head", "Heart", "Rshoulder", "Lshoulder", "Rhip", "Lhip"]
    
    target_cols = []
    for part in full_parts_names:
        target_cols.extend([f"{part}_X", f"{part}_Z"]) 
    y_true_coords = result_df[target_cols].values.astype(np.float32)

    safe_start = min(START_FRAME, len(X_raw) - TIME_STEPS - VIZ_FRAMES)
    
    X_seq = []
    y_true_viz = []
    for i in range(safe_start, safe_start + VIZ_FRAMES):
        X_seq.append(X_raw[i : i + TIME_STEPS])
        y_true_viz.append(y_true_coords[i + TIME_STEPS])
    X_seq = np.array(X_seq)
    y_true_viz = np.array(y_true_viz)

    print("予測中...")
    preds = model.predict(X_seq, verbose=0)
    
    frames = []
    print("ズーム画像を生成中...")
    
    bones = [
        (0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (1, 4), (1, 5), (2, 3), (4, 5)
    ]

    for k in range(len(preds)):
        heatmap_sum = np.sum(preds[k], axis=-1)
        heatmap_norm = (heatmap_sum - heatmap_sum.min()) / (heatmap_sum.max() - heatmap_sum.min() + 1e-6)
        
        # ズーム時は正方形に近い比率で見やすくする
        plt.figure(figsize=(10, 10))
        
        # 1. 背景ヒートマップ描画 (全体を描画してからZoomする)
        plt.imshow(heatmap_norm, cmap='magma', origin='upper', aspect='auto',
                   extent=[X_RANGE_FULL[0], X_RANGE_FULL[1], Z_RANGE_FULL[1], Z_RANGE_FULL[0]])
        
        px_list = []
        pz_list = []
        for j in range(6):
            tx = y_true_viz[k, j*2]
            tz = y_true_viz[k, j*2+1]
            px_list.append(tx)
            pz_list.append(tz)
        
        # 2. 骨（線）
        for b_start, b_end in bones:
            x_vals = [px_list[b_start], px_list[b_end]]
            z_vals = [pz_list[b_start], pz_list[b_end]]
            plt.plot(x_vals, z_vals, c='white', linewidth=6, alpha=0.7) # 太く
            plt.plot(x_vals, z_vals, c='blue', linewidth=3, alpha=1.0)

        # 3. 関節（点）
        plt.scatter(px_list, pz_list, c='white', s=400, edgecolors='black', linewidth=2, zorder=5) # 大きく
        
        # 4. ラベル
        for j, name in enumerate(parts):
            plt.text(px_list[j], pz_list[j] - 20, name, 
                     color='cyan', fontsize=16, fontweight='bold', ha='center',
                     bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=2))

        plt.title(f"Zoomed Prediction (Frame {START_FRAME + k})", fontsize=18)
        plt.xlabel("Bed Length (X) [mm]", fontsize=14)
        plt.ylabel("Bed Width (Z) [mm]", fontsize=14)
        
        # ★★★ ここで表示範囲を限定（ズーム） ★★★
        plt.xlim(ZOOM_X[0], ZOOM_X[1])
        
        # Z軸の指定に注意: imshowのorigin='upper'設定に合わせる
        # 上(Top)が -250 (値は小さい), 下(Bottom)が 200 (値は大きい)
        # plt.ylim(Bottom, Top) の順で指定すると反転しない
        plt.ylim(ZOOM_Z[1], ZOOM_Z[0]) 
        
        plt.grid(True, color='white', alpha=0.3)
        plt.tick_params(labelsize=12)

        # 保存
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        buf.seek(0)
        frames.append(Image.open(buf))

    print("GIF保存中...")
    frames[0].save(
        "skeleton_zoom.gif",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )
    # サンプル静止画
    frames[10].save("skeleton_zoom_sample.png")
    print("✅ 'skeleton_zoom.gif' を作成しました。指定範囲が拡大表示されています！")

if __name__ == "__main__":
    main()