import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import os
import io

# ▼▼▼ 設定 (ここを合わせてください) ▼▼▼
# 1. 計算用の全範囲 (モデル学習時と同じにする)
X_RANGE_CALC = (-2000.0, 2000.0)
Z_RANGE_CALC = (-1000.0, 1000.0)

# 2. 動画の表示範囲 (ここを狭くすれば大きく見えます！)
# ベッドより少し広いくらい (-200~1100, -600~600) がおすすめ
DISPLAY_X = (-400, 1400)   # 表示したいX範囲
DISPLAY_Z = (-800, 800)    # 表示したいZ範囲

# 3. ベッドの物理サイズ (枠線描画用)
BED_X = (0, 1980)    # 長さ
BED_Z = (0, 900)     # 幅 (中心が0でない場合は補正が必要。今は0~900と仮定)
# ※もしベッド中心がZ=0なら BED_Z = (-450, 450) にしてください

TIME_STEPS = 20
MODEL_PATH = "models/model_final_complete.keras"
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
    
    # Logic 5 Swap
    pred_x = x_range[0] + norm_w * (x_range[1] - x_range[0])
    pred_z = z_range[0] + norm_h * (z_range[1] - z_range[0])
    return pred_x, pred_z

def main():
    if not os.path.exists(MODEL_PATH):
        print("❌ モデルがありません")
        return
    model = tf.keras.models.load_model(MODEL_PATH)

    print("データ読み込み中...")
    try:
        input_df = pd.read_csv("inputs/input2.csv") # Bさんデータ
        result_df = pd.read_csv("inputs/result2.csv")
    except:
        return
    
    # 正規化の適用（もしモデル学習時に正規化していたらここでも必要！）
    # input_df = input_df.fillna(-120.0)
    # ... (正規化コードがあればここに挿入) ...
    
    # 簡易的に欠損埋めのみ
    input_df = input_df.fillna(-120.0)
    rssi_cols = [c for c in input_df.columns if "rssi" in c]
    X_raw = input_df[rssi_cols].values.astype(np.float32)

    parts = ["Head", "Heart", "R.Sho", "L.Sho", "R.Hip", "L.Hip"]
    full_parts_names = ["Head", "Heart", "Rshoulder", "Lshoulder", "Rhip", "Lhip"]
    
    target_cols = []
    for part in full_parts_names:
        target_cols.extend([f"{part}_X", f"{part}_Z"]) 
    y_true_coords = result_df[target_cols].values.astype(np.float32)

    X_seq = []
    y_true_viz = []
    for i in range(START_FRAME, START_FRAME + VIZ_FRAMES):
        if i + TIME_STEPS < len(X_raw):
            X_seq.append(X_raw[i : i + TIME_STEPS])
            y_true_viz.append(y_true_coords[i + TIME_STEPS])
    X_seq = np.array(X_seq)
    y_true_viz = np.array(y_true_viz)

    print("予測中...")
    preds = model.predict(X_seq, verbose=0)
    
    frames = []
    print("画像を生成中...")
    
    bones = [
        (0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (1, 4), (1, 5), (2, 3), (4, 5)
    ]

    for k in range(len(preds)):
        # 背景用ヒートマップ (全パーツの合計)
        heatmap_sum = np.sum(preds[k], axis=-1)
        heatmap_norm = (heatmap_sum - heatmap_sum.min()) / (heatmap_sum.max() - heatmap_sum.min() + 1e-6)
        
        plt.figure(figsize=(8, 10)) # 縦長画像
        
        # 1. 背景ヒートマップ描画 (interpolation='bilinear'で滑らかに！)
        # aspect='equal' にして歪みを防ぐ
        plt.imshow(heatmap_norm, cmap='magma', origin='upper', aspect='equal', interpolation='bilinear',
                   extent=[X_RANGE_CALC[0], X_RANGE_CALC[1], Z_RANGE_CALC[1], Z_RANGE_CALC[0]], alpha=0.8)
        
        # 2. ベッドの枠線を描画 (これが重要！)
        # Zが横軸(imshowの列方向)、Xが縦軸(imshowの行方向)のプロットに注意
        # imshowのextent指定により、通常のplot(Z, X)ではなく、plot(X, Z)などの軸定義に依存します
        # ここではextent=[Xmin, Xmax, Zmax, Zmin] (origin=upper) なので、横軸=X, 縦軸=Z です。
        # ※ユーザーの座標系設定(X=長辺, Z=短辺)に合わせて描画します。
        
        # ベッド矩形 (Width=900, Length=1980)
        # Z軸中心が0の場合: (-450, 450), X軸 (0, 1980)
        bed_z_min, bed_z_max = -450, 450
        bed_x_min, bed_x_max = 0, 1980
        
        # Rectangle((x, y), width, height)
        rect = Rectangle((bed_x_min, bed_z_max), # 左上 (X, Z) ※Z軸反転に注意
                         bed_x_max - bed_x_min,  # Width (X方向)
                         bed_z_min - bed_z_max,  # Height (Z方向: 下に向かってマイナスならこう書く)
                         linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--', label='Bed Area')
        
        # 現在のplot設定: X軸が横、Z軸が縦になっているか確認
        # extentの順序 [X_min, X_max, Z_max, Z_min] なので、
        # 横軸 = X, 縦軸 = Z (上の方が値が大きいZ_max, 下がZ_min)
        
        ax = plt.gca()
        # ベッド枠を手動でLinePlotで描く方が確実
        plt.plot([bed_x_min, bed_x_max, bed_x_max, bed_x_min, bed_x_min], 
                 [bed_z_min, bed_z_min, bed_z_max, bed_z_max, bed_z_min],
                 color='cyan', linestyle='--', linewidth=2, label='Bed')
        
        # 3. 骨格座標の計算と描画
        px_list = []
        pz_list = []
        for j in range(6):
            # 予測座標
            px, pz = get_coords_logic5(preds[k, :, :, j], X_RANGE_CALC, Z_RANGE_CALC)
            px_list.append(px)
            pz_list.append(pz)

        # 骨
        for b_start, b_end in bones:
            plt.plot([px_list[b_start], px_list[b_end]], [pz_list[b_start], pz_list[b_end]], 
                     c='white', linewidth=5, alpha=0.7)
            plt.plot([px_list[b_start], px_list[b_end]], [pz_list[b_start], pz_list[b_end]], 
                     c='lime', linewidth=2, alpha=1.0) # 色を鮮やかに

        # 関節
        plt.scatter(px_list, pz_list, c='white', s=100, edgecolors='black', zorder=5)
        
        # タイトルと範囲設定
        plt.title(f"Prediction Frame {START_FRAME + k}", fontsize=14)
        plt.xlabel("Bed Length (X) [mm]")
        plt.ylabel("Bed Width (Z) [mm]")
        
        # ★★★ ここで表示範囲を絞る（ズーム） ★★★
        plt.xlim(DISPLAY_X[0], DISPLAY_X[1])
        # Z軸は imshow の origin='upper' (上がMax, 下がMin) に合わせる
        plt.ylim(DISPLAY_Z[1], DISPLAY_Z[0]) 
        
        plt.grid(True, color='white', alpha=0.2)

        # 保存
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        plt.close()
        buf.seek(0)
        frames.append(Image.open(buf))

    print("GIF保存中...")
    frames[0].save("skeleton_viz_v2.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
    print("✅ 作成完了: skeleton_viz_v2.gif")

if __name__ == "__main__":
    main()