import pandas as pd
import numpy as np
import tensorflow as tf
import os
import gc

# ▼▼▼ 設定 ▼▼▼
X_RANGE = (-1000.0, 1200.0)
Z_RANGE = (-600.0, 1000.0)
TIME_STEPS = 20
MODEL_PATH = "models/seq20_Without_Pooling.keras"

# スムージング設定 (フレーム数)
# 動きのノイズを消して精度を上げる (5〜10推奨)
SMOOTHING_WINDOW = 5
# ▲▲▲▲▲▲▲▲▲▲▲▲▲

def get_coords_logic5(heatmap, x_range, z_range):
    """
    【優勝ロジック: Logic 5】
    Swap = True (軸入れ替え)
    Invert = False (反転なし)
    """
    h, w = heatmap.shape
    threshold = np.max(heatmap) * 0.2
    heatmap_thresh = np.where(heatmap > threshold, heatmap, 0)
    total_mass = np.sum(heatmap_thresh)
    
    if total_mass <= 1e-6:
        # 重心計算できない場合は最大値
        y_idx, x_idx = np.unravel_index(np.argmax(heatmap), (h, w))
        center_w = x_idx
        center_h = y_idx
    else:
        x_grid = np.arange(w)
        y_grid = np.arange(h)
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
        
        center_w = np.sum(X_mesh * heatmap_thresh) / total_mass
        center_h = np.sum(Y_mesh * heatmap_thresh) / total_mass
    
    # 正規化 (0.0 ~ 1.0)
    norm_w = center_w / (w - 1)
    norm_h = center_h / (h - 1)
    
    # === Logic 5: Swap(No Inv) ===
    # 画像の Width を 物理 X に
    # 画像の Height を 物理 Z に
    val_x = norm_w
    val_z = norm_h
    
    # 物理座標へ変換
    pred_x = x_range[0] + val_x * (x_range[1] - x_range[0])
    pred_z = z_range[0] + val_z * (z_range[1] - z_range[0])
    
    return pred_x, pred_z

def main():
    print(f"モデル読み込み中: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print("❌ モデルが見つかりません")
        return
    model = tf.keras.models.load_model(MODEL_PATH)

    print("データ読み込み中...")
    try:
        input_df = pd.read_csv("inputs/input2.csv")
        result_df = pd.read_csv("inputs/result2.csv")
    except:
        print("❌ データなし")
        return

    input_df = input_df.fillna(-120.0)
    rssi_cols = [c for c in input_df.columns if "rssi" in c]
    X_raw = input_df[rssi_cols].values.astype(np.float32)

    parts = ["Head", "Heart", "Rshoulder", "Lshoulder", "Rhip", "Lhip"]
    target_cols = []
    for part in parts:
        target_cols.extend([f"{part}_X", f"{part}_Z"]) 
    y_true_coords = result_df[target_cols].values.astype(np.float32)

    # 全データ予測
    num_samples = len(X_raw) - TIME_STEPS
    print(f"全データ評価開始 ({num_samples} frames)...")
    
    all_preds_x = []
    all_preds_z = []
    
    BATCH_SIZE = 256
    
    for i in range(0, num_samples, BATCH_SIZE):
        end_ix = min(i + BATCH_SIZE, num_samples)
        X_batch = []
        for k in range(i, end_ix):
            X_batch.append(X_raw[k : k + TIME_STEPS])
        X_batch = np.array(X_batch)
        
        preds = model.predict(X_batch, verbose=0)
        
        for k in range(len(preds)):
            row_x = []
            row_z = []
            for j in range(len(parts)):
                px, pz = get_coords_logic5(preds[k, :, :, j], X_RANGE, Z_RANGE)
                row_x.append(px)
                row_z.append(pz)
            all_preds_x.append(row_x)
            all_preds_z.append(row_z)
            
        if (i + BATCH_SIZE) % 10000 < BATCH_SIZE:
             print(f"  Processed {end_ix}...")
        gc.collect()

    # Numpy配列化
    pred_x_arr = np.array(all_preds_x)
    pred_z_arr = np.array(all_preds_z)
    
    # === スムージング処理 (Moving Average) ===
    print(f"\nスムージング適用中 (Window={SMOOTHING_WINDOW})...")
    
    # DataFrameを使って一括処理
    df_x = pd.DataFrame(pred_x_arr)
    df_z = pd.DataFrame(pred_z_arr)
    
    df_x_smooth = df_x.rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean()
    df_z_smooth = df_z.rolling(window=SMOOTHING_WINDOW, center=True, min_periods=1).mean()
    
    pred_x_final = df_x_smooth.values
    pred_z_final = df_z_smooth.values
    
    # 誤差計算
    total_error = 0.0
    part_errors = {p: 0.0 for p in parts}
    count = 0
    
    y_true_valid = y_true_coords[TIME_STEPS : TIME_STEPS + num_samples]
    
    for i in range(num_samples):
        for j, part in enumerate(parts):
            px = pred_x_final[i, j]
            pz = pred_z_final[i, j]
            
            tx = y_true_valid[i, j*2]
            tz = y_true_valid[i, j*2+1]
            
            dist = np.sqrt((px - tx)**2 + (pz - tz)**2)
            part_errors[part] += dist
            total_error += dist
        count += 1
        
    avg_error = total_error / (count * len(parts))
    
    print("\n" + "="*50)
    print(f"🎉 最終評価結果 (Logic 5 + Smoothing)")
    print("="*50)
    print(f"🏆 平均誤差: {avg_error:.2f} mm")
    print("-" * 50)
    for part in parts:
        print(f"  - {part:10s}: {part_errors[part]/count:.2f} mm")
    print("="*50)

if __name__ == "__main__":
    main()