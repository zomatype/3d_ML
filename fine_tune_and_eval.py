import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
import os
import gc

# ▼▼▼ 設定 ▼▼▼
X_RANGE = (-1000.0, 1200.0)
Z_RANGE = (-600.0, 1000.0)
MAP_SIZE = (64, 32)
TIME_STEPS = 20
MODEL_PATH = "models/seq20_Without_Pooling.keras"

# ランダムに混ぜて学習する
FINE_TUNE_RATIO = 0.2  # 20%
EPOCHS = 15
LEARNING_RATE = 0.0001
# ▲▲▲▲▲▲▲▲▲▲▲▲▲

class HeatmapGeneratorLogic5(Sequence):
    def __init__(self, x_set, y_coords, batch_size, map_size, x_range, z_range, sigma=1.5):
        self.x = x_set
        self.y = y_coords
        self.batch_size = batch_size
        self.h, self.w = map_size
        self.x_range = x_range
        self.z_range = z_range
        self.sigma = sigma
        self.indices = np.arange(len(self.x))
        x_grid = np.arange(0, self.w, 1, np.float32)
        y_grid = np.arange(0, self.h, 1, np.float32)
        self.X_grid, self.Y_grid = np.meshgrid(x_grid, y_grid)

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_y_coords = self.y[inds]
        return batch_x, self._generate_heatmaps(batch_y_coords)

    def _generate_heatmaps(self, coords):
        batch_size = len(coords)
        num_keypoints = coords.shape[1] // 2
        heatmaps = np.zeros((batch_size, self.h, self.w, num_keypoints), dtype=np.float32)
        min_x, max_x = self.x_range
        min_z, max_z = self.z_range

        for i in range(batch_size):
            flat_coords = coords[i]
            for k in range(num_keypoints):
                real_x = flat_coords[k*2]
                real_z = flat_coords[k*2 + 1]
                
                if real_x < min_x or real_x > max_x or real_z < min_z or real_z > max_z: continue

                # Logic 5 (Swap)
                norm_x = (real_x - min_x) / (max_x - min_x)
                norm_z = (real_z - min_z) / (max_z - min_z)
                
                center_w = norm_x * (self.w - 1)
                center_h = norm_z * (self.h - 1)

                d2 = (self.X_grid - center_w)**2 + (self.Y_grid - center_h)**2
                heatmaps[i, :, :, k] = np.exp(-d2 / (2 * self.sigma**2))
        return heatmaps
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

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
    
    # Logic 5
    pred_x = x_range[0] + norm_w * (x_range[1] - x_range[0])
    pred_z = z_range[0] + norm_h * (z_range[1] - z_range[0])
    return pred_x, pred_z

def main():
    if not os.path.exists(MODEL_PATH):
        print("❌ ベースモデルなし")
        return
    
    print(f"ベースモデル読み込み: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    model.compile(optimizer=opt, loss='mse')

    print("Bさんデータ読み込み...")
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
    y_coords = result_df[target_cols].values.astype(np.float32)
    
    X_seq, y_seq = [], []
    for i in range(len(X_raw) - TIME_STEPS):
        X_seq.append(X_raw[i : i + TIME_STEPS])
        y_seq.append(y_coords[i + TIME_STEPS])
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # ★★★ ここが重要！ランダムに混ぜて分割 ★★★
    # shuffle=True (デフォルト) なので、動いている区間も学習データに入る
    X_ft, X_eval, y_ft, y_eval = train_test_split(
        X_seq, y_seq, train_size=FINE_TUNE_RATIO, random_state=42, shuffle=True
    )
    
    print(f"ランダム分割完了: Tuning={len(X_ft)}, Eval={len(X_eval)}")

    print("\n🚀 Fine-Tuning開始 (ランダムデータ)...")
    ft_gen = HeatmapGeneratorLogic5(X_ft, y_ft, 16, MAP_SIZE, X_RANGE, Z_RANGE)
    model.fit(ft_gen, epochs=EPOCHS, verbose=1)
    
    print("\n📊 最終評価 (残りの80%データ)...")
    total_error = 0.0
    part_errors = {p: 0.0 for p in parts}
    count = 0
    BATCH_SIZE = 100
    
    for i in range(0, len(X_eval), BATCH_SIZE):
        end_ix = min(i + BATCH_SIZE, len(X_eval))
        X_batch = X_eval[i : end_ix]
        y_batch = y_eval[i : end_ix]
        
        preds = model.predict(X_batch, verbose=0)
        for k in range(len(preds)):
            for j, part in enumerate(parts):
                px, pz = get_coords_logic5(preds[k, :, :, j], X_RANGE, Z_RANGE)
                tx, tz = y_batch[k, j*2], y_batch[k, j*2+1]
                dist = np.sqrt((px - tx)**2 + (pz - tz)**2)
                part_errors[part] += dist
                total_error += dist
            count += 1

    avg_error = total_error / (count * len(parts))
    
    print("\n" + "="*50)
    print(f"🎉 最終結果 (Random Split Fine-Tuning)")
    print("="*50)
    print(f"🏆 平均誤差: {avg_error:.2f} mm")
    print("-" * 50)
    for part in parts:
        print(f"  - {part:10s}: {part_errors[part]/count:.2f} mm")
    print("="*50)

if __name__ == "__main__":
    main()