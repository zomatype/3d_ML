import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence
import os
import gc

# ▼▼▼ 設定 ▼▼▼
# 被験者名(ファイル名に使用)
NAME = "okabe"  # 例: "soma" 

# パスの自動生成
INPUT_FILE_PATH = f"inputs/input_{NAME}.csv"
RESULT_FILE_PATH = f"inputs/result_{NAME}.csv"
SAVE_MODEL_PATH = f"models/model_finetuned_{NAME}.keras"

# Z軸（短辺）: データ最大値 846.5mm をカバーするため、余裕を持って ±1000mm に設定
Z_RANGE = (-1000.0, 1000.0)  # 幅 2000mm

# X軸（長辺）: Z軸の幅(2000mm) の「2倍」の幅を確保して、比率2:1を維持
X_RANGE = (-2000.0, 2000.0)  # 幅 4000mm (-2000 ~ 2000)
MAP_SIZE = (64, 32)
TIME_STEPS = 20
MODEL_PATH = "models/model_base.keras"

# 学習データの使用割合 (0.0 にすると学習なしで評価のみ)
FINE_TUNE_RATIO = 0.2  # 20%
EPOCHS = 15
LEARNING_RATE = 0.0001

# ▼▼▼ 正規化の適用 ▼▼▼
# 【戦略選択】
# STRATEGY = "Global"  # Aさんの統計量を使う (従来手法)
STRATEGY = "SubjectSpecific"  # Bさんの統計量を使う (提案手法: キャリブレーション)
# ▲▲▲▲▲▲▲▲▲▲▲▲▲

class HeatmapGenerator(Sequence):
    def __init__(self, x_set, z_coords, batch_size, map_size, x_range, z_range, sigma=1.5):
        self.x = x_set
        self.z = z_coords
        self.batch_size = batch_size
        
        # map_size=(64, 32) は (Width, Height)
        self.w, self.h = map_size 
        
        self.x_range = x_range
        self.z_range = z_range
        self.sigma = sigma
        self.indices = np.arange(len(self.x))

        # グリッド作成 (Shape: Height x Width)
        x_grid = np.arange(0, self.w, 1, np.float32)
        z_grid = np.arange(0, self.h, 1, np.float32)
        self.X_grid, self.Z_grid = np.meshgrid(x_grid, z_grid)

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_z_coords = self.z[inds]
        batch_heatmaps = self._generate_heatmaps(batch_z_coords)
        return batch_x, batch_heatmaps

    def _generate_heatmaps(self, coords):
        batch_size = len(coords)
        num_keypoints = coords.shape[1] // 2
        
        # shape: (Batch, Height, Width, Channels) -> (B, 32, 64, 6)
        # モデルの出力形状 (32, 64) に合わせる
        heatmaps = np.zeros((batch_size, self.h, self.w, num_keypoints), dtype=np.float32)

        min_x, max_x = self.x_range
        min_z, max_z = self.z_range

        for i in range(batch_size):
            flat_coords = coords[i]
            for k in range(num_keypoints):
                real_x = flat_coords[k*2]
                real_z = flat_coords[k*2 + 1]

                # クリッピング
                if real_x < min_x: real_x = min_x
                if real_x > max_x: real_x = max_x
                if real_z < min_z: real_z = min_z
                if real_z > max_z: real_z = max_z

                norm_x = (real_x - min_x) / (max_x - min_x)
                norm_z = (real_z - min_z) / (max_z - min_z)

                center_x = norm_x * (self.w - 1)
                center_z = norm_z * (self.h - 1)

                d2 = (self.X_grid - center_x)**2 + (self.Z_grid - center_z)**2
                g = np.exp(-d2 / (2 * self.sigma**2))
                heatmaps[i, :, :, k] = g

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
        print(f"❌ ベースモデルなし: {MODEL_PATH}")
        print("rssi_to_pose_heatmap.py を実行してモデルを作成してください。")
        return
    
    print(f"ベースモデル読み込み: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    model.compile(optimizer=opt, loss='mse')

    print(f"検証用データ読み込み (ID: {NAME})...")
    try:
        # 設定で指定したパスを使用
        print(f" - Input: {INPUT_FILE_PATH}")
        print(f" - Result: {RESULT_FILE_PATH}")
        
        input_df = pd.read_csv(INPUT_FILE_PATH)
        result_df = pd.read_csv(RESULT_FILE_PATH)
    except FileNotFoundError:
        print(f"❌ ファイルが見つかりません: {INPUT_FILE_PATH} または {RESULT_FILE_PATH}")
        return
    except Exception as e:
        print(f"❌ 読み込みエラー: {e}")
        return

    # fillnaは正規化の前に行う
    input_df = input_df.fillna(-120.0) 
    
    rssi_cols = [c for c in input_df.columns if "rssi" in c]
    X_raw = input_df[rssi_cols].values.astype(np.float32)


    if STRATEGY == "Global":
        # 学習時の統計量をロード
        try:
            mean = np.load("models/train_mean.npy")
            std = np.load("models/train_std.npy")
            print("Strategy: Global Normalization (Using Train Stats)")
        except:
            print("⚠️ 統計量ファイルが見つかりません。SubjectSpecificに切り替えます。")
            mean = np.mean(X_raw, axis=0)
            std = np.std(X_raw, axis=0)
            std = np.where(std < 1e-6, 1.0, std)
    else:
        # その場のデータから計算 (自己正規化)
        mean = np.mean(X_raw, axis=0)
        std = np.std(X_raw, axis=0)
        std = np.where(std < 1e-6, 1.0, std)
        print("Strategy: Subject-Specific Normalization (Using Test Stats)")

    # 正規化適用
    X_raw = (X_raw - mean) / std
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

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
    
    # ▼▼▼ データ分割の修正 (時系列分割 & 0%対応) ▼▼▼
    if FINE_TUNE_RATIO > 0.0:
        # 時系列を維持して分割 (前半を学習、後半を評価)
        split_idx = int(len(X_seq) * FINE_TUNE_RATIO)
        
        if split_idx == 0:
            print("⚠️ データが少なすぎます。Fine-Tuningなしで評価します。")
            X_ft, y_ft = [], []
            X_eval, y_eval = X_seq, y_seq
            do_finetuning = False
        else:
            X_ft = X_seq[:split_idx]
            y_ft = y_seq[:split_idx]
            X_eval = X_seq[split_idx:]
            y_eval = y_seq[split_idx:]
            do_finetuning = True
            print(f"時系列分割完了: Tuning用(前半)={len(X_ft)}, Eval用(後半)={len(X_eval)}")
    else:
        # 比率0なら全て評価に回す
        print("設定: Fine-Tuningなし (Baseline評価)")
        X_ft, y_ft = [], []
        X_eval, y_eval = X_seq, y_seq
        do_finetuning = False

    # ▼▼▼ Fine-Tuning実行 ▼▼▼
    if do_finetuning:
        print("\n🚀 Fine-Tuning開始...")
        # 正しいクラス HeatmapGenerator を使用
        ft_gen = HeatmapGenerator(X_ft, y_ft, 16, MAP_SIZE, X_RANGE, Z_RANGE)
        model.fit(ft_gen, epochs=EPOCHS, verbose=1)
        
        # モデル保存 (設定された動的パスを使用)
        model.save(SAVE_MODEL_PATH)
        print(f"💾 Fine-Tuning済みモデルを保存しました: {SAVE_MODEL_PATH}")
    else:
        print("Fine-Tuningをスキップしました。")

    print("\n📊 最終評価 (評価用データ)...")
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

    if count > 0:
        avg_error = total_error / (count * len(parts))
        print("\n" + "="*50)
        print(f"🎉 最終結果 (ID: {NAME}, Fine-Tuning: {FINE_TUNE_RATIO*100}%)")
        print("="*50)
        print(f"🏆 平均誤差: {avg_error:.2f} mm")
        print("-" * 50)
        for part in parts:
            print(f"  - {part:10s}: {part_errors[part]/count:.2f} mm")
        print("="*50)
    else:
        print("評価データがありませんでした。")

if __name__ == "__main__":
    main()