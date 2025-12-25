import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, LSTM, Dense, Dropout, MaxPooling1D,
    Reshape, Conv2DTranspose, Conv2D, BatchNormalization
)
from tensorflow.keras.utils import Sequence
import os

# --- ディレクトリ作成 ---
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ▼▼▼ 設定 (ここを固定！) ▼▼▼
# X軸: 長辺方向 (4000mm幅)
X_RANGE = (-2000.0, 2000.0)
# Z軸: 短辺方向 (2000mm幅)
Z_RANGE = (-1000.0, 1000.0)

MAP_SIZE = (64, 32)
TIME_STEPS = 20
BATCH_SIZE = 16
EPOCHS = 30
# ▲▲▲▲▲▲▲▲▲▲▲▲▲

class HeatmapGenerator(Sequence):
    def __init__(self, x_set, z_coords, batch_size, map_size, x_range, z_range, sigma=1.5):
        self.x = x_set
        self.z = z_coords
        self.batch_size = batch_size
        self.h, self.w = map_size # (32, 64) -> h=32, w=64 (通常map_sizeは(w, h)の順)
        # ※修正: map_size=(64, 32)の場合、width=64, height=32
        self.w, self.h = map_size 
        self.x_range = x_range
        self.z_range = z_range
        self.sigma = sigma
        self.indices = np.arange(len(self.x))

        # グリッド作成 (w=64, h=32)
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
        # shape: (B, H, W, K) -> (B, 32, 64, 6)
        heatmaps = np.zeros((batch_size, self.h, self.w, num_keypoints), dtype=np.float32)

        min_x, max_x = self.x_range
        min_z, max_z = self.z_range

        for i in range(batch_size):
            flat_coords = coords[i]
            for k in range(num_keypoints):
                real_x = flat_coords[k*2]
                real_z = flat_coords[k*2 + 1]

                # 範囲外チェック（完全無視せず、端に寄せるクリッピングを行うと安全）
                if real_x < min_x: real_x = min_x
                if real_x > max_x: real_x = max_x
                if real_z < min_z: real_z = min_z
                if real_z > max_z: real_z = max_z

                # 正規化 (0.0 ~ 1.0)
                norm_x = (real_x - min_x) / (max_x - min_x)
                norm_z = (real_z - min_z) / (max_z - min_z)

                # ピクセル座標 (Logic 5: Width=X, Height=Z)
                center_x = norm_x * (self.w - 1)
                center_z = norm_z * (self.h - 1)

                d2 = (self.X_grid - center_x)**2 + (self.Z_grid - center_z)**2
                g = np.exp(-d2 / (2 * self.sigma**2))
                heatmaps[i, :, :, k] = g

        return heatmaps

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# --- データ読み込み ---
print("データ読み込み中 (inputs/input.csv)...")
try:
    input_df = pd.read_csv("inputs/input.csv")
    output_df = pd.read_csv("inputs/result.csv")
except FileNotFoundError:
    print("❌ データファイルが見つかりません。convert_data.pyを実行してください。")
    exit(1)

# RSSIデータ (X)
# NaN埋め (-120.0)
input_df = input_df.fillna(-120.0)
rssi_columns = [col for col in input_df.columns if "rssi" in col]
X_all_raw = input_df[rssi_columns].values.astype(np.float32)

# ▼▼▼ 正規化プロセス (平均・標準偏差の計算と保存) ▼▼▼
print("正規化統計量を計算中...")
train_mean = np.mean(X_all_raw, axis=0)
train_std = np.std(X_all_raw, axis=0)
train_std = np.where(train_std < 1e-6, 1.0, train_std) # ゼロ除算防止

# 適用
X_all_raw = (X_all_raw - train_mean) / train_std

# 保存 (ここが重要！)
np.save(os.path.join("models", "train_mean.npy"), train_mean)
np.save(os.path.join("models", "train_std.npy"), train_std)
print(f"✅ 統計量を保存しました: Mean={train_mean[:3]}...")
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

# 座標データ (z)
target_columns = []
for part in ["Head", "Heart", "Rshoulder", "Lshoulder", "Rhip", "Lhip"]:
    target_columns.extend([f"{part}_X", f"{part}_Z"])
z_all_coords = output_df[target_columns].values.astype(np.float32)

# 時系列作成
def create_sequences(X, z, time_steps):
    X_seq, z_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:(i + time_steps)])
        z_seq.append(z[i + time_steps])
    return np.array(X_seq), np.array(z_seq)

X_seq, z_seq = create_sequences(X_all_raw, z_all_coords, TIME_STEPS)
print(f"学習データセット作成完了: {X_seq.shape}")

# モデル構築
def build_model(input_shape):
    model = Sequential()
    # Encoder
    model.add(Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))
    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))
    model.add(LSTM(128))

    # Decoder
    # MapSize (64, 32) -> 最終出力 (32, 64) に合わせるための逆算
    # 最終: (32, 64)
    # Conv2DTransposeで倍々にしていく: 4x8 -> 8x16 -> 16x32 -> 32x64
    
    model.add(Dense(4 * 8 * 256, activation='relu')) # 初期サイズ 4x8 (縦x横)
    model.add(BatchNormalization())
    model.add(Reshape((4, 8, 256))) # (H, W, C)

    model.add(Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')) # -> 8x16
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(128, 3, strides=2, padding='same', activation='relu')) # -> 16x32
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu'))  # -> 32x64
    model.add(BatchNormalization())
    
    # 出力層 (MapSize Height=32, Width=64, Channels=6)
    model.add(Conv2D(6, 1, activation='sigmoid', padding='same')) 

    model.compile(optimizer='adam', loss='mse')
    return model

# 学習実行
X_train, X_test, z_train, z_test = train_test_split(X_seq, z_seq, test_size=0.2, random_state=42)

train_gen = HeatmapGenerator(X_train, z_train, BATCH_SIZE, MAP_SIZE, X_RANGE, Z_RANGE)
test_gen = HeatmapGenerator(X_test, z_test, BATCH_SIZE, MAP_SIZE, X_RANGE, Z_RANGE)

model = build_model((TIME_STEPS, X_seq.shape[2]))
print("\n🚀 学習開始...")
history = model.fit(train_gen, validation_data=test_gen, epochs=EPOCHS, verbose=1)

# 保存
save_path = "models/model_base.keras"
model.save(save_path)
print(f"\n💾 ベースモデル保存完了: {save_path}")
print("次は viz_skelton_overlay_v4.py で可視化を確認してください。")