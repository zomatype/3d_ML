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
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# --- ディレクトリ作成 ---
os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ▼▼▼ ユーザー設定値 ▼▼▼
# Z軸（短辺）: データ最大値 846.5mm をカバーするため、余裕を持って ±1000mm に設定
Z_RANGE = (-1000.0, 1000.0)  # 幅 2000mm

# X軸（長辺）: Z軸の幅(2000mm) の「2倍」の幅を確保して、比率2:1を維持
X_RANGE = (-2000.0, 2000.0)  # 幅 4000mm (-2000 ~ 2000)
MAP_SIZE = (64, 32)
# ▲▲▲ ユーザー設定値 ▲▲▲

# --- ジェネレータクラス (メモリ節約の要) ---
class HeatmapGenerator(Sequence):
    """
    学習時にリアルタイムでヒートマップを生成するクラス。
    これにより、全データをメモリに展開するのを防ぎます。
    """

    def __init__(self, x_set, z_coords, batch_size, map_size, x_range, z_range, sigma=1.5):
        self.x = x_set
        self.z = z_coords  # ここには座標データ(N, 12)が入る
        self.batch_size = batch_size
        self.map_size = map_size
        self.h, self.w = map_size
        self.x_range = x_range
        self.z_range = z_range
        self.sigma = sigma
        self.indices = np.arange(len(self.x))

        # ガウス分布計算用のグリッドを事前作成
        x_grid = np.arange(0, self.w, 1, np.float32)
        z_grid = np.arange(0, self.h, 1, np.float32)
        self.X_grid, self.Z_grid = np.meshgrid(x_grid, z_grid)

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, idx):
        # バッチ分のインデックスを取得
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = self.x[inds]
        batch_z_coords = self.z[inds]

        # ヒートマップ生成処理
        batch_heatmaps = self._generate_heatmaps(batch_z_coords)

        return batch_x, batch_heatmaps

    def _generate_heatmaps(self, coords):
        batch_size = len(coords)
        num_keypoints = coords.shape[1] // 2
        heatmaps = np.zeros((batch_size, self.h, self.w,
                            num_keypoints), dtype=np.float32)

        min_x, max_x = self.x_range
        min_z, max_z = self.z_range

        for i in range(batch_size):
            # 座標を取り出す (x1, x2... z1, z2...) の順になっていると想定してreshape
            # ここでは (head_x, head_z, heart_x, heart_z...) の順に並んでいる前提で処理
            # カラム順序: Head_X, Head_Z, Heart_X, Heart_Z ...

            flat_coords = coords[i]

            for k in range(num_keypoints):
                real_x = flat_coords[k*2]     # _X
                real_z = flat_coords[k*2 + 1]  # _Z

                # 正規化
                norm_x = (real_x - min_x) / (max_x - min_x)
                norm_z = (real_z - min_z) / (max_z - min_z)

                # ピクセル座標
                center_x = norm_x * (self.w - 1)
                center_z = norm_z * (self.h - 1)

                # ガウス分布作成 (高速化のためメッシュグリッド計算)
                d2 = (self.X_grid - center_x)**2 + (self.Z_grid - center_z)**2
                g = np.exp(-d2 / (2 * self.sigma**2))

                heatmaps[i, :, :, k] = g

        return heatmaps

    def on_epoch_end(self):
        # エポックごとにデータをシャッフル
        np.random.shuffle(self.indices)


# --- データ読み込み ---
input_dir = "inputs"
try:
    input_df = pd.read_csv(os.path.join(input_dir, "input.csv"))
    output_df = pd.read_csv(os.path.join(input_dir, "result.csv"))
except FileNotFoundError:
    print("データファイルが見つかりません")
    exit(1)

# RSSIデータ (X)
rssi_columns = [col for col in input_df.columns if "rssi" in col]
rssi_columns = [col for col in input_df.columns if "rssi" in col]
X_all_raw = input_df[rssi_columns].values.astype(np.float32)

# 1. 平均と標準偏差を計算 (各タグごとに計算 = axis=0)
train_mean = np.mean(X_all_raw, axis=0)
train_std = np.std(X_all_raw, axis=0)

# ゼロ除算防止のため、stdが0の場合は1にするなどの安全策
train_std = np.where(train_std < 1e-6, 1.0, train_std)

# 2. 正規化実行 (Z-score: (x - mean) / std)
X_all_raw = (X_all_raw - train_mean) / train_std

print("★ 正規化完了: 平均0, 分散1 に変換しました。")
print(f"  Mean[0]: {train_mean[0]:.2f}, Std[0]: {train_std[0]:.2f}")

# 3. 統計量を保存 (推論時に使うため)
np.save(os.path.join("models", "train_mean.npy"), train_mean)
np.save(os.path.join("models", "train_std.npy"), train_std)

# 座標データ (z) - ヒートマップに変換せず、座標のまま持つ
# カラム順序を固定: Head_X, Head_Z, Heart_X, Heart_Z ...
target_columns = []
for part in ["Head", "Heart", "Rshoulder", "Lshoulder", "Rhip", "Lhip"]:
    target_columns.extend([f"{part}_X", f"{part}_Z"])

z_all_coords = output_df[target_columns].values.astype(np.float32)

print(f"データ読み込み完了: Samples={len(X_all_raw)}")


# --- 時系列データの作成 (スライディングウィンドウ) ---
def create_sequences_coords(X, z, time_steps):
    X_seq, z_seq = [], []
    # zは "最後の時刻" の座標を使う
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:(i + time_steps)])
        z_seq.append(z[i + time_steps])
    return np.array(X_seq), np.array(z_seq)


# --- モデル構築関数 ---
def build_model_heatmap(input_shape, output_shape=(64, 32, 6), use_pooling=False):
    model = Sequential()
    # Encoder
    model.add(Conv1D(64, 3, activation='relu',
              padding='same', input_shape=input_shape))
    if use_pooling:
        model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    if use_pooling:
        model.add(MaxPooling1D(2))
    model.add(Dropout(0.2))

    model.add(LSTM(128))

    # Decoder
    model.add(Dense(4 * 2 * 256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Reshape((4, 2, 256)))

    model.add(Conv2DTranspose(128, 3, strides=2,
              padding='same', activation='relu'))  # 8x4
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(128, 3, strides=2,
              padding='same', activation='relu'))  # 16x8
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, 3, strides=2,
              padding='same', activation='relu'))  # 32x16
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(32, 3, strides=2,
              padding='same', activation='relu'))  # 64x32
    model.add(BatchNormalization())

    model.add(Conv2D(output_shape[2], 1, activation='sigmoid', padding='same'))

    model.compile(optimizer='adam', loss='mse')
    return model


# --- 学習処理 ---
def train_model_generator(X_seq, z_coords, time_steps, use_pooling, run_label=""):
    # Split
    X_train, X_test, z_train, z_test = train_test_split(
        X_seq, z_coords, test_size=0.2, random_state=42
    )

    # Generator作成
    batch_size = 16  # メモリがきつい場合は減らす (8など)
    train_gen = HeatmapGenerator(
        X_train, z_train, batch_size, MAP_SIZE, X_RANGE, Z_RANGE)
    test_gen = HeatmapGenerator(
        X_test, z_test, batch_size, MAP_SIZE, X_RANGE, Z_RANGE)

    model = build_model_heatmap(
        (time_steps, X_seq.shape[2]),
        (MAP_SIZE[0], MAP_SIZE[1], 6),
        use_pooling=use_pooling
    )

    print(f"\n--- Training {run_label} ---")

    # fitにはgeneratorを渡す
    history = model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=30,  # 時間短縮のため少し減らしました
        verbose=1,
        # GPUがない場合、multiprocessingを使うと逆に遅くなることがあるのでworkers=1推奨
    )

    return model, history


def plot_history(history, title):
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    file_path = os.path.join("results", f"{title}.png")
    plt.savefig(file_path)
    plt.close()


# --- メインループ ---
# テストのため設定を少し絞る
for is_pooling, label in [(False, "Without_Pooling"), (True, "With_Pooling")]:
    print(f"\n{'='*25} {label} {'='*25}")
    for ts in [20]:  # まずは系列長20だけで試すのをお勧めします（10,20,30全てやると時間がかかります）
        print(f"\n--- TimeSteps {ts} ---")
        X_seq, z_seq = create_sequences_coords(X_all_raw, z_all_coords, ts)

        if is_pooling and ts < 8:
            continue

        run_label = f"seq{ts}_{label}"
        model, history = train_model_generator(
            X_seq, z_seq, ts, is_pooling, run_label)
        plot_history(history, f"History_{run_label}")

        # モデル保存
        model.save(os.path.join("models", f"{run_label}.keras"))

print("学習完了")
