import pandas as pd

# ファイル読み込み
df = pd.read_csv("inputs/result2.csv")

# 座標カラムを抽出
coord_cols = [c for c in df.columns if "_X" in c or "_Z" in c]
coords = df[coord_cols]

print("=== 座標データの範囲確認 (mm) ===")
print(coords.describe().loc[['min', 'max', 'mean']])

print("\n=== 現在の設定値 (確認) ===")
print("X_RANGE = (-600.0, 600.0)")
print("Z_RANGE = (-1200.0, 1200.0)")
