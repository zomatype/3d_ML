import pandas as pd

# ファイル読み込み
df = pd.read_csv("inputs/result.csv") # result2.csv も確認推奨

# 座標カラムを抽出
coord_cols = [c for c in df.columns if "_X" in c or "_Z" in c]
coords = df[coord_cols]

print("=== 座標データの範囲確認 (mm) ===")
stats = coords.describe().loc[['min', 'max']]
print(stats)

print("\n=== 推奨設定での判定 ===")
# 推奨値
limit_x = (-1200.0, 1200.0)
limit_z = (-600.0, 600.0)

min_vals = stats.loc['min']
max_vals = stats.loc['max']

is_safe = True
# X軸チェック
x_cols = [c for c in coord_cols if "_X" in c]
if min_vals[x_cols].min() < limit_x[0] or max_vals[x_cols].max() > limit_x[1]:
    print(f"⚠️ X軸がはみ出しています！ データ範囲: {min_vals[x_cols].min()} ~ {max_vals[x_cols].max()}")
    is_safe = False

# Z軸チェック
z_cols = [c for c in coord_cols if "_Z" in c]
if min_vals[z_cols].min() < limit_z[0] or max_vals[z_cols].max() > limit_z[1]:
    print(f"⚠️ Z軸がはみ出しています！ データ範囲: {min_vals[z_cols].min()} ~ {max_vals[z_cols].max()}")
    is_safe = False

if is_safe:
    print("✅ データは推奨範囲 (-1200~1200, -600~600) に収まっています。")
else:
    print("❌ 範囲を広げる必要があります。")