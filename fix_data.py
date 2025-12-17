import pandas as pd
import os


def fix_csv(filepath):
    if not os.path.exists(filepath):
        print(f"スキップ: {filepath} が見つかりません。")
        return

    print(f"修正中: {filepath} ...")
    df = pd.read_csv(filepath)

    # NaNがあるか確認
    if df.isnull().sum().sum() > 0:
        # NaNを -120.0 で埋める
        df = df.fillna(-120.0)
        df.to_csv(filepath, index=False)
        print("  ✅ NaNを -120.0 に置換して保存しました。")
    else:
        print("  ✅ NaNはありませんでした。修正不要です。")


# 学習用とテスト用の両方を修正
fix_csv("inputs/input.csv")
fix_csv("inputs/input2.csv")  # もしあれば
