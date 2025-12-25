import pandas as pd
import matplotlib.pyplot as plt
import os

# ▼▼▼ 設定 ▼▼▼
TARGET_FILE = "3d_sokuzai_1.csv"
FRAMES_TO_PLOT = [0, 1, 10, 1000]  # 確認したいフレーム番号

# ベッド枠 (メートル単位: 幅0.9m, 長さ1.98m)
BED_W = 0.90
BED_L = 1.98
# 原点中心の矩形座標
BED_X = [-BED_L/2, BED_L/2, BED_L/2, -BED_L/2, -BED_L/2]
BED_Z = [-BED_W/2, -BED_W/2, BED_W/2, BED_W/2, -BED_W/2]

# 表示範囲 (メートル)
VIEW_X = (-1.5, 1.5)
VIEW_Z = (-1.0, 1.0)
# ▲▲▲▲▲▲▲▲▲▲▲▲▲

def main():
    if not os.path.exists(TARGET_FILE):
        print(f"❌ ファイルが見つかりません: {TARGET_FILE}")
        return

    print(f"読み込み中: {TARGET_FILE}")
    df = pd.read_csv(TARGET_FILE)
    df.columns = df.columns.str.strip()

    # パーツ定義
    parts_map = {
        "Head": ("Head_Rel_X", "Head_Rel_Z"),
        "Chest": ("Chest_Rel_X", "Chest_Rel_Z"),
        "R_Sho": ("RightShoulder_Rel_X", "RightShoulder_Rel_Z"),
        "L_Sho": ("LeftShoulder_Rel_X", "LeftShoulder_Rel_Z"),
        "R_Leg": ("RightUpperLeg_Rel_X", "RightUpperLeg_Rel_Z"),
        "L_Leg": ("LeftUpperLeg_Rel_X", "LeftUpperLeg_Rel_Z"),
    }
    
    bones = [
        ("Head", "Chest"), ("Chest", "R_Sho"), ("Chest", "L_Sho"),
        ("Chest", "R_Leg"), ("Chest", "L_Leg"), ("R_Sho", "L_Sho"),
        ("R_Leg", "L_Leg")
    ]

    for frame_idx in FRAMES_TO_PLOT:
        if frame_idx >= len(df):
            print(f"⚠️ フレーム {frame_idx} はデータ範囲外です (全 {len(df)} 行)")
            continue

        plt.figure(figsize=(6, 8))
        
        # 1. ベッド枠
        plt.plot(BED_X, BED_Z, color='cyan', linewidth=2, label='Bed Frame')

        # 2. 骨格プロット
        row = df.iloc[frame_idx]
        points = {}
        x_vals, z_vals = [], []

        for name, (cx, cz) in parts_map.items():
            if cx in df.columns and cz in df.columns:
                x, z = row[cx], row[cz]
                points[name] = (x, z)
                x_vals.append(x)
                z_vals.append(z)
        
        # ボーン
        for p1, p2 in bones:
            if p1 in points and p2 in points:
                plt.plot([points[p1][0], points[p2][0]], 
                         [points[p1][1], points[p2][1]], color='red', linewidth=2)
        
        # 点
        plt.scatter(x_vals, z_vals, c='red', s=80, edgecolors='black', zorder=5)

        # 座標値のテキスト表示 (デバッグ用)
        if "Chest" in points:
            cx, cz = points["Chest"]
            plt.text(cx, cz + 0.1, f"Chest\n({cx:.2f}, {cz:.2f})", 
                     fontsize=9, color='blue', ha='center')

        plt.title(f"Frame {frame_idx} (Time: {row.get('Timestamp', 'N/A')})")
        plt.xlabel("X [m] (Length)")
        plt.ylabel("Z [m] (Width)")
        
        plt.xlim(VIEW_X)
        plt.ylim(VIEW_Z)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        filename = f"frame_{frame_idx}.png"
        plt.savefig(filename, dpi=100)
        plt.close()
        print(f"✅ 保存完了: {filename}")

if __name__ == "__main__":
    main()