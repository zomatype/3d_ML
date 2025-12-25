import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import io
import os

# ▼▼▼ 設定 ▼▼▼
TARGET_CSV = "inputs/result.csv"  # Bさんのデータ

# ベッド枠 (原点中心: 幅900, 長さ1980)
BED_X = (-990, 990)
BED_Z = (-450, 450)

# 表示範囲
DISPLAY_X = (-2000, 2000)
DISPLAY_Z = (-1000, 1000)

START_FRAME = 0
FRAMES_TO_VISUALIZE = 200
SKIP_STEP = 2
# ▲▲▲▲▲▲▲▲▲▲▲▲▲

def main():
    if not os.path.exists(TARGET_CSV):
        print(f"❌ ファイルが見つかりません: {TARGET_CSV}")
        return

    print(f"データ読み込み中: {TARGET_CSV} ...")
    df = pd.read_csv(TARGET_CSV)
    
    parts = ["Head", "Heart", "Rshoulder", "Lshoulder", "Rhip", "Lhip"]
    bones = [
        ("Head", "Heart"), ("Heart", "Rshoulder"), ("Heart", "Lshoulder"),
        ("Heart", "Rhip"), ("Heart", "Lhip"), ("Rshoulder", "Lshoulder"),
        ("Rhip", "Lhip"), ("Rshoulder", "Rhip"), ("Lshoulder", "Lhip")
    ]
    
    frames = []
    print("\nGIF生成中...")

    max_frame = min(len(df), START_FRAME + FRAMES_TO_VISUALIZE)
    
    for i in range(START_FRAME, max_frame, SKIP_STEP):
        # 図のサイズを正方形に近くする (アスペクト比を合わせやすくするため)
        plt.figure(figsize=(10, 6)) 
        
        # 1. ベッド枠描画
        bx = [BED_X[0], BED_X[1], BED_X[1], BED_X[0], BED_X[0]]
        bz = [BED_Z[0], BED_Z[0], BED_Z[1], BED_Z[1], BED_Z[0]]
        plt.plot(bx, bz, color='cyan', linestyle='-', linewidth=2, label='Bed Frame')

        # 2. 骨格描画
        x_points = []
        z_points = []
        
        for part in parts:
            if f"{part}_X" in df.columns:
                px = df.iloc[i][f"{part}_X"]
                pz = df.iloc[i][f"{part}_Z"]
                x_points.append(px)
                z_points.append(pz)
        
        for p1_name, p2_name in bones:
            if f"{p1_name}_X" in df.columns and f"{p2_name}_X" in df.columns:
                x1 = df.iloc[i][f"{p1_name}_X"]
                z1 = df.iloc[i][f"{p1_name}_Z"]
                x2 = df.iloc[i][f"{p2_name}_X"]
                z2 = df.iloc[i][f"{p2_name}_Z"]
                plt.plot([x1, x2], [z1, z2], c='blue', linewidth=3, alpha=0.7)

        plt.scatter(x_points, z_points, c='blue', s=100, edgecolors='white', zorder=5, label='Ground Truth')
        
        plt.title(f"Ground Truth Check: Frame {i}", fontsize=14)
        plt.xlabel("X (Long) [mm]")
        plt.ylabel("Z (Short) [mm]")
        plt.legend(loc='upper right')
        
        plt.xlim(DISPLAY_X[0], DISPLAY_X[1])
        plt.ylim(DISPLAY_Z[0], DISPLAY_Z[1])
        
        # ★★★ 追加: 縦横の縮尺を強制的に合わせる ★★★
        plt.gca().set_aspect('equal', adjustable='box')
        
        plt.grid(True, linestyle='--', alpha=0.5)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=80)
        plt.close()
        buf.seek(0)
        frames.append(Image.open(buf))

    frames[0].save("check_gt_scale_fixed.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
    print(f"✅ 作成完了: check_gt_scale_fixed.gif")

if __name__ == "__main__":
    main()