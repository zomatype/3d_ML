import pandas as pd
import numpy as np
import os
import glob

# ================= 設定 =================
# 被験者名(ファイル名に使用)
NAME = "okabe"  # 例: "soma" 

# 処理対象のフォルダ
TARGET_PERSON_DIR = f"data_raw/{NAME}"
OUTPUT_DIR = "inputs"
OUTPUT_SUFFIX = f"_{NAME}"   # 学習用なら "", テスト用なら "2" など変更してください
# リサンプリング間隔
RESAMPLE_RULE = '50ms'

# RSSIがない場所を埋める値 (fix_data.pyの機能)
RSSI_FILL_VALUE = -120.0

# カラム名の対応表
COLUMN_MAPPING = {
    'Head_Rel': 'Head',
    'Chest_Rel': 'Heart',
    'RightUpperArm_Rel': 'Rshoulder',
    'LeftUpperArm_Rel': 'Lshoulder',
    'RightUpperLeg_Rel': 'Rhip',
    'LeftUpperLeg_Rel': 'Lhip'
}

# ★重要★ タグの物理配置順序 (左上から順)
EPC_ORDER = [
    'E280116060000204AC6AD1FD', # Tag0
    'E280116060000204AC6AD0E6', # Tag1
    'E280116060000204AC6AD0EC', # Tag2
    'E280116060000204AC6AD1FA', # Tag3
    'E280116060000204AC6AC8F0', # Tag4
    'E280116060000204AC6AD0E9', # Tag5
    'E280116060000204AC6AD1FE', # Tag6
    'E280116060000204AC6AD1F9', # Tag7
    'E280116060000204AC6AD1FC', # Tag8
    'E280116060000204AC6AD1FB'  # Tag9
]
# ========================================


def load_rssi_fixed(filepath):
    try:
        df = pd.read_csv(filepath)
        # カラム名の空白除去
        df.columns = df.columns.str.strip()

        df['timestamp_str'] = df['date'].astype(str) + ' ' + df['time']
        df['timestamp'] = pd.to_datetime(df['timestamp_str'])

        pivot_df = df.pivot_table(
            index='timestamp',
            columns='EPC',
            values='RSSI',
            aggfunc='mean'
        )

        # ★修正ポイント1: 指定した順序で並べ替え＆不足タグをNaN列として作成
        pivot_df = pivot_df.reindex(columns=EPC_ORDER)

        # Tag0_rssi, Tag1_rssi... にリネーム
        new_columns = {}
        for idx, epc in enumerate(EPC_ORDER):
            new_columns[epc] = f"Tag{idx}_rssi"
        
        pivot_df = pivot_df.rename(columns=new_columns)

        start_time = pivot_df.index[0]
        pivot_df.index = pivot_df.index - start_time

        return pivot_df
    except Exception as e:
        print(f"  [Error] RSSI load failed {filepath}: {e}")
        return None


def load_pose_fixed(filepath):
    try:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()

        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.set_index('Timestamp').sort_index()

        rename_dict = {}
        for src, dst in COLUMN_MAPPING.items():
            rename_dict[f"{src}_X"] = f"{dst}_X"
            rename_dict[f"{src}_Z"] = f"{dst}_Z"

        available_cols = [c for c in rename_dict.keys() if c in df.columns]

        if not available_cols:
            print(f"  [Warning] No matching columns found in {os.path.basename(filepath)}")
            return None

        df = df[available_cols].rename(columns=rename_dict)
        df = df * 1000.0  # m -> mm

        start_time = df.index[0]
        df.index = df.index - start_time

        return df
    except Exception as e:
        print(f"  [Error] Pose load failed {filepath}: {e}")
        return None


def process_pair(rssi_path, pose_path):
    print(f"Processing: {os.path.basename(rssi_path)} ... ", end="")
    df_rssi = load_rssi_fixed(rssi_path)
    df_pose = load_pose_fixed(pose_path)

    if df_rssi is None or df_pose is None:
        print("Failed to load.")
        return None

    # 時刻合わせ
    max_duration = min(df_rssi.index[-1], df_pose.index[-1])
    common_index = pd.timedelta_range(start='0ms', end=max_duration, freq=RESAMPLE_RULE)

    # ★修正ポイント2: ここでfillna(RSSI_FILL_VALUE)を実行
    # これにより、reindexで発生したNaNや、補間できなかった部分が -120.0 になります
    df_rssi_resampled = df_rssi.reindex(common_index).interpolate(method='time').fillna(RSSI_FILL_VALUE)
    
    # Poseデータは線形補間 (端っこはffill/bfillで埋める)
    df_pose_resampled = df_pose.reindex(common_index).interpolate(method='time').ffill().bfill()

    merged = pd.concat([df_rssi_resampled, df_pose_resampled], axis=1)
    
    # 念のため、Poseデータがない行（もしあれば）だけ落とす
    # RSSIはすべて埋まっているので、Pose側の欠損のみチェック
    final_data = merged.dropna()

    print(f"Done. Shape: {final_data.shape}")
    return final_data


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    search_pattern = os.path.join(TARGET_PERSON_DIR, "rssi*.csv")
    rssi_files = sorted(glob.glob(search_pattern))

    print(f"Target: {TARGET_PERSON_DIR} ({len(rssi_files)} files)")
    print(f"Defined EPC Order (Top to Bottom): {len(EPC_ORDER)} tags")

    all_data_list = []

    for rssi_path in rssi_files:
        filename = os.path.basename(rssi_path)
        pose_filename = filename.replace("rssi", "3d") # もし命名規則が違う場合は適宜修正
        pose_path = os.path.join(TARGET_PERSON_DIR, pose_filename)

        if not os.path.exists(pose_path):
            continue

        merged_pair = process_pair(rssi_path, pose_path)
        if merged_pair is not None:
            all_data_list.append(merged_pair)

    if not all_data_list:
        print("❌ Error: No valid data pairs found.")
        return

    final_df = pd.concat(all_data_list, ignore_index=True)

    # 最終チェック: もし万が一NaNが残っていたら再度埋める (Safety Net)
    if final_df.isnull().sum().sum() > 0:
        print("⚠️ Warning: NaNs found in final buffer. Filling with default values.")
        rssi_cols = [c for c in final_df.columns if "rssi" in c]
        final_df[rssi_cols] = final_df[rssi_cols].fillna(RSSI_FILL_VALUE)
        final_df = final_df.fillna(method='ffill').fillna(method='bfill')

    rssi_cols = [c for c in final_df.columns if "rssi" in c]
    pose_cols = [c for c in final_df.columns if c not in rssi_cols]

    input_df = final_df[rssi_cols]
    result_df = final_df[pose_cols]

    input_path = os.path.join(OUTPUT_DIR, f"input{OUTPUT_SUFFIX}.csv")
    result_path = os.path.join(OUTPUT_DIR, f"result{OUTPUT_SUFFIX}.csv")

    input_df.to_csv(input_path, index=False)
    result_df.to_csv(result_path, index=False)

    print(f"\n✅ Success! Saved to:\n  - {input_path}\n  - {result_path}")
    print(f"  Input Shape:  {input_df.shape}")
    print(f"  Result Shape: {result_df.shape}")


if __name__ == "__main__":
    main()