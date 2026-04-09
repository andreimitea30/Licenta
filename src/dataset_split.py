from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
METADATA_CSV = ROOT_DIR / "HAA500_consolidated_metadata.csv"
VIDEO_DIR = ROOT_DIR / "haa500_v1_1" / "video"
OUTPUT_DIR = ROOT_DIR / "splits"


def main():
    print("Loading consolidated metadata...")
    df = pd.read_csv(METADATA_CSV)

    df['class_video_idx'] = df.groupby('class_label').cumcount()

    video_paths = []
    valid_indices = []

    print("Verifying physical video files...")
    for idx, row in df.iterrows():
        class_name = row['class_label']
        vid_num = row['class_video_idx']

        mp4_path = VIDEO_DIR / class_name / f"{class_name}_{vid_num:03d}.mp4"
        avi_path = VIDEO_DIR / class_name / f"{class_name}_{vid_num:03d}.avi"

        if mp4_path.exists():
            video_paths.append(str(mp4_path))
            valid_indices.append(idx)
        elif avi_path.exists():
            video_paths.append(str(avi_path))
            valid_indices.append(idx)

    df_valid = df.loc[valid_indices].copy()
    df_valid['video_path'] = video_paths

    print(f"Found {len(df_valid)} valid videos out of {len(df)} metadata entries.")

    X = df_valid['video_path']
    y = df_valid['class_label']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    train_df = df_valid.loc[X_train.index]
    val_df = df_valid.loc[X_val.index]
    test_df = df_valid.loc[X_test.index]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(OUTPUT_DIR / "train_split.csv", index=False)
    val_df.to_csv(OUTPUT_DIR / "val_split.csv", index=False)
    test_df.to_csv(OUTPUT_DIR / "test_split.csv", index=False)

    print(f"\nSplits created successfully in {OUTPUT_DIR}:")
    print(f"Train: {len(train_df)} videos")
    print(f"Validation: {len(val_df)} videos")
    print(f"Test: {len(test_df)} videos")


if __name__ == "__main__":
    main()
