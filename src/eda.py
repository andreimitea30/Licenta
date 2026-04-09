import os
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_BASE_PATH = ROOT_DIR / "haa500_v1_1"
METADATA_PATH = DATASET_BASE_PATH / "raw"
VIDEO_PATH = DATASET_BASE_PATH / "video"

ANALYZE_VIDEO_FILES = True


def load_metadata(metadata_path):
    print(f"Scanning {metadata_path} for metadata files (txt)...")

    all_txt_files = glob.glob(str(metadata_path / "*.txt"))

    if not all_txt_files:
        print(f"No TXT files found in {metadata_path}! Check your path.")
        return pd.DataFrame()

    df_list = []
    headers = ['youtube_url', 'start_time', 'end_time', 'is_camera_moving', 'num_of_dominant_figure']

    for filename in tqdm(all_txt_files, desc="Loading Metadata"):
        try:
            temp_df = pd.read_csv(filename, header=None, names=headers)
            class_name = os.path.splitext(os.path.basename(filename))[0]
            temp_df['class_label'] = class_name
            temp_df['clip_duration'] = temp_df['end_time'] - temp_df['start_time']
            df_list.append(temp_df)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not df_list:
        return pd.DataFrame()

    full_df = pd.concat(df_list, ignore_index=True)
    print(f"Metadata loaded. Total samples: {len(full_df)}")
    return full_df


def get_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    duration = frame_count / fps if fps > 0 else 0

    return {
        'vid_width': width,
        'vid_height': height,
        'vid_fps': fps,
        'vid_frame_count': frame_count,
        'vid_actual_duration': duration,
        'aspect_ratio': width / height if height > 0 else 0,
    }


def enrich_with_video_data(df, video_root_path):
    print(f"Scanning {video_root_path} for video files...")

    video_files = glob.glob(str(video_root_path / "**" / "*.mp4"), recursive=True)
    video_files += glob.glob(str(video_root_path / "**" / "*.avi"), recursive=True)

    vid_map = {os.path.basename(v): v for v in video_files}

    print(f"Found {len(video_files)} video files.")
    print("Analyzing video files (Resolution, FPS, Integrity)... this may take time.")

    df['class_video_idx'] = df.groupby('class_label').cumcount()
    results = []

    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Inspecting Videos"):
        class_name = row['class_label']
        video_idx = row['class_video_idx']

        expected_name = f"{class_name}_{video_idx:03d}.mp4"
        expected_name_avi = f"{class_name}_{video_idx:03d}.avi"

        found_path = vid_map.get(expected_name) or vid_map.get(expected_name_avi)

        if found_path:
            props = get_video_properties(found_path)
            if props:
                props['index'] = idx
                results.append(props)

    df.drop(columns=['class_video_idx'], inplace=True)

    video_df = pd.DataFrame(results)
    if not video_df.empty:
        video_df.set_index('index', inplace=True)
        return df.join(video_df)

    print("Could not match videos to CSV entries. Returning metadata only.")
    return df


def classify_resolution(row):
    if pd.isna(row['vid_width']):
        return "Unknown"
    w, h = row['vid_width'], row['vid_height']
    orientation = "Vertical" if h > w else "Landscape"
    max_dim = max(w, h)
    if max_dim >= 3800:
        res_name = "4K"
    elif max_dim >= 1900:
        res_name = "FHD (1080p)"
    elif max_dim >= 1200:
        res_name = "HD (720p)"
    elif max_dim >= 600:
        res_name = "SD (480p)"
    else:
        res_name = "Low Res"
    return f"{res_name} {orientation}"


def plot_eda(df):
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle("HAA500 Exploratory Data Analysis", fontsize=20)

    ax1 = plt.subplot(3, 2, 1)
    class_counts = df['class_label'].value_counts()
    size_distribution = class_counts.value_counts().sort_index()
    sns.barplot(x=size_distribution.index, y=size_distribution.values, ax=ax1, palette="viridis")
    ax1.set_title("Distribution of Class Sizes")
    ax1.set_xlabel("Number of Samples (Videos)")
    ax1.set_ylabel("Count of Classes")

    ax2 = plt.subplot(3, 2, 2)
    if 'is_camera_moving' in df.columns:
        counts = df['is_camera_moving'].value_counts()
        ax2.pie(counts, labels=counts.index, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
        ax2.set_title("Camera Motion Distribution")

    ax3 = plt.subplot(3, 2, 3)
    if 'num_of_dominant_figure' in df.columns:
        sns.histplot(data=df, x='num_of_dominant_figure', bins=10, ax=ax3, kde=False, color="orange")
        ax3.set_title("Number of Dominant Figures")

    ax4 = plt.subplot(3, 2, 4)
    if 'clip_duration' in df.columns:
        sns.histplot(data=df, x='clip_duration', bins=30, ax=ax4, kde=True, color="green")
        ax4.set_title("Clip Duration Distribution (from CSV timestamps)")
        ax4.set_xlabel("Seconds")

    ax5 = plt.subplot(3, 2, 5)
    if 'vid_width' in df.columns:
        df['res_category'] = df.apply(classify_resolution, axis=1)
        res_counts = df['res_category'].value_counts()
        sns.barplot(x=res_counts.values, y=res_counts.index, ax=ax5, palette="rocket")
        ax5.set_title("Video Resolution & Orientation")
    else:
        ax5.text(0.5, 0.5, "Video Analysis Skipped", ha='center')

    ax6 = plt.subplot(3, 2, 6)
    if 'vid_fps' in df.columns:
        sns.histplot(data=df, x='vid_fps', ax=ax6)
        ax6.set_title("Frame Rate Distribution")
        ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45)
    else:
        ax6.text(0.5, 0.5, "Video Signal Analysis Skipped", ha='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path = ROOT_DIR / "haa500_eda_report.png"
    plt.savefig(output_path)
    print(f"EDA plot saved to '{output_path}'")
    plt.show()


if __name__ == "__main__":
    if not DATASET_BASE_PATH.exists():
        print(f"Error: Base path '{DATASET_BASE_PATH}' does not exist.")
        exit(1)

    df = load_metadata(METADATA_PATH)

    if not df.empty:
        if ANALYZE_VIDEO_FILES:
            df = enrich_with_video_data(df, VIDEO_PATH)

        print("\n--- DATASET SUMMARY ---")
        print(df.info())
        print("\n--- NUMERICAL STATS ---")
        print(df.describe())

        plot_eda(df)

        output_csv = ROOT_DIR / "HAA500_consolidated_metadata.csv"
        df.to_csv(output_csv, index=False)
        print(f"Consolidated metadata saved to '{output_csv}'")
    else:
        print("No data loaded.")
