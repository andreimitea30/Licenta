import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT_DIR / "splits"
OUTPUT_FEATURES_DIR = ROOT_DIR / "extracted_skeletons"
MODEL_PATH = ROOT_DIR / "pose_landmarker_full.task"

def ensure_model():
    if not MODEL_PATH.exists():
        print("Downloading MediaPipe Pose Landmarker model...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
            MODEL_PATH,
        )

def make_landmarker_options():
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    return vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

def extract_skeleton_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0

    frames_data = []
    frame_idx = 0
    last_timestamp_ms = -1

    with vision.PoseLandmarker.create_from_options(make_landmarker_options()) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

            timestamp_ms = int((frame_idx / fps) * 1000)
            if timestamp_ms <= last_timestamp_ms:
                timestamp_ms = last_timestamp_ms + 1
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            frame_landmarks = np.zeros((33, 3), dtype=np.float32)
            if result.pose_landmarks:
                for idx, landmark in enumerate(result.pose_landmarks[0]):
                    frame_landmarks[idx] = [landmark.x, landmark.y, landmark.visibility]

            frames_data.append(frame_landmarks)
            frame_idx += 1

    cap.release()

    if len(frames_data) == 0:
        return None

    return np.array(frames_data)

def process_split(split_name):
    csv_path = SPLITS_DIR / f"{split_name}_split.csv"
    if not csv_path.exists():
        print(f"Could not find {csv_path}")
        return

    df = pd.read_csv(csv_path)
    output_split_dir = OUTPUT_FEATURES_DIR / split_name
    output_split_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {split_name} split ({len(df)} videos)...")

    success_count = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_path = row['video_path']
        class_name = row['class_label']

        base_name = Path(video_path).stem
        class_dir = output_split_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        save_path = class_dir / f"{base_name}.npy"
        if save_path.exists():
            success_count += 1
            continue

        try:
            skeleton_data = extract_skeleton_from_video(video_path)
            if skeleton_data is not None:
                np.save(save_path, skeleton_data)
                success_count += 1
        except Exception as e:
            print(f"Error processing {video_path}: {e}")

    print(f"Finished {split_name}: {success_count}/{len(df)} successfully extracted.")

if __name__ == "__main__":
    ensure_model()
    process_split("train")
    process_split("val")
    process_split("test")
