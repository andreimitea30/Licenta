from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

ROOT_DIR            = Path(__file__).resolve().parent.parent
SPLITS_DIR          = ROOT_DIR / "splits"
OUTPUT_FEATURES_DIR = ROOT_DIR / "extracted_skeletons_world"
MODEL_PATH          = ROOT_DIR / "pose_landmarker_full.task"


def make_landmarker_options():
    base_options = python.BaseOptions(
        model_asset_path=str(MODEL_PATH),
        delegate=python.BaseOptions.Delegate.CPU,
    )
    return vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def extract_skeleton_3d(video_path: str):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    frames_data = []
    frame_idx   = 0
    last_ts_ms  = -1

    with vision.PoseLandmarker.create_from_options(make_landmarker_options()) as lm:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            ts_ms = int((frame_idx / fps) * 1000)
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms

            result = lm.detect_for_video(mp_img, ts_ms)

            row = np.zeros((33, 4), dtype=np.float32)
            if result.pose_world_landmarks:
                for i, lmk in enumerate(result.pose_world_landmarks[0]):
                    row[i] = [lmk.x, lmk.y, lmk.z, lmk.visibility]
            elif result.pose_landmarks:
                for i, lmk in enumerate(result.pose_landmarks[0]):
                    row[i] = [lmk.x, lmk.y, 0.0, lmk.visibility]

            frames_data.append(row)
            frame_idx += 1

    cap.release()
    return np.array(frames_data) if frames_data else None


def process_split(split_name: str):
    csv_path = SPLITS_DIR / f"{split_name}_split.csv"
    if not csv_path.exists():
        print(f"  Missing {csv_path}, skipping.")
        return

    df      = pd.read_csv(csv_path)
    out_dir = OUTPUT_FEATURES_DIR / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = skip = err = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
        class_name = row["class_label"]
        stem       = Path(row["video_path"]).stem
        save_path  = out_dir / class_name / f"{stem}.npy"

        if save_path.exists():
            skip += 1
            continue

        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = extract_skeleton_3d(row["video_path"])
            if data is not None:
                np.save(save_path, data)
                ok += 1
            else:
                err += 1
        except Exception as e:
            tqdm.write(f"  ERROR {row['video_path']}: {e}")
            err += 1

    print(f"{split_name}: saved={ok}  skipped={skip}  errors={err}")


if __name__ == "__main__":
    assert MODEL_PATH.exists(), f"Task file not found: {MODEL_PATH}"
    for split in ("train", "val", "test"):
        process_split(split)
    print("3D extraction complete.")
