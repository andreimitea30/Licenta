"""Extract body + hand landmarks via the modern MediaPipe Tasks API.

Per-frame output: (75, 4) = [x, y, z, visibility]
  Indices  0..32  : 33 body landmarks (PoseLandmarker, image-space)
  Indices 33..53  : 21 LEFT hand landmarks (HandLandmarker, image-space)
  Indices 54..74  : 21 RIGHT hand landmarks (HandLandmarker, image-space)

All x, y are in [0, 1] image-normalized. z is depth (different reference for
body vs hands, but consistent within each part). Visibility is the real
PoseLandmarker score for body joints; for hands we set 1.0 if a landmark was
detected and 0.0 (with all-zero coords) if the hand was missing in that frame.

Per-video output saved as a (T, 75, 4) float32 numpy array under
extracted_skeletons_holistic/{train,val,test}/<class>/<video>.npy
"""
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
OUTPUT_FEATURES_DIR = ROOT_DIR / "extracted_skeletons_holistic"
POSE_MODEL          = ROOT_DIR / "pose_landmarker_full.task"
HAND_MODEL          = ROOT_DIR / "hand_landmarker.task"

NUM_BODY  = 33
NUM_HAND  = 21
NUM_NODES = NUM_BODY + 2 * NUM_HAND   # 75


def _make_pose_options():
    return vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(POSE_MODEL),
                                        delegate=python.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _make_hand_options():
    return vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL),
                                        delegate=python.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )


def _resolve_video_path(raw_path: str) -> str:
    """Splits CSVs were generated under WSL with Linux absolute paths.
    Translate the suffix after 'haa500_v1_1/' to the local Windows ROOT_DIR."""
    p = raw_path.replace("\\", "/")
    if "haa500_v1_1/" in p:
        suffix = p.split("haa500_v1_1/", 1)[1]
        return str(ROOT_DIR / "haa500_v1_1" / suffix)
    return raw_path


def extract_holistic(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    pose_lm = vision.PoseLandmarker.create_from_options(_make_pose_options())
    hand_lm = vision.HandLandmarker.create_from_options(_make_hand_options())

    frames = []
    frame_idx  = 0
    last_ts_ms = -1
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((frame_idx / fps) * 1000)
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms

            pose_res = pose_lm.detect_for_video(mp_img, ts_ms)
            hand_res = hand_lm.detect_for_video(mp_img, ts_ms)

            row = np.zeros((NUM_NODES, 4), dtype=np.float32)

            # Body
            if pose_res.pose_landmarks:
                for i, lm in enumerate(pose_res.pose_landmarks[0]):
                    row[i] = [lm.x, lm.y, lm.z, lm.visibility]

            # Hands — assign by handedness category. MediaPipe's "Left"/"Right"
            # refers to the person's actual hand (camera-facing user convention).
            if hand_res.hand_landmarks:
                for hand_lms, handedness in zip(hand_res.hand_landmarks,
                                                hand_res.handedness):
                    label = handedness[0].category_name  # "Left" or "Right"
                    base  = NUM_BODY if label == "Left" else NUM_BODY + NUM_HAND
                    for i, lm in enumerate(hand_lms):
                        row[base + i] = [lm.x, lm.y, lm.z, 1.0]

            frames.append(row)
            frame_idx += 1
    finally:
        pose_lm.close()
        hand_lm.close()
        cap.release()

    if not frames:
        return None
    return np.stack(frames, axis=0)


def process_split(split_name: str, dry_limit: int = 0):
    csv_path = SPLITS_DIR / f"{split_name}_split.csv"
    if not csv_path.exists():
        print(f"  Missing {csv_path}, skipping.")
        return

    df = pd.read_csv(csv_path)
    if dry_limit:
        df = df.head(dry_limit)
    out_dir = OUTPUT_FEATURES_DIR / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = skip = err = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
        class_name = row["class_label"]
        video_path = _resolve_video_path(row["video_path"])
        stem       = Path(video_path).stem
        save_path  = out_dir / class_name / f"{stem}.npy"

        if save_path.exists():
            skip += 1
            continue

        save_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = extract_holistic(video_path)
            if data is not None:
                np.save(save_path, data)
                ok += 1
            else:
                err += 1
        except Exception as e:
            tqdm.write(f"  ERROR {video_path}: {e}")
            err += 1

    print(f"{split_name}: saved={ok}  skipped={skip}  errors={err}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    ap.add_argument("--dry-limit", type=int, default=0,
                    help="If >0, process only the first N rows (for timing tests).")
    args = ap.parse_args()

    assert POSE_MODEL.exists(), f"Missing pose task: {POSE_MODEL}"
    assert HAND_MODEL.exists(), f"Missing hand task: {HAND_MODEL}"

    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    for s in splits:
        process_split(s, dry_limit=args.dry_limit)
    print("Holistic extraction complete.")
