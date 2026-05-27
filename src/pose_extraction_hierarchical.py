from concurrent.futures import ProcessPoolExecutor, as_completed
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
OUTPUT_FEATURES_DIR = ROOT_DIR / "extracted_skeletons_hierarchical"
POSE_MODEL          = ROOT_DIR / "pose_landmarker_full.task"
HAND_MODEL          = ROOT_DIR / "hand_landmarker.task"
FACE_MODEL          = ROOT_DIR / "face_landmarker.task"

NUM_BODY  = 33
NUM_HAND  = 21

FACE_INDICES = (
    10, 152, 234, 454, 132, 58, 288, 361,
    70, 107, 336, 300,
    33, 133, 159, 145,
    362, 263, 386, 374,
    1, 4,
    61, 291, 13, 14,
    78, 308, 87, 317,
)
NUM_FACE  = len(FACE_INDICES)
NUM_NODES = NUM_BODY + 2 * NUM_HAND + NUM_FACE

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

def _make_face_options():
    return vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL),
                                        delegate=python.BaseOptions.Delegate.CPU),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

def _resolve_video_path(raw_path: str) -> str:
    p = raw_path.replace("\\", "/")
    if "haa500_v1_1/" in p:
        suffix = p.split("haa500_v1_1/", 1)[1]
        return str(ROOT_DIR / "haa500_v1_1" / suffix)
    return raw_path

def _assign_hands(hand_res):
    best = {"Left": None, "Right": None}
    if not hand_res.hand_landmarks:
        return best
    for hand_lms, handedness in zip(hand_res.hand_landmarks, hand_res.handedness):
        cat   = handedness[0]
        label = cat.category_name
        score = float(cat.score)
        if label not in best:
            continue
        if best[label] is None or score > best[label][0]:
            best[label] = (score, hand_lms)
    return best

def extract_hierarchical(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0

    pose_lm = vision.PoseLandmarker.create_from_options(_make_pose_options())
    hand_lm = vision.HandLandmarker.create_from_options(_make_hand_options())
    face_lm = vision.FaceLandmarker.create_from_options(_make_face_options())

    frames     = []
    frame_idx  = 0
    last_ts_ms = -1
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms  = int((frame_idx / fps) * 1000)
            if ts_ms <= last_ts_ms:
                ts_ms = last_ts_ms + 1
            last_ts_ms = ts_ms

            pose_res = pose_lm.detect_for_video(mp_img, ts_ms)
            hand_res = hand_lm.detect_for_video(mp_img, ts_ms)
            face_res = face_lm.detect_for_video(mp_img, ts_ms)

            row = np.zeros((NUM_NODES, 4), dtype=np.float32)

            if pose_res.pose_landmarks:
                for i, lm in enumerate(pose_res.pose_landmarks[0]):
                    row[i] = [lm.x, lm.y, lm.z, lm.visibility]

            hands = _assign_hands(hand_res)
            for label, base in (("Left", NUM_BODY),
                                ("Right", NUM_BODY + NUM_HAND)):
                entry = hands.get(label)
                if entry is None:
                    continue
                _, hand_lms = entry
                for i, lm in enumerate(hand_lms):
                    row[base + i] = [lm.x, lm.y, lm.z, 1.0]

            if face_res.face_landmarks:
                face_lms = face_res.face_landmarks[0]
                base = NUM_BODY + 2 * NUM_HAND
                for slot, src_idx in enumerate(FACE_INDICES):
                    if src_idx < len(face_lms):
                        lm = face_lms[src_idx]
                        row[base + slot] = [lm.x, lm.y, lm.z, 1.0]

            frames.append(row)
            frame_idx += 1
    finally:
        pose_lm.close()
        hand_lm.close()
        face_lm.close()
        cap.release()

    if not frames:
        return None
    return np.stack(frames, axis=0)

def _worker(args):
    video_path, save_path = args
    save_path = Path(save_path)
    if save_path.exists():
        return ("skip", str(save_path), None)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = extract_hierarchical(video_path)
        if data is not None:
            np.save(save_path, data)
            return ("ok", str(save_path), None)
        return ("err", str(save_path), "extractor returned None")
    except Exception as e:
        return ("err", str(save_path), f"{type(e).__name__}: {e}")

def process_split(split_name: str, dry_limit: int = 0, workers: int = 1):
    csv_path = SPLITS_DIR / f"{split_name}_split.csv"
    if not csv_path.exists():
        print(f"  Missing {csv_path}, skipping.")
        return

    df = pd.read_csv(csv_path)
    if dry_limit:
        df = df.head(dry_limit)
    out_dir = OUTPUT_FEATURES_DIR / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for _, row in df.iterrows():
        class_name = row["class_label"]
        video_path = _resolve_video_path(row["video_path"])
        stem       = Path(video_path).stem
        save_path  = out_dir / class_name / f"{stem}.npy"
        tasks.append((video_path, str(save_path)))

    ok = skip = err = 0

    if workers <= 1:
        for t in tqdm(tasks, desc=split_name):
            status, sp, info = _worker(t)
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            else:
                err += 1
                tqdm.write(f"  ERROR {sp}: {info}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_worker, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"{split_name} (x{workers})"):
                status, sp, info = fut.result()
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skip += 1
                else:
                    err += 1
                    tqdm.write(f"  ERROR {sp}: {info}")

    print(f"{split_name}: saved={ok}  skipped={skip}  errors={err}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    ap.add_argument("--dry-limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of parallel extraction processes (default: 1, sequential).")
    args = ap.parse_args()

    for tf in (POSE_MODEL, HAND_MODEL, FACE_MODEL):
        assert tf.exists(), f"Missing task file: {tf}"

    splits = ("train", "val", "test") if args.split == "all" else (args.split,)
    for s in splits:
        process_split(s, dry_limit=args.dry_limit, workers=args.workers)
    print("Hierarchical extraction complete.")
