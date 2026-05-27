import argparse
import random
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT_DIR    = Path(__file__).resolve().parent
SPLITS_DIR  = ROOT_DIR / "splits"
OUT_DIR     = ROOT_DIR / "visualization"
POSE_MODEL  = ROOT_DIR / "pose_landmarker_full.task"
HAND_MODEL  = ROOT_DIR / "hand_landmarker.task"
FACE_MODEL  = ROOT_DIR / "face_landmarker.task"

NUM_BODY = 33
NUM_HAND = 21

FACE_INDICES = (
    10, 152, 234, 454, 132, 58, 288, 361,
    70, 107, 336, 300,
    33, 133, 159, 145,
    362, 263, 386, 374,
    1, 4,
    61, 291, 13, 14,
    78, 308, 87, 317,
)
NUM_FACE = len(FACE_INDICES)

BODY_EDGES = [
    (0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
    (11,12),(11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
    (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
    (11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),
    (27,29),(28,30),(29,31),(30,32),(27,31),(28,32),
]

HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]

FACE_EDGES_LOCAL = [
    (0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0),
    (8,9),(10,11),
    (12,13),(14,12),(14,13),
    (16,17),(18,16),(18,17),
    (20,21),
    (22,23),(24,22),(24,23),
    (26,27),(28,26),(28,27),
    (22,26),
]

COLOR_BODY  = (60, 240, 60)
COLOR_LHAND = (60, 60, 240)
COLOR_RHAND = (240, 90, 90)
COLOR_FACE  = (60, 240, 240)

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

def _draw_skeleton(frame, points, edges, color, point_radius=3, line_thickness=1):
    h, w = frame.shape[:2]
    for (i, j) in edges:
        if i >= len(points) or j >= len(points):
            continue
        p1, p2 = points[i], points[j]
        if p1 is None or p2 is None:
            continue
        x1, y1 = int(p1[0] * w), int(p1[1] * h)
        x2, y2 = int(p2[0] * w), int(p2[1] * h)
        cv2.line(frame, (x1, y1), (x2, y2), color, line_thickness, cv2.LINE_AA)
    for p in points:
        if p is None:
            continue
        x, y = int(p[0] * w), int(p[1] * h)
        cv2.circle(frame, (x, y), point_radius, color, -1, cv2.LINE_AA)

def process_video(video_path: Path, out_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Could not open {video_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    pose_lm = vision.PoseLandmarker.create_from_options(_make_pose_options())
    hand_lm = vision.HandLandmarker.create_from_options(_make_hand_options())
    face_lm = vision.FaceLandmarker.create_from_options(_make_face_options())

    stats = {"frames": 0, "pose": 0, "lhand": 0, "rhand": 0, "face": 0}
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

            body_pts  = [None] * NUM_BODY
            lhand_pts = [None] * NUM_HAND
            rhand_pts = [None] * NUM_HAND
            face_pts  = [None] * NUM_FACE

            if pose_res.pose_landmarks:
                stats["pose"] += 1
                for i, lm in enumerate(pose_res.pose_landmarks[0]):
                    body_pts[i] = (lm.x, lm.y)

            if hand_res.hand_landmarks:
                for hand_lms, handedness in zip(hand_res.hand_landmarks, hand_res.handedness):
                    label = handedness[0].category_name
                    target = lhand_pts if label == "Left" else rhand_pts
                    if target is lhand_pts:
                        stats["lhand"] += 1
                    else:
                        stats["rhand"] += 1
                    for i, lm in enumerate(hand_lms):
                        target[i] = (lm.x, lm.y)

            if face_res.face_landmarks:
                stats["face"] += 1
                face_lms = face_res.face_landmarks[0]
                for slot, src_idx in enumerate(FACE_INDICES):
                    if src_idx < len(face_lms):
                        lm = face_lms[src_idx]
                        face_pts[slot] = (lm.x, lm.y)

            _draw_skeleton(frame, body_pts,  BODY_EDGES, COLOR_BODY)
            _draw_skeleton(frame, lhand_pts, HAND_EDGES, COLOR_LHAND, point_radius=2)
            _draw_skeleton(frame, rhand_pts, HAND_EDGES, COLOR_RHAND, point_radius=2)
            _draw_skeleton(frame, face_pts,  FACE_EDGES_LOCAL, COLOR_FACE, point_radius=2)

            legend = [
                ("body 33",     COLOR_BODY),
                ("L hand 21",   COLOR_LHAND),
                ("R hand 21",   COLOR_RHAND),
                (f"face {NUM_FACE}",  COLOR_FACE),
            ]
            for i, (text, col) in enumerate(legend):
                y0 = 18 + i * 18
                cv2.rectangle(frame, (8, y0 - 10), (24, y0 + 4), col, -1)
                cv2.putText(frame, text, (30, y0 + 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)

            writer.write(frame)
            stats["frames"] += 1
            frame_idx += 1
    finally:
        pose_lm.close()
        hand_lm.close()
        face_lm.close()
        cap.release()
        writer.release()

    f = max(1, stats["frames"])
    print(f"  {video_path.name}: {stats['frames']} frames | "
          f"pose {100*stats['pose']/f:.0f}% | "
          f"Lhand {100*stats['lhand']/f:.0f}% | "
          f"Rhand {100*stats['rhand']/f:.0f}% | "
          f"face {100*stats['face']/f:.0f}%")
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",     type=int, default=5)
    ap.add_argument("--seed",  type=int, default=42)
    ap.add_argument("--split", default="val",
                    help="Which split CSV to sample from (default: val)")
    ap.add_argument("--videos", nargs="*", default=None,
                    help="Explicit video paths (skip random sampling)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for tf in (POSE_MODEL, HAND_MODEL, FACE_MODEL):
        if not tf.exists():
            print(f"ERROR: missing task file {tf}")
            sys.exit(1)

    if args.videos:
        videos = [Path(v) for v in args.videos]
    else:
        csv_path = SPLITS_DIR / f"{args.split}_split.csv"
        if not csv_path.exists():
            print(f"ERROR: missing splits CSV {csv_path}")
            sys.exit(1)
        df = pd.read_csv(csv_path)
        rng = random.Random(args.seed)
        rows = rng.sample(list(df.itertuples(index=False)), k=min(args.n, len(df)))
        videos = [Path(_resolve_video_path(r.video_path)) for r in rows]

    print(f"Rendering {len(videos)} clip(s) to {OUT_DIR}\n")
    for v in videos:
        if not v.exists():
            print(f"  SKIP (missing): {v}")
            continue
        out_name = f"{v.parent.name}__{v.stem}_overlay.mp4"
        out_path = OUT_DIR / out_name
        if out_path.exists():
            out_path.unlink()
        ok = process_video(v, out_path)
        if ok:
            print(f"    -> {out_path}\n")

if __name__ == "__main__":
    main()
