from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
SKELETONS_DIR = ROOT_DIR / "extracted_skeletons"
VIDEO_DIR = ROOT_DIR / "haa500_v1_1" / "video"
OUTPUT_VIDEO = ROOT_DIR / "skeleton_compilation.mp4"

OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 360
OUTPUT_FPS = 25.0
CLIP_SECONDS = 3
CLIP_FRAMES = int(OUTPUT_FPS * CLIP_SECONDS)

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def draw_skeleton(frame, landmarks, width, height):
    for start_idx, end_idx in POSE_CONNECTIONS:
        pt1, pt2 = landmarks[start_idx], landmarks[end_idx]
        if pt1[2] > 0.5 and pt2[2] > 0.5:
            x1, y1 = int(pt1[0] * width), int(pt1[1] * height)
            x2, y2 = int(pt2[0] * width), int(pt2[1] * height)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for landmark in landmarks:
        if landmark[2] > 0.5:
            x, y = int(landmark[0] * width), int(landmark[1] * height)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def draw_label(frame, class_name):
    label = class_name.replace("_", " ")
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.7, 2
    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
    pad = 6
    cv2.rectangle(frame, (8, 8), (8 + tw + pad * 2, 8 + th + pad * 2 + baseline), (0, 0, 0), -1)
    cv2.putText(frame, label, (8 + pad, 8 + pad + th), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def find_clip_for_class(class_name):
    for split in ("train", "val", "test"):
        npy_dir = SKELETONS_DIR / split / class_name
        if not npy_dir.exists():
            continue
        for npy_path in sorted(npy_dir.glob("*.npy")):
            stem = npy_path.stem
            for ext in (".mp4", ".avi"):
                video_path = VIDEO_DIR / class_name / (stem + ext)
                if video_path.exists():
                    return video_path, npy_path
    return None, None


def render_clip(cap, skeleton_data, class_name, writer):
    src_fps = cap.get(cv2.CAP_PROP_FPS) or OUTPUT_FPS
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_frames_needed = min(int(src_fps * CLIP_SECONDS), total_src_frames)

    out_frame_idx = 0
    for i in range(CLIP_FRAMES):
        src_idx = int(i * src_frames_needed / CLIP_FRAMES)
        cap.set(cv2.CAP_PROP_POS_FRAMES, src_idx)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        skel_idx = min(int(i * len(skeleton_data) / CLIP_FRAMES), len(skeleton_data) - 1)
        draw_skeleton(frame, skeleton_data[skel_idx], OUTPUT_WIDTH, OUTPUT_HEIGHT)
        draw_label(frame, class_name)
        writer.write(frame)
        out_frame_idx += 1

    if out_frame_idx < CLIP_FRAMES and out_frame_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_src_frames - 1)
        ret, last_frame = cap.read()
        if ret:
            last_frame = cv2.resize(last_frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
            skel_idx = len(skeleton_data) - 1
            draw_skeleton(last_frame, skeleton_data[skel_idx], OUTPUT_WIDTH, OUTPUT_HEIGHT)
            draw_label(last_frame, class_name)
            for _ in range(CLIP_FRAMES - out_frame_idx):
                writer.write(last_frame)


def main():
    classes = sorted([d.name for d in VIDEO_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(classes)} classes.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(OUTPUT_VIDEO), fourcc, OUTPUT_FPS, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    if not writer.isOpened():
        print(f"Error: Could not open video writer for {OUTPUT_VIDEO}")
        return

    skipped = 0
    for class_name in tqdm(classes, desc="Compiling clips"):
        video_path, npy_path = find_clip_for_class(class_name)
        if video_path is None:
            tqdm.write(f"  Skipped (no paired video+skeleton): {class_name}")
            skipped += 1
            continue

        skeleton_data = np.load(npy_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            tqdm.write(f"  Skipped (cannot open video): {class_name}")
            skipped += 1
            continue

        render_clip(cap, skeleton_data, class_name, writer)
        cap.release()

    writer.release()

    if skipped:
        print(f"\nSkipped {skipped} / {len(classes)} classes.")
    print(f"Compilation saved to {OUTPUT_VIDEO}")
    print(f"Total duration: ~{(len(classes) - skipped) * CLIP_SECONDS / 60:.1f} minutes")


if __name__ == "__main__":
    main()
