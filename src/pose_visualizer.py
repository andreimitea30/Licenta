import sys
from pathlib import Path

import cv2
import numpy as np

POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


def visualize_skeleton(video_path, npy_path):
    video_path = Path(video_path)
    npy_path = Path(npy_path)

    if not video_path.exists():
        print(f"Error: Video not found at {video_path}")
        return
    if not npy_path.exists():
        print(f"Error: Skeleton data not found at {npy_path}")
        return

    skeleton_data = np.load(npy_path)
    print(f"Loaded skeleton data shape: {skeleton_data.shape}")

    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0

    print("Playing video... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        if frame_idx < len(skeleton_data):
            frame_landmarks = skeleton_data[frame_idx]

            for start_idx, end_idx in POSE_CONNECTIONS:
                pt1 = frame_landmarks[start_idx]
                pt2 = frame_landmarks[end_idx]
                if pt1[2] > 0.5 and pt2[2] > 0.5:
                    x1, y1 = int(pt1[0] * w), int(pt1[1] * h)
                    x2, y2 = int(pt2[0] * w), int(pt2[1] * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for landmark in frame_landmarks:
                if landmark[2] > 0.5:
                    x, y = int(landmark[0] * w), int(landmark[1] * h)
                    cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        cv2.imshow('Skeleton Verification', frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).resolve().parent.parent

    if len(sys.argv) == 3:
        video_arg = sys.argv[1]
        npy_arg = sys.argv[2]
    else:
        video_arg = ROOT_DIR / "haa500_v1_1" / "video" / "air_guitar" / "air_guitar_001.mp4"
        npy_arg = ROOT_DIR / "extracted_skeletons" / "train" / "air_guitar" / "air_guitar_001.npy"

    visualize_skeleton(video_arg, npy_arg)
