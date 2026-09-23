from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from .util import LOG

# indices into the 33 pose landmarks we care about for the skeleton
BODY_INDICES = list(range(0, 33))

# Face Mesh landmark indices (478 landmarks with refine_landmarks=True)
LEFT_EYE = ((33, 133), ((160, 144), (158, 153)))  # corners, (upper, lower) pairs
RIGHT_EYE = ((362, 263), ((387, 374), (385, 380)))  # corners, (upper, lower) pairs
MOUTH_OUTER = (13, 14)  # top center, bottom center of outer lips
MOUTH_CORNERS = (61, 291)  # left / right mouth corners


def extract_poses(video_path: Path, work_dirs: dict, video_id: str, target_fps: int, max_duration_sec: int) -> Path:
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps = min(target_fps, max(10.0, float(src_fps)))
    frame_ms = 1000.0 / fps
    max_frames = int(max_duration_sec * fps + 1)

    all_lm = []
    all_vis = []
    all_face = []
    prev_lm = None
    prev_face = None

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_face = mp.solutions.face_mesh
    face_mesh = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        for idx in range(max_frames):
            pos_ms = idx * frame_ms
            cap.set(cv2.CAP_PROP_POS_MSEC, pos_ms)
            ok, frame = cap.read()
            if not ok or frame is None:
                if idx >= 8:
                    break
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = pose.process(rgb)
            landmarks = pick_pose(result, BODY_INDICES)

            fm = face_mesh.process(rgb)
            metrics = extract_face(fm)

            if metrics is not None:
                prev_face = metrics

            if landmarks is None:
                if prev_lm is None:
                    continue
                lm = np.full((33, 3), np.nan, dtype=np.float32)
                lm[:, :2] = prev_lm[:, :2]
                lm[:, 2] = 0.0
                all_lm.append(lm[:, :2])
                all_vis.append(lm[:, 2])
            else:
                prev_lm = landmarks
                all_lm.append(landmarks[:, :2])
                all_vis.append(landmarks[:, 2])

            all_face.append(prev_face if prev_face is not None else np.zeros(4, dtype=np.float32))
    finally:
        cap.release()
        pose.close()
        face_mesh.close()

    if not all_lm:
        raise RuntimeError(f"No pose detected in {video_path.name}")

    coords = np.asarray(all_lm, dtype=np.float32)
    vis = np.asarray(all_vis, dtype=np.float32)
    face = np.asarray(all_face, dtype=np.float32)
    out = work_dirs["poses"] / f"{video_id}.npz"
    np.savez(out, coords=coords, vis=vis, face=face, fps=fps)
    LOG.info(
        "Extracted %d pose+face frames (%ds @ %.1ffps) -> %s",
        coords.shape[0],
        round(coords.shape[0] / fps),
        fps,
        out.name,
    )
    return out


def pick_pose(result, indices) -> np.ndarray | None:
    """Return the 33-landmark array if a usable pose was detected, else None."""
    if not result or not result.pose_landmarks:
        return None
    pts = np.asarray(
        [[p.x, p.y, p.visibility] for p in result.pose_landmarks.landmark], dtype=np.float32
    )
    score = float(np.mean(pts[indices, 2]))
    if score < 0.25:
        return None
    return pts


def extract_face(fm_result) -> np.ndarray | None:
    """Return [eye_open_l, eye_open_r, mouth_open, smile] normalized, or None."""
    if not fm_result or not fm_result.multi_face_landmarks:
        return None
    lm = np.asarray(
        [[p.x, p.y] for p in fm_result.multi_face_landmarks[0].landmark], dtype=np.float32
    )

    eye_l = _eye_openness(lm, *LEFT_EYE)
    eye_r = _eye_openness(lm, *RIGHT_EYE)
    top, bottom = lm[13], lm[14]
    inter_eye = float(
        np.hypot(
            lm[133][0] + lm[33][0] - lm[362][0] - lm[263][0],
            lm[133][1] + lm[33][1] - lm[362][1] - lm[263][1],
        )
        * 0.5
    )
    if inter_eye < 1e-6:
        inter_eye = 1.0
    mouth_open = float(np.hypot(top[0] - bottom[0], top[1] - bottom[1])) / inter_eye
    lip_mid_y = float((top[1] + bottom[1]) * 0.5)
    corner_y = float((lm[61][1] + lm[291][1]) * 0.5)
    smile = (lip_mid_y - corner_y) / inter_eye
    return np.asarray([eye_l, eye_r, mouth_open, smile], dtype=np.float32)


def _eye_openness(lm, corners, pairs) -> float:
    """Eye Aspect Ratio projected onto the eye axis (robust to head tilt)."""
    (ax, ay), (bx, by) = lm[corners[0]], lm[corners[1]]
    width = float(np.hypot(bx - ax, by - ay))
    if width < 1e-6:
        return 0.4
    px, py = (by - ay) / width, (ax - bx) / width
    vals = []
    for upper, lower in pairs:
        dx, dy = lm[upper][0] - lm[lower][0], lm[upper][1] - lm[lower][1]
        vals.append(abs(dx * px + dy * py))
    return float(np.mean(vals)) / width