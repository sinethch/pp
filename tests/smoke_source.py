"""Local smoke test for background=source (original video as background).

Builds a synthetic 'dance' video on the fly (no network/MediaPipe needed):
a vivid magenta blob acting as the dancer over a static blue scene, then
renders the stick figure on it and checks the blob was replaced by the
background while the figure is drawn on top. Requires opencv-python.

Run with:  python tests/smoke_source.py
"""

import math
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.smoke_render import make_synthetic
from src.config import ensure_work_dirs
from src.render import FunnyStickRenderer


def make_source_video(path: Path, coords, n_frames: int, fps: float, w: int, h: int):
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    t = np.linspace(0, 4, n_frames)
    xc = coords[:, 11, 0]  # follow the left shoulder ≈ body
    yc = coords[:, 11, 1]
    for i in range(n_frames):
        frame = np.full((h, w, 3), (235, 234, 224), dtype=np.uint8)  # light paper bg
        # static blue shapes
        cv2.rectangle(frame, (20, 40), (140, 120), (200, 200, 120), -1)
        cv2.circle(frame, (w - 90, 90), 50, (120, 160, 220), 10)
        cv2.line(frame, (0, 0), (w, h // 2), (150, 150, 160), 6)
        cv2.line(frame, (0, h), (w, h // 2), (150, 150, 160), 6)
        # magenta dancer blob following the body
        cx, cy = int(xc[i] * w), int(yc[i] * h)
        cv2.ellipse(frame, (cx, cy - 80), (50, 70), 0, 0, 360, (255, 30, 180), -1)
        cv2.ellipse(frame, (cx, cy + 20), (40, 90), 0, 0, 360, (255, 30, 180), -1)
        vw.write(frame)
    vw.release()


def main() -> None:
    n_frames, fps, w, h = 90, 30.0, 540, 960
    cfg = {
        "render": {
            "width": w, "height": h, "fps": fps, "line_width": 14,
            "stick_color": "#22d3ee", "head_color": "#fef08a",
            "background": "source", "confetti": False,
        },
        "work": {"dir": "work", "raw_dir": "raw", "poses_dir": "poses", "frames_dir": "frames", "out_dir": "out"},
    }
    dirs = ensure_work_dirs(cfg)
    for d in (dirs["frames"], dirs["poses"]):
        shutil.rmtree(d, ignore_errors=True)

    coords, vis, face, _ = make_synthetic(n_frames, fps)
    coords = coords * np.array([1.0, 0.9]) + np.array([0.0, 0.06])  # center figure mid-frame
    video = dirs["raw"] / "synth.mp4"
    make_source_video(video, coords, n_frames, fps, w, h)

    poses_file = dirs["poses"] / "synth.npz"
    dirs["poses"].mkdir(parents=True, exist_ok=True)
    np.savez(poses_file, coords=coords, vis=vis, face=face, fps=fps)

    r = FunnyStickRenderer(cfg)
    frame_dir, _, _ = r.render(poses_file, "synth", dirs, "Source Bg Test", "SourceMode", source_video=video)

    # checks on a middle frame
    from PIL import Image

    im = np.asarray(Image.open(list(sorted(frame_dir.glob("*.jpg")))[n_frames // 2]).convert("RGB")).astype(int)
    magenta = ((im[..., 0] > 200) & (im[..., 1] < 90) & (im[..., 2] > 140) & (im[..., 2] < 220)).sum()
    cyan = ((im[..., 0] < 90) & (im[..., 1] > 180) & (im[..., 2] > 200)).sum()
    paper = ((im[..., 0] > 220) & (im[..., 1] > 215) & (im[..., 2] > 205)).sum()
    print(f"residual magenta = {magenta} (expect ~0, dancer erased)")
    print(f"stick cyan = {cyan} (expect > 1000, figure drawn)")
    print(f"background paper px = {paper} (expect a lot, source bg kept)")
    assert magenta < 400, "dancer blob was not erased"
    assert cyan > 1000, "stick figure missing"
    print("OK - source background pipeline works")


if __name__ == "__main__":
    main()