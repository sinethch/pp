"""Local smoke test for the stick-figure renderer using synthetic dance poses.

Requires only numpy + Pillow. Run with:
    python tests/smoke_render.py
"""

import math
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import DEFAULTS, ensure_work_dirs
from src.render import FunnyStickRenderer


def make_synthetic(n_frames: int = 120, fps: float = 30.0):
    """A little dancer: hips sway, arms swing, knees bounce, head bobs."""
    t = np.linspace(0, 1, n_frames)
    cx, cy = 0.5, 0.62
    sway = 0.05 * np.sin(t * math.tau * 2)
    hip_y = cy + 0.03 * np.sin(t * math.tau + 0.3)
    sho_y = hip_y - 0.20 - 0.02 * np.sin(t * math.tau * 2)
    head_y = sho_y - 0.16
    ph = math.tau * 2

    def P(x, y):
        return np.column_stack([x, y])

    coords = np.zeros((n_frames, 33, 2), dtype=np.float32)
    coords[:, 0] = P(cx + sway * 1.2, head_y)
    coords[:, 7] = P(cx - 0.07 + sway, head_y + 0.05)
    coords[:, 8] = P(cx + 0.07 + sway, head_y + 0.05)
    coords[:, 11] = P(cx - 0.09 + sway, sho_y)
    coords[:, 12] = P(cx + 0.09 + sway, sho_y)
    coords[:, 13] = P(cx - 0.22 + sway + 0.02 * np.sin(ph * t), sho_y + 0.02 + 0.13 * np.sin(ph * t + 0.5))
    coords[:, 14] = P(cx + 0.22 + sway - 0.02 * np.cos(ph * t), sho_y + 0.02 + 0.13 * np.sin(ph * t + 2.0))
    coords[:, 15] = P(cx - 0.40 + sway + 0.05 * np.sin(ph * t), sho_y + 0.30 + 0.12 * np.sin(ph * t + 0.5))
    coords[:, 16] = P(cx + 0.40 + sway - 0.05 * np.cos(ph * t), sho_y + 0.30 + 0.12 * np.sin(ph * t + 2.0))
    coords[:, 23] = P(cx - 0.08 + sway, hip_y)
    coords[:, 24] = P(cx + 0.08 + sway, hip_y)
    coords[:, 25] = P(cx - 0.09 + sway + 0.02 * np.sin(ph * t), hip_y + 0.24)
    coords[:, 26] = P(cx + 0.09 + sway - 0.02 * np.cos(ph * t), hip_y + 0.24)
    coords[:, 27] = P(cx - 0.10 + sway + 0.03 * np.sin(ph * t), hip_y + 0.40)
    coords[:, 28] = P(cx + 0.10 + sway - 0.03 * np.cos(ph * t), hip_y + 0.40)

    vis = np.full((n_frames, 33), 0.9, dtype=np.float32)

    # face metrics: [eye_l, eye_r, mouth_open, smile]
    face = np.zeros((n_frames, 4), dtype=np.float32)
    face[:, 3] = 0.12
    blink = [int(n_frames * 0.25), int(n_frames * 0.6)]
    for i in range(n_frames):
        open_l, open_r = 0.30, 0.30
        for b in blink:
            if abs(i - b) <= 4:
                k = max(0.0, 1 - abs(i - b) / 4.0)  # deep 1-2 frame closure
                open_l = min(open_l, 0.02 + 0.7 * k)
                open_r = min(open_r, 0.02 + 0.7 * k)
        face[i, 0], face[i, 1] = open_l, open_r
        if int(n_frames * 0.3) <= i <= int(n_frames * 0.75):
            face[i, 2] = 0.06 + 0.5 * (0.5 + 0.5 * math.sin(i * 0.9))
        else:
            face[i, 2] = 0.06
    return coords, vis, face, fps


def main() -> None:
    cfg = dict(DEFAULTS)
    dirs = ensure_work_dirs(cfg)
    if dirs["frames"].exists():
        shutil.rmtree(dirs["frames"])

    coords, vis, face, fps = make_synthetic()
    poses_file = dirs["poses"] / "synthetic.npz"
    np.savez(poses_file, coords=coords, vis=vis, face=face, fps=fps)

    renderer = FunnyStickRenderer(cfg)
    frame_dir, used_fps, n = renderer.render(poses_file, "synthetic", dirs, "Synthetic Dancer", "SmokeTest")
    frames = sorted(frame_dir.glob("*.jpg"))
    assert len(frames) == n, f"expected {n} frames, got {len(frames)}"
    assert frames, "no frames rendered"
    preview = ROOT / "work" / "preview.jpg"
    shutil.copy(frames[0], preview)
    # sanity check: eyes and mouth pixels exist on the rendered head
    im = np.asarray(Image.open(frames[len(frames) // 2]).convert("RGB")).astype(int)
    head_mask = (im[..., 0] > 200) & (im[..., 1] > 190) & (im[..., 2] < 200)  # yellow head
    ys, xs = np.where(head_mask)
    if len(xs):
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        region = im[y0:y1 + 1, x0:x1 + 1]
        dark = ((region[..., 0] < 90) & (region[..., 1] < 90) & (region[..., 2] < 90)).sum()
        white = ((region[..., 0] > 200) & (region[..., 1] > 200) & (region[..., 2] > 200)).sum()
        red = ((region[..., 0] > 100) & (region[..., 1] < 90) & (region[..., 2] < 110)).sum()
        print(f"face sanity: head {len(xs)}px, ink(dark) {dark}px, white {white}px, mouth-red {red}px")
    print(f"OK - {len(frames)} frames rendered at {used_fps:.1f} fps; preview saved to {preview}")


if __name__ == "__main__":
    main()