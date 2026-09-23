import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .util import LOG

# body landmark indices used to build the figure
BODY = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
LIMBS = [
    (11, 13, 15),  # left arm
    (12, 14, 16),  # right arm
    (11, 23), (12, 24),  # torso
    (23, 25, 27),  # left leg
    (24, 26, 28),  # right leg
    (11, 12),  # shoulders
]

INK = (30, 30, 40, 255)
SKETCH_INK = (59, 66, 82, 255)
WHITE = (255, 255, 255, 255)

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def _hex(color: str) -> tuple:
    c = color.lstrip("#")
    return (int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16), 255)


def _seed_rng(seed: int):
    return random.Random(seed)


def _smooth(coords: np.ndarray, window: int = 5) -> np.ndarray:
    if window < 2:
        return coords
    pad = window // 2
    padded = np.pad(coords, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    for i in range(33):
        padded[:, i, 0] = np.convolve(padded[:, i, 0], kernel, mode="same")
        padded[:, i, 1] = np.convolve(padded[:, i, 1], kernel, mode="same")
    return padded[pad:-pad]


def _smooth_faces(face: np.ndarray, window: int = 3) -> np.ndarray:
    """Median filter per metric column - removes noise but preserves sharp blinks / lip flaps."""
    if window < 2:
        return face
    pad = window // 2
    padded = np.pad(face, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(face)
    for t in range(len(face)):
        out[t] = np.median(padded[t : t + window], axis=0)
    return out


class FunnyStickRenderer:
    def __init__(self, config: dict):
        r = config["render"]
        self.W = int(r["width"])
        self.H = int(r["height"])
        self.line_w = int(r.get("line_width", 16))
        self.stick = _hex(r.get("stick_color", "#22d3ee"))
        self.head = _hex(r.get("head_color", "#fef08a"))
        self.bg_top = _hex(r.get("bg_top", "#0b0f1a"))
        self.bg_bottom = _hex(r.get("bg_bottom", "#2a1450"))
        self.confetti_on = bool(r.get("confetti", True))
        self.style = str(r.get("background", "neon"))
        self.top_band = int(self.H * 0.09)
        self.bottom_band = int(self.H * 0.13)

    # ----------------------------------------------------------------- setup
    def _background(self) -> Image.Image:
        img = Image.new("RGB", (self.W, self.H))
        px = img.load()
        rng = _seed_rng(self._bg_seed)
        for y in range(self.H):
            t = y / self.H
            if self.style == "sketch":
                c0, c1 = (247, 243, 232), (232, 224, 205)
            else:
                c0, c1 = self.bg_top, self.bg_bottom
            r = int(c0[0] + (c1[0] - c0[0]) * t)
            g = int(c0[1] + (c1[1] - c0[1]) * t)
            b = int(c0[2] + (c1[2] - c0[2]) * t)
            color = (r, g, b)
            for x in range(0, self.W):
                px[x, y] = color

        if self.style == "sketch":
            d = ImageDraw.Draw(img, "RGBA")
            self._draw_paper_grain(d, rng)
            self._draw_sketch_scene(d, rng)
        return img

    def _fit(self, coords, vis):
        body_vis = vis[:, BODY] > 0.6
        xs = coords[:, BODY, 0][body_vis]
        ys = coords[:, BODY, 1][body_vis]
        if len(xs) == 0:
            min_x, max_x, min_y, max_y = 0.05, 0.95, 0.05, 0.95
        else:
            min_x, max_x, min_y, max_y = xs.min(), xs.max(), ys.min(), ys.max()
        pad_x = (max_x - min_x) * 0.06
        pad_y = (max_y - min_y) * 0.06
        min_x, max_x = min_x - pad_x, max_x + pad_x
        min_y, max_y = min_y - pad_y, max_y + pad_y

        top, bottom = self.top_band, self.H - self.bottom_band
        play_h, play_w = bottom - top, self.W - 90
        scale = min(play_h / max(max_y - min_y, 1e-6), play_w / max(max_x - min_x, 1e-6))
        out_w, out_h = (max_x - min_x) * scale, (max_y - min_y) * scale
        ox, oy = (self.W - out_w) / 2, top + (play_h - out_h) / 2
        return {"scale": scale, "ox": ox, "oy": oy, "min_x": min_x, "min_y": min_y}

    def _map_frame(self, coords, t, fit):
        c = coords[t]
        pts = np.empty((33, 2), dtype=np.float32)
        for i in range(33):
            pts[i, 0] = fit["ox"] + (c[i, 0] - fit["min_x"]) * fit["scale"]
            pts[i, 1] = fit["oy"] + (c[i, 1] - fit["min_y"]) * fit["scale"]

        shoulder_w = self._dist(pts[11], pts[12])
        head_r = max(38.0, min(110.0, shoulder_w * 0.55))

        nose = pts[0]
        head_c = np.array([float(nose[0]), float(nose[1])]) if np.isfinite(nose).all() else None

        ankles = [pts[i] for i in (25, 26, 27, 28) if np.isfinite(pts[i]).all()]
        feet = np.mean(ankles, axis=0) if ankles else np.array([self.W / 2, self.H - self.bottom_band])
        return pts, head_c, head_r, feet

    # ------------------------------------------------------------- rendering
    def render(self, npz_path: Path, video_id: str, work_dirs: dict, title_text: str, tag: str, source_video=None):
        data = np.load(npz_path)
        coords = _smooth(data["coords"], 5)
        vis = data["vis"]
        fps = float(data["fps"])
        n_frames = len(coords)

        if "face" in data.files:
            face = _smooth_faces(data["face"].astype(np.float32), 3)
        else:
            face = np.full((n_frames, 4), 0.35, dtype=np.float32)
            face[:, 2] = 0.06  # default closed mouth
            face[:, 3] = 0.12  # default smile

        fit = self._fit(coords, vis)

        frame_dir = work_dirs["frames"] / video_id
        frame_dir.mkdir(parents=True, exist_ok=True)

        self._bg_seed = sum(ord(ch) * (i + 3) for i, ch in enumerate(video_id))

        sketch = self.style == "sketch"
        source_mode = self.style == "source"
        if source_mode:
            if source_video is None or not Path(source_video).exists():
                LOG.warning("background=source needs a source_video; falling back to neon")
                source_mode, self.style = False, "neon"
            else:
                fps_ms = 1000.0 / fps
                src_w, src_h, median_q = self._load_source_median(source_video, n_frames, fps_ms)
                self._src = {"path": str(source_video), "ms": fps_ms, "w": src_w, "h": src_h, "median": median_q}
                LOG.info("Using source video as background (%dx%d)", src_w, src_h)

        bg = self._background() if not source_mode else None
        rng = random.Random(self._bg_seed)
        confetti = self._make_confetti(rng)
        phases = np.linspace(0, math.tau, 26, endpoint=False)

        font_big = _font(int(self.W * 0.05))
        font_small = _font(int(self.W * 0.028))
        human_scale = max(self.W, self.H) or 1

        for t in range(n_frames):
            t01 = t / max(n_frames - 1, 1)
            if source_mode:
                img, pts, head_c, head_r, feet = self._source_frame(coords, vis, t)
            else:
                img = bg.copy()
                pts, head_c, head_r, feet = self._map_frame(coords, t, fit)
            d = ImageDraw.Draw(img, "RGBA")

            if not source_mode:
                if sketch:
                    self._draw_sketch_anim(d, t01, phases)
                    if self.confetti_on:
                        self._draw_sketch_confetti(d, t01, confetti)
                    self._draw_sketch_eq(d, t01, phases)
                else:
                    self._draw_glow(d, t01)
                    if self.confetti_on:
                        self._draw_confetti(d, t01, confetti)
                    self._draw_music_bars(d, t01, phases)

            face_face = face[t]
            face_layer = None
            if head_c is not None:
                self._draw_shadow(d, feet, head_r, sketch)
                face_layer = self._draw_figure(d, pts, head_c, head_r, t01, face_face)
                sparkle_fit = fit if not source_mode else {"scale": human_scale}
                self._draw_sparkles(d, coords, t, pts, sparkle_fit, sketch)

            self._draw_title(d, title_text, tag, font_big, font_small, sketch)

            if face_layer is not None:
                img.paste(face_layer, (int(head_c[0] - head_r), int(head_c[1] - head_r)), face_layer)

            img.convert("RGB").save(frame_dir / f"{t:05d}.jpg", quality=88)
            if t % 60 == 0:
                LOG.info("  rendered %d/%d frames", t + 1, n_frames)

        LOG.info("Rendered %d frames for %s at %.1f fps", n_frames, video_id, fps)
        return frame_dir, fps, n_frames

    # ---------------------------------------------------- source background
    def _load_source_median(self, source_video, n_frames, frame_ms):
        import cv2

        cap = cv2.VideoCapture(str(source_video))
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 720)
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 1280)
        scale = 0.25
        qw, qh = max(16, int(src_w * scale)), max(28, int(src_h * scale))
        stride = 2 if n_frames > 1600 else 1
        stack = []
        try:
            for i in range(0, n_frames, stride):
                cap.set(cv2.CAP_PROP_POS_MSEC, i * frame_ms)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                small = cv2.resize(frame, (qw, qh), interpolation=cv2.INTER_AREA)
                stack.append(small)
        finally:
            cap.release()
        if not stack:
            raise RuntimeError("Could not decode any source frame for background")
        median = np.median(np.asarray(stack, dtype=np.uint8), axis=0).astype(np.uint8)
        return src_w, src_h, median

    def _body_bbox(self, coords, t, w, h):
        pts_x = coords[t, BODY, 0][np.isfinite(coords[t, BODY, 0])]
        pts_y = coords[t, BODY, 1][np.isfinite(coords[t, BODY, 1])]
        if len(pts_x) == 0:
            return None
        pad = max(8, int(0.03 * max(w, h)))
        x0 = max(0, int(pts_x.min() * w) - pad)
        y0 = max(0, int(pts_y.min() * h) - pad)
        x1 = min(w, int(pts_x.max() * w) + pad)
        y1 = min(h, int(pts_y.max() * h) + pad)
        return x0, y0, x1, y1

    def _source_frame(self, coords, vis, t):
        import cv2

        from PIL import Image as _Image

        src = self._src
        w, h = src["w"], src["h"]
        cap = cv2.VideoCapture(src["path"])
        cap.set(cv2.CAP_PROP_POS_MSEC, t * src["ms"])
        ok, frame = cap.read()
        cap.release()

        c = coords[t]
        pts = np.empty((33, 2), dtype=np.float32)
        pts[:, 0] = c[:, 0] * w
        pts[:, 1] = c[:, 1] * h

        head_c = None
        if np.isfinite(pts[0]).all():
            head_c = np.array([float(pts[0, 0]), float(pts[0, 1])])
        shoulder_w = self._dist(pts[11], pts[12]) if np.isfinite(pts[11]).all() and np.isfinite(pts[12]).all() else w * 0.25
        head_r = max(38.0, min(120.0, shoulder_w * 0.55))

        ankles = [pts[i] for i in (25, 26, 27, 28) if np.isfinite(pts[i]).all()]
        feet = np.mean(ankles, axis=0) if ankles else np.array([w / 2, h * 0.9])

        bbox = self._body_bbox(coords, t, w, h)
        if ok and frame is not None:
            base = frame.copy()
            if bbox is not None:
                x0, y0, x1, y1 = bbox
                if x1 > x0 and y1 > y0:
                    patch = cv2.resize(src["median"], (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
                    base[y0:y1, x0:x1] = patch
            rgb = cv2.cvtColor(base, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.resize(src["median"], (w, h), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        img = _Image.fromarray(rgb)
        return img, pts, head_c, head_r, feet

    def _draw_glow(self, d, t01):
        pulse = 0.5 + 0.5 * math.sin(t01 * math.tau * 2)
        color = (int(90 + 90 * pulse), int(60 + 50 * pulse), int(180 + 60 * pulse), int(20 + 26 * pulse))
        d.ellipse([self.W / 2 - self.W * 0.85, -self.H * 0.45, self.W / 2 + self.W * 0.85, self.H], fill=color)

    def _make_confetti(self, rng):
        parts = []
        for _ in range(46):
            parts.append(
                {
                    "x": rng.random(),
                    "size": rng.randint(6, 16),
                    "speed": 0.35 + rng.random() * 0.85,
                    "phase": rng.random(),
                    "color": (rng.randint(70, 255), rng.randint(70, 255), rng.randint(70, 255), 160),
                }
            )
        return parts

    def _draw_confetti(self, d, t01, parts):
        for p in parts:
            y = ((t01 * p["speed"] * 2.0 + p["phase"]) % 1.05) * self.H - p["size"]
            x = p["x"] * self.W + math.sin(t01 * math.tau * 2 + p["phase"]) * p["size"]
            d.rectangle([x, y, x + p["size"], y + p["size"]], fill=p["color"])

    def _draw_music_bars(self, d, t01, phases):
        n = len(phases)
        bar_w = self.W / n * 0.5
        base_y = self.H - self.bottom_band
        max_h = self.bottom_band - 16
        for i, ph in enumerate(phases):
            beat = 0.5 + 0.5 * math.sin(t01 * math.tau * 2 + ph)
            h = (0.3 + 0.7 * beat) * max_h
            x = i * (self.W / n) + self.W / n * 0.22
            r = int(40 + 55 * (1 - i / n))
            g = int(180 + 60 * (i / n))
            b = int(220 - 80 * (1 - i / n))
            d.rounded_rectangle([x, base_y - h, x + bar_w, base_y], radius=bar_w / 2, fill=(r, g, b, 205))

    # ---------------------------------------------- sketch style helpers
    @staticmethod
    def _squiggle_pts(x0, y0, x1, y1, rng, bump=6.0, steps=24):
        pts = []
        for i in range(steps + 1):
            t = i / steps
            x = x0 + (x1 - x0) * t
            y = y0 + (y1 - y0) * t + (rng.random() - 0.5) * 2 * bump * math.sin(math.pi * t)
            pts.append((x, y))
        return pts

    def _wobble_circle(self, d, cx, cy, r, seed, color, width):
        rng = random.Random(seed)
        pts = []
        for i in range(92):
            a = 2 * math.pi * i / 92
            jit = 1 + (rng.random() - 0.5) * 0.045
            pts.append((cx + math.cos(a) * r * jit, cy + math.sin(a) * r * jit))
        pts.append(pts[0])
        d.line(pts, fill=color, width=width, joint="curve")

    def _draw_rot(self, pts, cx, cy, ang):
        c, s = math.cos(ang), math.sin(ang)
        return [(cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c) for x, y in pts]

    def _draw_note(self, d, cx, cy, s, seed, color, rot=0.0):
        rng = random.Random(seed)
        head = [(cx + (rng.random() - 0.5) * s * 0.1, cy + (rng.random() - 0.5) * s * 0.1)]
        full = self._draw_rot(head, cx, cy, rot)
        hx, hy = full[0]
        d.ellipse([hx - s * 0.72, hy - s * 0.55, hx + s * 0.72, hy + s * 0.55], fill=color)
        tx, ty = self._rotate_pt(cx + s * 0.5, cy - s * 2.3, cx, cy, rot)
        d.line([(hx + s * 0.4, hy - s * 0.45), (tx, ty)], fill=color, width=max(3, int(s * 0.18)))
        flag = self._draw_rot(
            [(tx, ty), (tx + s * 0.7, ty + s * 0.5), (tx, ty + s * 1.15), (tx + s * 0.4, ty + s * 1.55), (tx, ty + s * 2.0)],
            cx, cy, rot,
        )
        d.line(flag + [flag[0]], fill=color, width=max(2, int(s * 0.14)))

    def _rotate_pt(self, x, y, cx, cy, ang):
        c, s = math.cos(ang), math.sin(ang)
        return (cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c)

    def _draw_star4(self, d, cx, cy, r, color, width, rot=0.0):
        p = [
            (cx, cy - r), (cx + r * 0.22, cy - r * 0.22), (cx + r, cy),
            (cx + r * 0.22, cy + r * 0.22), (cx, cy + r),
            (cx - r * 0.22, cy + r * 0.22), (cx - r, cy),
            (cx - r * 0.22, cy - r * 0.22),
        ]
        d.line(self._draw_rot(p, cx, cy, rot), fill=color, width=width, joint="curve")

    def _draw_paper_grain(self, d, rng):
        for _ in range(320):
            x = int(rng.random() * self.W)
            y = int(rng.random() * self.H)
            a = int(rng.random() * 18)
            d.point((x, y), fill=(120, 110, 90, a))

    def _draw_sketch_scene(self, d, rng):
        ink = SKETCH_INK
        gy = self.H - self.bottom_band + 34
        # squiggly ground
        pts = self._squiggle_pts(0, gy, self.W, gy, rng, bump=14, steps=40)
        d.line(pts, fill=ink, width=4)
        # hatching under the ground
        hatch_rng = random.Random(self._bg_seed + 7)
        for _ in range(90):
            x = hatch_rng.random() * self.W
            y = gy + 6 + hatch_rng.random() * 40
            le = 10 + hatch_rng.random() * 18
            ang = 0.5 + hatch_rng.random() * 0.5
            d.line([(x, y), (x + math.cos(ang) * le, y + math.sin(ang) * le)], fill=(59, 66, 82, 60), width=2)
        # doodle sun (top right, static body; rays animated per-frame)
        sun = (self.W * 0.84, self.H * 0.13)
        self._sun = sun
        self._sun_r = 64
        self._wobble_circle(d, sun[0], sun[1], self._sun_r, self._bg_seed + 1, ink, 5)
        self._wobble_circle(d, sun[0] - 14, sun[1] + 12, self._sun_r + 10, self._bg_seed + 2, ink, 3)
        # bouncy-ball doodles
        self._wobble_circle(d, self.W * 0.13, self.H * 0.28, 46, self._bg_seed + 3, ink, 4)
        self._wobble_circle(d, self.W * 0.5, self.H * 0.12, 30, self._bg_seed + 4, ink, 4)
        # cloud puffs
        for k, (cx, cy, r0) in enumerate(((0.22, 0.07, 34), (0.68, 0.24, 26), (0.05, 0.52, 22))):
            self._wobble_circle(d, self.W * cx, self.H * cy, r0, self._bg_seed + 10 + k, ink, 3)
            self._wobble_circle(d, self.W * cx + r0 * 0.9, self.H * cy + r0 * 0.35, r0 * 0.8, self._bg_seed + 20 + k, ink, 3)
        # hand-drawn stars
        star_rng = random.Random(self._bg_seed + 33)
        for k in range(6):
            self._draw_star4(
                d,
                star_rng.random() * self.W,
                star_rng.random() * (self.H * 0.75),
                22 + star_rng.random() * 22,
                ink,
                4,
                rot=star_rng.random() * math.tau,
            )

    def _draw_sketch_anim(self, d, t01, phases):
        ink = SKETCH_INK
        if not hasattr(self, "_sun"):
            return
        # rotating sun rays
        base = (self._sun[0], self._sun[1])
        rot = t01 * math.tau * 0.35 + 0.4
        for k in range(11):
            a = 2 * math.pi * k / 11 + rot
            x0 = base[0] + math.cos(a) * (self._sun_r + 16)
            y0 = base[1] + math.sin(a) * (self._sun_r + 16)
            x1 = base[0] + math.cos(a) * (self._sun_r + 34 + (1 if k % 2 == 0 else 0))
            y1 = base[1] + math.sin(a) * (self._sun_r + 34 + (1 if k % 2 == 0 else 0))
            wob = (k * 0.7 + t01 * 2) % 1
            d.line([(x0 + wob * 3, y0 - wob * 2), (x1, y1)], fill=ink, width=4)
        # bobbing music notes
        for k, (fx, fy) in enumerate(((0.12, 0.42), (0.3, 0.22), (0.6, 0.34), (0.9, 0.3))):
            bx = self.W * fx + math.sin(t01 * math.tau + k) * 18
            by = self.H * fy + math.sin(t01 * math.tau * 2 + k * 1.3) * 22
            self._draw_note(d, bx, by, 26 + 6 * (k % 2), self._bg_seed + k * 5, ink, rot=t01 * 0.7 + k * 0.4)

    def _draw_sketch_confetti(self, d, t01, parts):
        for p in parts:
            y = ((t01 * p["speed"] * 2.0 + p["phase"]) % 1.05) * self.H - p["size"]
            x = p["x"] * self.W + math.sin(t01 * math.tau * 2 + p["phase"]) * p["size"]
            rot = t01 * math.tau + p["phase"] * 4
            if p["size"] % 3 == 0:
                self._draw_star4(d, x, y, p["size"], (59, 66, 82, 170), 3, rot)
            else:
                d.line(
                    [(x - p["size"], y), (x + p["size"], y), (x, y + p["size"]), (x, y - p["size"])],
                    fill=(59, 66, 82, 150),
                    width=3,
                )

    def _draw_sketch_eq(self, d, t01, phases):
        n = len(phases)
        gy = self.H - self.bottom_band + 40
        max_h = self.bottom_band * 0.75
        rng = random.Random(self._bg_seed + 77)
        for i, ph in enumerate(phases):
            beat = 0.5 + 0.5 * math.sin(t01 * math.tau * 2 + ph)
            h = 18 + (0.25 + 0.75 * beat) * max_h
            x = self.W * (i + 0.5) / n
            wob = 3 + (rng.random() - 0.5) * 2
            d.line(
                self._squiggle_pts(x, gy, x + wob, gy - h, rng, bump=4, steps=10),
                fill=(59, 66, 82, 200),
                width=3,
            )

    def _draw_shadow(self, d, feet, head_r, sketch=False):
        w = max(200.0, head_r * 3.2)
        h = w * 0.18
        color = (60, 60, 60, 70) if sketch else (0, 0, 0, 95)
        d.ellipse([feet[0] - w / 2, feet[1] - h / 2, feet[0] + w / 2, feet[1] + h / 2], fill=color)

    def _draw_figure(self, d, pts, head_c, head_r, t01, face_face):
        stick = (self.stick[0], self.stick[1], self.stick[2], 255)
        for limb in LIMBS:
            for i in range(len(limb) - 1):
                a, b = pts[limb[i]], pts[limb[i + 1]]
                if np.isfinite(a).all() and np.isfinite(b).all():
                    d.line([a[0], a[1], b[0], b[1]], fill=stick, width=self.line_w, joint="curve")
        joint_r = self.line_w // 2
        for idx in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
            p = pts[idx]
            if np.isfinite(p).all():
                d.ellipse([p[0] - joint_r, p[1] - joint_r, p[0] + joint_r, p[1] + joint_r], fill=stick)

        mid_sh = 0.5 * (pts[11] + pts[12])
        if np.isfinite(mid_sh).all():
            neck_w = max(4, self.line_w // 3)
            d.line([mid_sh[0], mid_sh[1], head_c[0], head_c[1]], fill=stick, width=neck_w)

        r = head_r
        d.ellipse(
            [head_c[0] - r, head_c[1] - r, head_c[0] + r, head_c[1] + r],
            fill=self.head,
            outline=INK,
            width=self.line_w // 3,
        )

        # normalized face metrics -> pixel params
        eye_l = min(1.0, max(0.0, float(face_face[0]) / 0.35))
        eye_r = min(1.0, max(0.0, float(face_face[1]) / 0.35))
        mouth_n = min(1.4, max(0.0, (float(face_face[2]) - 0.05) / 0.55))
        smile_px = max(-head_r, min(head_r, float(face_face[3]) * head_r * 6.0))

        face = self._face_layer(head_r, t01, eye_l, eye_r, mouth_n, smile_px)
        tilt = self._head_tilt(pts)
        return face.rotate(tilt, resample=Image.BICUBIC, expand=True)

    def _face_layer(self, head_r, t01, eye_l, eye_r, mouth_n, smile_px):
        pad = 12
        side = int(head_r * 2.4) + pad * 2
        face = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        fd = ImageDraw.Draw(face)
        c = side // 2
        r = head_r
        eye_r_val = r * 0.26
        eye_dx = int(r * 0.52)
        eye_dy = int(r * 0.20)

        fd.ellipse([eye_dx - eye_r_val, eye_dy - eye_r_val * 1.1, eye_dx + eye_r_val, eye_dy + eye_r_val * 1.1], outline=(0, 0, 0, 40))
        self._draw_eye(fd, c - eye_dx, c + eye_dy, eye_r_val, eye_l, t01, -1)
        self._draw_eye(fd, c + eye_dx, c + eye_dy, eye_r_val, eye_r, t01, 1)

        mx, my = c, c + int(r * 0.52)
        mw = int(r * 0.55)
        self._draw_mouth(fd, mx, my, mw, mouth_n, smile_px, r)

        blush_r = int(head_r * 0.12)
        for sgn in (-1, 1):
            fd.ellipse(
                [c + sgn * eye_dx - blush_r * 1.4, c + r * 0.78, c + sgn * eye_dx + blush_r * 1.4, c + r * 0.78 + blush_r * 0.9],
                fill=(255, 110, 120, 120),
            )
        return face

    def _draw_eye(self, fd, cx, cy, r, openness, t01, sgn):
        if openness < 0.30:
            # closed eye -> happy dome
            w = max(3, int(r * 0.45))
            fd.arc([cx - r, cy, cx + r, cy + 2 * r], 0, 180, fill=INK, width=w)
            return
        eh = r * (0.45 + 0.55 * openness)
        fd.ellipse([cx - r, cy - eh, cx + r, cy + eh], fill=WHITE, outline=INK, width=max(2, int(r * 0.12)))
        wob = math.sin(t01 * math.tau * 3 + sgn) * r * 0.3
        pr = r * (0.34 + 0.16 * openness)
        fd.ellipse([cx + wob - pr, cy - pr, cx + wob + pr, cy + pr], fill=INK)

    def _draw_mouth(self, fd, cx, cy, w, mouth_n, smile_px, r):
        lw = max(3, int(r * 0.1))
        if mouth_n <= 0.22:
            s = int(r * 0.5 + abs(smile_px) * 0.35)
            if smile_px > -4:
                fd.arc([cx - w, cy - s, cx + w, cy + s], 180, 360, fill=INK, width=lw)  # smile
            else:
                fd.arc([cx - w, cy - s, cx + w, cy + s], 0, 180, fill=INK, width=lw)  # frown
            return
        # speaking -> open mouth whose height follows the lip separation
        h = int(min(w * 1.35, w * (0.25 + 0.75 * min(mouth_n, 1.4))))
        interior = (122, 24, 40, 255)
        fd.ellipse([cx - w, cy, cx + w, cy + 2 * h], fill=interior)
        fd.arc([cx - w, cy - lw, cx + w, cy + lw * 2], 0, 180, fill=INK, width=lw)   # upper lip
        fd.arc([cx - w, cy + 2 * h - lw, cx + w, cy + 2 * h + lw], 180, 360, fill=INK, width=lw)  # lower lip
        if mouth_n > 0.6:
            tw = int(w * 0.6)
            th = int(h * 0.6)
            fd.ellipse([cx - tw, cy + 2 * h - th, cx + tw, cy + 2 * h], fill=(235, 120, 130, 235))

    def _head_tilt(self, pts):
        mid_sh = 0.5 * (pts[11] + pts[12])
        nose = pts[0]
        if not (np.isfinite(mid_sh).all() and np.isfinite(nose).all()):
            return 0.0
        return math.degrees(math.atan2(nose[0] - mid_sh[0], nose[1] - mid_sh[1]))

    def _draw_sparkles(self, d, coords, t, pts, fit, sketch=False):
        if t < 2:
            return
        speed_left = float(np.linalg.norm(coords[t, 15, :2] - coords[t - 1, 15, :2]))
        speed_right = float(np.linalg.norm(coords[t, 16, :2] - coords[t - 1, 16, :2]))
        color = (70, 70, 90, 210) if sketch else (255, 235, 150, 210)
        for i, speed in ((15, speed_left), (16, speed_right)):
            if speed * fit["scale"] > 0.028 and np.isfinite(pts[i]).all():
                p = pts[i]
                for k in range(3):
                    ang = k * math.tau / 3 + t * 0.45
                    dx, dy = math.cos(ang) * 26, math.sin(ang) * 26
                    d.polygon(
                        [
                            (p[0], p[1] - 30), (p[0] + 8, p[1] - 8), (p[0] + 30, p[1]),
                            (p[0] + 8, p[1] + 8), (p[0], p[1] + 30), (p[0] - 8, p[1] + 8),
                            (p[0] - 30, p[1]), (p[0] - 8, p[1] - 8),
                        ],
                        fill=color,
                    )

    def _draw_title(self, d, title_text, tag, font_big, font_small, sketch=False):
        text = (title_text or "DANCE MODE").upper()
        bb = d.textbbox((0, 0), text, font=font_big)
        th = bb[3] - bb[1]
        y0 = 26
        if sketch:
            d.rounded_rectangle(
                [40, y0 - 8, self.W - 40, y0 + th + 16],
                radius=22, fill=(255, 253, 245, 215), outline=(59, 66, 82, 255), width=3,
            )
            d.text((60, y0), text, font=font_big, fill=(48, 54, 68, 255))
            if tag:
                d.text((60, y0 + th + 22), tag.upper(), font=font_small, fill=(170, 120, 20, 255))
            return
        d.rounded_rectangle([40, y0 - 8, self.W - 40, y0 + th + 16], radius=22, fill=(0, 0, 0, 150))
        d.text((60, y0), text, font=font_big, fill=(250, 250, 250, 255), stroke_width=4, stroke_fill=(15, 15, 35, 255))
        if tag:
            tb = d.textbbox((0, 0), tag.upper(), font=font_small)
            d.text((60, y0 + th + 22), tag.upper(), font=font_small, fill=(253, 224, 71, 235))

    def _dist(self, a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))