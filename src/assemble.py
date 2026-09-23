import subprocess
from pathlib import Path

from .util import LOG, retry


def assemble(frame_dir: Path, video_id: str, source_video: Path, fps: float, out_dir: Path) -> Path:
    out = out_dir / f"{video_id}.mp4"
    pattern = (frame_dir / "%05d.jpg").as_posix()
    cmd = [
        "ffmpeg", "-y",
        "-framerate", f"{fps:.2f}",
        "-i", pattern,
        "-i", str(source_video),
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2",
        "-r", f"{fps:.2f}",
        "-shortest",
        "-movflags", "+faststart",
        str(out),
    ]
    LOG.info("Assembling video with ffmpeg: %s", out.name)

    def _run():
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{proc.stderr[-3000:]}")
        if not out.exists() or out.stat().st_size < 1000:
            raise RuntimeError("ffmpeg produced no output")

    retry(_run, attempts=2, label=f"assemble {video_id}")
    LOG.info("Assembled %s (%.1f MB)", out.name, out.stat().st_size / 1e6)
    return out