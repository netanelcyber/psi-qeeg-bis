"""Reproducible video/EEG replay: synthetic EEG + scripted avatar, not a patient.

Requires FFmpeg on PATH in addition to the existing project dependencies.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from matplotlib import font_manager

from demo_fft_tracking import create_fixture
from psi_qeeg.recording import read_recording
from psi_qeeg.video_eeg import write_video_eeg_report


def make_video(path, seconds=180, fps=8, rate=1.0001):
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise ValueError("Install FFmpeg (ffmpeg and ffprobe) to generate the video demonstration")
    font_path = font_manager.findfont("DejaVu Sans")
    title, small = ImageFont.truetype(font_path, 21), ImageFont.truetype(font_path, 13)
    command = [executable, "-y", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
               "-video_size", "640x360", "-framerate", str(fps), "-i", "pipe:0", "-an",
               "-c:v", "libx264", "-preset", "fast", "-crf", "25", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(path)]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
        try:
            for i in range(round(seconds*fps)):
                video_t, eeg_t = i/fps, rate*i/fps
                moving = 120 <= eeg_t < 124
                sway = 20*np.sin(2*np.pi*2*eeg_t) if moving else 1.5*np.sin(2*np.pi*.25*eeg_t)
                image = Image.new("RGB", (640, 360), "#0c1626")
                draw = ImageDraw.Draw(image)
                draw.text((22, 16), "SYNTHETIC PARTICIPANT / EEG DEMO", font=title, fill="#e7effa")
                draw.text((22, 46), "Scripted avatar. No patient video or clinical labels.", font=small, fill="#96abc5")
                draw.rounded_rectangle((190, 76, 450, 300), radius=18, fill="#14243b", outline="#29425f")
                head_x, head_y = 320+sway, 132
                draw.line((head_x, 168, 320, 236), fill="#a5b9d5", width=10)
                draw.line((320, 226, 280, 275), fill="#819bbc", width=9)
                draw.line((320, 226, 360, 275), fill="#819bbc", width=9)
                arm_y = 194 + (35*np.sin(2*np.pi*3*eeg_t) if moving else 12)
                draw.line((head_x, 181, 270, arm_y), fill="#819bbc", width=9)
                draw.line((head_x, 181, 370, arm_y), fill="#819bbc", width=9)
                draw.ellipse((head_x-31, head_y-34, head_x+31, head_y+34), fill="#c1d0e5")
                draw.arc((head_x-28, head_y-26, head_x+28, head_y+22), 190, 350, fill="#5de0cf", width=4)
                for dx in (-19, 19):
                    draw.ellipse((head_x+dx-3, head_y-24, head_x+dx+3, head_y-18), fill="#103b3b")
                phase = "Declared baseline" if eeg_t < 60 else "Scripted alpha / beta modulation" if eeg_t >= 100 else "Reference interval"
                if moving:
                    phase = "Scripted movement + EEG contamination"
                draw.text((22, 309), phase, font=small, fill="#ffc870" if moving else "#5de0cf")
                draw.text((22, 334), f"Video {video_t:07.3f} s  |  EEG clock {100+eeg_t:07.3f} s", font=small, fill="#96abc5")
                process.stdin.write(image.tobytes())
            process.stdin.close()
            if process.wait():
                raise ValueError("FFmpeg could not encode the synthetic video")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("outputs/video_eeg_demo"))
    parser.add_argument("--bootstrap-draws", type=int, default=1024)
    args = parser.parse_args()
    frame, metadata, model = create_fixture(args.out)
    relative = frame.time_s.to_numpy().copy()
    frame["time_s"] += 100.0
    moving = (relative >= 120) & (relative < 124)
    frame.loc[moving, "F3"] += 60*np.sin(2*np.pi*35*relative[moving])
    frame.loc[(relative >= 145) & (relative < 145.125), "C3"] = np.nan
    frame.to_csv(args.out/"recording.csv", index=False)
    metadata["source"] += "; native clock offset 100 s; scripted contamination at 220-224 s and C3 gap at 245 s"
    (args.out/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    sync = {"subject_id": metadata["subject_id"], "video_kind": "synthetic",
            "source": "Generated avatar and EEG fixture share a scripted timeline; not recorded from a person",
            "license": "CC0-1.0 (synthetic video)", "evidence": "Generator mapping: EEG = 100 + 1.0001 * video_s",
            "anchors": [{"video_s": 0, "eeg_s": 100}, {"video_s": 170, "eeg_s": 270.017}],
            "uncertainty_ms": 1.0, "motion_roi": [.29, .2, .42, .64]}
    (args.out/"sync.json").write_text(json.dumps(sync, indent=2)+"\n", encoding="utf-8")
    pd.DataFrame([
        {"start_s":100,"end_s":160,"clock":"eeg","label":"Declared calibration","source":"synthetic_script"},
        {"start_s":200,"end_s":275,"clock":"eeg","label":"Scripted spectral modulation","source":"synthetic_script"},
        {"start_s":220,"end_s":224,"clock":"eeg","label":"Scripted movement / HF artifact","source":"synthetic_script"},
        {"start_s":245,"end_s":245.125,"clock":"eeg","label":"Scripted missing EEG","source":"synthetic_script"},
    ]).to_csv(args.out/"events.csv", index=False)
    make_video(args.out/"avatar.mp4")
    recording = read_recording(args.out/"recording.csv", args.out/"metadata.json")
    summary = write_video_eeg_report(recording, model, 60, args.out/"avatar.mp4", sync, args.out,
                                     events=args.out/"events.csv", embed_video=True, motion_fps=4,
                                     draws=args.bootstrap_draws, seed=5,
                                     frontal_pair=("F3", "F4"), high_z_limit=6)
    print(json.dumps({"use": summary["use"], "report": str(args.out/"report.html"),
                      "windows": summary["monitoring_windows"], "updates": summary["measurement_updates"],
                      "clock": summary["video_eeg"]["synchronization"]}))


if __name__ == "__main__":
    main()
