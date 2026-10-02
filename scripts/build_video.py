"""Narrated draft video from the slide PNGs and narration (macOS `say` + ffmpeg).

    python scripts/build_video.py   -> writeup/video/ChipStain_video.mp4 (must stay under 5:00)
"""
import glob
import json
import os
import subprocess

OUT = "writeup/video"


def duration(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], capture_output=True, text=True)
    return float(r.stdout.strip())


def main():
    tpl = json.load(open("writeup/slides_template.json"))
    voice, rate = tpl.get("voice", "Samantha"), str(tpl.get("rate", 172))
    os.makedirs(OUT, exist_ok=True)
    parts = []
    for png in sorted(glob.glob("writeup/slides/slide_*.png")):
        stem = png[:-4]
        txt = open(stem + ".txt", encoding="utf-8").read()
        aiff, wav, mp4 = f"{OUT}/{os.path.basename(stem)}.aiff", f"{OUT}/{os.path.basename(stem)}.wav", f"{OUT}/{os.path.basename(stem)}.mp4"
        subprocess.run(["say", "-v", voice, "-r", rate, "-o", aiff, txt], check=True)
        # 0.5 s silence before, 0.7 s after each slide's narration
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", aiff, "-af", "adelay=500|500,apad=pad_dur=0.7", "-ar", "48000", "-ac", "2", wav], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", png, "-i", wav, "-c:v", "libx264", "-tune", "stillimage", "-pix_fmt", "yuv420p",
                        "-r", "30", "-c:a", "aac", "-b:a", "160k", "-shortest", mp4], check=True)
        parts.append(mp4)
    lst = f"{OUT}/parts.txt"
    open(lst, "w").write("".join(f"file '{os.path.abspath(p)}'\n" for p in parts))
    final = f"{OUT}/ChipStain_video.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", final], check=True)
    for p in glob.glob(f"{OUT}/slide_*"):
        os.remove(p)
    os.remove(lst)
    d = duration(final)
    print(f"video: {final}  duration {int(d // 60)}:{int(d % 60):02d}" + ("  (OVER 5:00!)" if d > 300 else "  (under the 5:00 limit)"))


if __name__ == "__main__":
    main()
