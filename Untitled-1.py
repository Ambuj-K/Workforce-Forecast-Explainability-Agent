"""
Transcribe call/voice recordings (Hindi / Hinglish) into timestamped text files.

SETUP (one time)
  1. Install Python 3.10+ from python.org (Windows: tick "Add Python to PATH").
  2. Open Terminal (Mac) or Command Prompt (Windows) and run:
         pip install faster-whisper
     (Mac: use pip3 / python3 if "pip" / "python" is not found.)

CHOOSE THE FOLDER (either way works)
  A) Edit RECORDINGS_FOLDER below, e.g. "~/Downloads/recordings"
     ("~" means your home folder; works on Windows and Mac.)
  B) Or pass the folder when running:
         python transcribe_recordings.py "C:\\Users\\YourName\\Downloads\\recordings"
         python3 transcribe_recordings.py ~/Downloads/recordings

RUN
         python transcribe_recordings.py

  - First run downloads the speech model (~1.5 GB for "medium"), once.
  - Transcripts are written to a "transcripts" folder INSIDE the recordings
    folder: <name>_transcript.txt, one line per segment with [hh:mm:ss].
  - Files already transcribed are skipped, so you can stop and restart safely.
  - Too slow? Change MODEL_SIZE to "small". NVIDIA GPU? Set DEVICE = "cuda".

Your ORIGINAL recordings are only read, never modified.
Send the *_transcript.txt files back for analysis.
"""

import os
import sys
import time

from faster_whisper import WhisperModel

# ---- settings -------------------------------------------------------------
RECORDINGS_FOLDER = "~/Downloads/recordings"   # change to your folder
MODEL_SIZE = "medium"   # "small" = faster, "large-v3" = most accurate, slowest
DEVICE = "cpu"          # "cuda" if you have an NVIDIA GPU
LANGUAGE = "hi"         # Hindi; handles mixed Hindi-English reasonably well
EXTENSIONS = (".m4a", ".mp3", ".wav", ".aac", ".ogg", ".amr", ".3gp")
# ---------------------------------------------------------------------------


def fmt(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def main() -> None:
    folder = sys.argv[1] if len(sys.argv) > 1 else RECORDINGS_FOLDER
    folder = os.path.abspath(os.path.expanduser(folder.strip().strip('"')))

    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        print("Edit RECORDINGS_FOLDER in the script, or pass the folder path.")
        sys.exit(1)

    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(EXTENSIONS))
    if not files:
        print(f"No recordings found in {folder}")
        sys.exit(1)

    out_dir = os.path.join(folder, "transcripts")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Recordings folder: {folder}")
    print(f"Found {len(files)} file(s): {', '.join(files)}")
    print(f"Transcripts will be saved in: {out_dir}\n")

    print(f"Loading model '{MODEL_SIZE}' (first time downloads it)...")
    compute = "float16" if DEVICE == "cuda" else "int8"
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=compute)

    for name in files:
        src = os.path.join(folder, name)
        out = os.path.join(out_dir, os.path.splitext(name)[0] + "_transcript.txt")
        if os.path.exists(out):
            print(f"Skipping {name} (already done)")
            continue

        print(f"\nTranscribing {name} ...")
        start = time.time()
        segments, info = model.transcribe(
            src,
            language=LANGUAGE,
            vad_filter=True,   # skips long silences
            beam_size=5,
        )
        tmp = out + ".partial"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(f"# File: {name}\n# Duration: {fmt(info.duration)}\n\n")
            for seg in segments:
                line = f"[{fmt(seg.start)}] {seg.text.strip()}"
                fh.write(line + "\n")
                fh.flush()
                print(line)
        os.replace(tmp, out)
        print(f"Done {name} in {fmt(time.time() - start)}")

    print(f"\nAll done. Transcripts are in: {out_dir}")


if __name__ == "__main__":
    main()