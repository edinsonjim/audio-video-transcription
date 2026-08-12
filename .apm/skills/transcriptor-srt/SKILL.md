---
name: transcriptor-srt
description: >-
  Transcribes audio or video files into SRT subtitle files using faster-whisper.
  Use whenever the user asks to transcribe, generate subtitles or captions, convert
  audio to SRT, or create a transcript with timestamps. Designed for very large
  files (multi-GB): processes in memory-bounded chunks, supports resume after
  interruption, and reports live progress via a JSON status file and [PROGRESS]
  stdout markers. Trigger keywords: transcribe, transcript, transcription, srt,
  subtitles, captions, audio to srt, subtitular, transcripción, subtítulos, whisper.
---

# Transcribe audio to SRT

Convert an audio or video file into an SRT subtitle file using the bundled
faster-whisper script.

## When to use

Use this skill whenever the user wants subtitles, captions, or a timestamped
transcript from an audio/video file (e.g. `.mp3`, `.wav`, `.m4a`, `.mp4`,
`.mkv`). It handles very large files (4 GB, 7 GB+) by transcribing the audio in
fixed-size chunks, so memory stays bounded regardless of the file size.

## Prerequisites

- `ffmpeg` / `ffprobe` must be on `PATH` (used to probe and extract audio).
- First run downloads the selected model to the Hugging Face cache and prints a
  warning about unauthenticated requests; warn the user it can take a while.

## Workflow

1. **Identify the input file.** The user usually provides a path or filename. If
   they only give a name, look for it in the working directory or ask for the
   path. Confirm the file exists (and that `ffmpeg`/`ffprobe` are installed)
   before running.

2. **Locate the bundled script.** It lives in the `scripts/` folder of this
   skill (`scripts/transcribe_srt.py`). Resolve its absolute path — it must be
   executed directly, not reimplemented.

3. **Run the script** with `uv run --with faster-whisper --with numpy` so
   dependencies are provisioned on demand without touching the project
   environment:

   ```bash
   uv run --with faster-whisper --with numpy python <path>/scripts/transcribe_srt.py <audio> \
     [--language es] [--model medium] [--output-dir <dir>] \
     --status-json <status.json>
   ```

   - `<audio>` — path to the audio/video file (required)
   - `--language` — spoken language code. Default `es`. Pass `auto` to auto-detect
     on the first chunk and lock it for the rest.
   - `--model` — whisper model size: `tiny`, `base`, `small`, `medium`, `large`
     (maps to `large-v3`), `turbo`. Default `medium` (~1.5 GB int8). Prefer a
     smaller model for short/fast jobs, `large`/`turbo` for accuracy.
   - `--output-dir` — directory for the generated `.srt`. Default is the current
     working directory.
   - `--status-json` — path for the live JSON status file. Default is
     `<output>.status.json`. Always pass an explicit path when the agent needs to
     poll progress.
   - `--chunk-secs` — seconds transcribed per chunk (default `300`). Reduce if
     RAM is tight. `--chunk-overlap` (default `15`) controls cross-chunk context.
   - `--vad` (default on, use `--no-vad` to disable) — skips silence, much faster
     on audio with pauses.
   - `--resume` — reuse completed chunk checkpoints after an interruption.
     `--force` re-transcribes everything.
   - `--quiet` — print only the machine markers `[PROGRESS]`/`[DONE]`/`[ERROR]`.

   **Important for large files:** the process works on GPU-less CPU with int8
   quantization by default (`--device cpu --compute-type int8`) — it is 3–4x
   faster and uses far less RAM than openai-whisper, but long videos still take
   significant wall-clock time. Run it and monitor, don't block.

4. **Monitor progress while it runs.** The script writes both machine-readable
   markers to stdout (every ~5 s or on stage/percent change) and an atomic JSON
   status file. See the [Monitoring](#monitoring) section below for how to read
   them and detect stalls.

5. **Report the result.** On success the script prints
   `[DONE] output=<path>`. Confirm to the user the file was created, and mention
   the location, the transcription language/model used, and any interruption that
   was resumed. If it ends with `[ERROR]`, relay the message; if it was
   interrupted, run again with `--resume` before giving up.

## Monitoring

While a run is active the agent should not just wait — it should track progress:

- **JSON status file** (`--status-json`, default `<output>.status.json`): read it
  periodically with your file reader. It is rewritten atomically, so it is always
  a complete document. Fields:

  ```json
  {
    "state": "running",
    "stage": "chunk",
    "percent": 42.5,
    "chunk_index": 3,
    "chunk_total": 24,
    "chunks_completed": 3,
    "language": "es",
    "elapsed_secs": 812.0,
    "eta_secs": 1100.0,
    "output": "/abs/path/episode-03.srt"
  }
  ```
  - `stage` values: `probing`, `model_load`, `chunk`, `merging`, `writing`,
    `done`, `error`, `interrupted`.
  - `state` is `running` until the run ends, then `done`/`error`/`interrupted`.

- **Stdout markers** (always flushed; with `--quiet` they're the only output):
  - `[PROGRESS] stage=chunk pct=42.5 chunk=3/24 eta=1100s elapsed=812s`
  - `[DONE] output=<path>`
  - `[ERROR] msg=<message>`
  - `[INFO] ...` human-readable narration (suppressed by `--quiet`).

- **Stall detection.** If `elapsed_secs` grows but `percent` stays flat for a long
  time, the model may be decoding a slow chunk — don't kill it immediately. Only
  after the window clearly exceeds normal chunk times consider interrupting.

- **Interruptions.** A killed process exits with code `130` and writes
  `state: interrupted`; previously completed chunks are safe on disk under
  `<output>.srt.chunks/`. Re-run the same command with `--resume` to continue
  without re-transcribing finished chunks.

- **Resume file semantics.** Checkpoint files (`chunk_%05d.json`) are tiny; they
  may be deleted to free disk, but then resume re-transcribes those chunks.

## Output

- **Filename:** same basename as the input audio, with a `.srt` extension
  (e.g. `episode-03.mp3` → `episode-03.srt`).
- **Location:** the directory passed via `--output-dir` (default: current dir).
- **Format:** standard SRT — numbered cues, `HH:MM:SS,mmm --> HH:MM:SS,mmm`
  timecodes, and the transcribed text for each segment. Chunk overlaps are
  deduplicated so cues never overlap.
- **Checkpoints:** `<output>.srt.chunks/` holds one JSON per chunk for resume.