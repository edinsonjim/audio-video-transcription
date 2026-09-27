---
name: audio-video-transcription
description: >-
  Use this skill to transcribe audio or video into SRT, JSONL, or plain text.
  Activate when the user asks for a transcript, subtitles, or captions. Uses
  faster-whisper and supports very large files (multi-GB) with chunking, resume,
  and live progress through a JSON status file and [PROGRESS] stdout markers.
  Trigger keywords: transcribe, transcript, transcription, srt, jsonl, subtitles,
  captions, audio to text, subtitular, transcripción, subtítulos, whisper.
---

# Transcribe audio or video

Convert an audio or video file into SRT subtitles, JSONL segments, or plain text
using the bundled faster-whisper script. SRT remains the default output.

## When to use

Use this skill whenever the user wants subtitles, captions, or a timestamped
transcript from an audio/video file (e.g. `.mp3`, `.wav`, `.m4a`, `.mp4`,
`.mkv`). It handles very large files (4 GB, 7 GB+) by transcribing the audio in
  fixed-size chunks, and writes the final transcript a chunk at a time so
  assembly memory does not grow with the recording duration.

## Prerequisites

- `ffmpeg` / `ffprobe` must be on `PATH` (used to probe and extract audio).
- First run downloads the selected model to the Hugging Face cache and prints a
  warning about unauthenticated requests; warn the user it can take a while.

## Workflow

1. **Identify the input file.** The user usually provides a path or filename. If
   they only give a name, look for it in the working directory or ask for the
   path. Confirm the file exists (and that `ffmpeg`/`ffprobe` are installed)
   before running.

2. **Locate the bundled script.** This repository is a single-skill bundle:
   `SKILL.md` and `scripts/` are at its root. The script is
   `scripts/transcribe.py`. Resolve its absolute path — it must be executed
   directly, not reimplemented.

3. **Choose the output format.** Apply these rules in order:
   - Supported formats are only `srt`, `jsonl`, `txt`, and `all`. If the user
     requests another format (for example, VTT), explain that it is unsupported
     and offer one of the supported formats; do not pretend to generate it.
   - If the user explicitly requests a supported format, use it.
   - For subtitles/captions, use `srt`.
   - For searching, asking questions, or extracting facts with timestamp
     references, use `jsonl`.
   - For plain text without timestamps, use `txt`.
   - If the user requests every format, use `all`.
   - If no format is specified, use `srt`.

   `jsonl` is one JSON object per segment with absolute `start`/`end` times in
   seconds, text, chunk number, and transcript-order ID. For multi-hour recordings,
   retrieve/process relevant records in batches; do not send the entire transcript
   to a model at once. `txt` omits timestamps.

4. **Run the script.** Resolve the absolute skill directory, input file, and
   output directory first. Replace every example path below with the actual
   absolute path; do not execute it with the example values unchanged. Keep the
   defaults shown unless the user requests different settings or a resource
   constraint requires an adjustment.

   ```bash
   SKILL_DIR="/absolute/path/to/audio-video-transcription"
   INPUT_FILE="/absolute/path/to/input.mp4"
   OUTPUT_DIR="/absolute/path/to/transcripts"
   STATUS_JSON="$OUTPUT_DIR/$(basename "$INPUT_FILE").status.json"
   FORMAT="srt"
   LANGUAGE="es"
   MODEL="medium"

   mkdir -p "$OUTPUT_DIR"
   uv run --with faster-whisper --with numpy python "$SKILL_DIR/scripts/transcribe.py" \
     "$INPUT_FILE" --language "$LANGUAGE" --model "$MODEL" --format "$FORMAT" \
     --output-dir "$OUTPUT_DIR" --status-json "$STATUS_JSON" \
     --chunk-secs 300 --chunk-overlap 15 --vad --beam-size 5 \
     --device cpu --compute-type int8
   ```

   Change `FORMAT` only according to step 3. `LANGUAGE` defaults to `es`; use
   `auto` only when the spoken language is unknown. `MODEL` defaults to `medium`;
   smaller models favor speed, while `large`/`turbo` favor accuracy. For CUDA,
   use `--device cuda` only when a compatible NVIDIA GPU is available, and select
   a supported `--compute-type`; otherwise keep `cpu`/`int8`.

   **Other options:** keep `--beam-size 5` unless there is a specific tuning
   request. `--word-timestamps` makes the decoder calculate word timings, but the
   current exporters still write segment-level SRT/JSONL/TXT. Do not add this flag
   expecting word-by-word output; if the user requires that granularity, explain
   that it is not currently supported. The flag is slower and works with or
   without VAD. The command enables VAD by default; use `--no-vad` only if the
   user requests disabling it.
   Reduce `--chunk-secs` if memory is constrained; keep the 15-second overlap
   unless there is a reason to change it.

   **Large files:** audio and final output are processed chunk by chunk, so
   assembly memory does not grow with recording duration. The first model run may
   download about 1.5 GB for `medium`; long transcriptions can still take hours.

5. **Monitor progress while it runs.** The script writes both machine-readable
   markers to stdout (every ~5 s or on stage/percent change) and an atomic JSON
   status file. See the [Monitoring](#monitoring) section below for how to read
   them and detect stalls.

6. **Report the result.** On success the script prints
   `[DONE] output=<path>`. Confirm to the user the file was created, and mention
   the location, the transcription language/model used, and whether checkpoints
   were reused. If it ends with `[ERROR]`, relay the message; if interrupted,
   rerun the same command to continue from checkpoints.

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
    "output": "/abs/path/episode-03.srt",
    "outputs": ["/abs/path/episode-03.srt"]
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
  `state: interrupted`; completed chunks are checkpointed under
  `<basename>.srt.chunks/`. Rerun the same command to continue.

- **Checkpoint rule:** the script automatically loads existing checkpoints
  unless `--force` is passed; `--resume` is accepted but currently does not change
  that behavior. Reuse checkpoints only for the exact same input and transcription
  settings. If the input or settings changed, add `--force` to transcribe every
  chunk again.

- **Resume file semantics.** Checkpoint files (`chunk_%05d.json`) are tiny; they
  may be deleted to free disk, but then those chunks must be transcribed again.

## Output

- **Filename:** same basename as the input audio, with the selected extension
  (e.g. `episode-03.mp3` → `episode-03.srt`, `.jsonl`, or `.txt`). With
  `--format all`, all three files are generated.
- **Location:** the directory passed via `--output-dir` (default: current dir).
- **SRT:** standard numbered cues with `HH:MM:SS,mmm --> HH:MM:SS,mmm`
  timecodes.
- **JSONL:** one compact JSON object per segment, with `id`, 1-based `chunk`,
  absolute `start`/`end` times in seconds, and `text`.
- **TXT:** one transcribed segment per line, without timestamps.
- **Overlap handling:** duplicate segments from chunk overlaps are removed for
  all output formats.
- **Checkpoints:** `<basename>.srt.chunks/` holds one JSON file per audio chunk
  for resume. This legacy location is retained for every format, so existing
  checkpoints can be reused with `--format jsonl` or `txt`.
