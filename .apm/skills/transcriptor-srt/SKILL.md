---
name: transcriptor-srt
description: >-
  Transcribes audio or video files into SRT subtitle files using OpenAI Whisper.
  Use whenever the user asks to transcribe, generate subtitles or captions, convert
  audio to SRT, or create a transcript with timestamps. Trigger keywords: transcribe,
  transcript, transcription, srt, subtitles, captions, audio to srt, subtitular,
  transcripción, subtítulos, whisper.
---

# Transcribe audio to SRT

Convert an audio or video file into an SRT subtitle file using the bundled
Whisper-based script.

## When to use

Use this skill whenever the user wants subtitles, captions, or a timestamped
transcript from an audio/video file (e.g. `.mp3`, `.wav`, `.m4a`, `.mp4`,
`.mkv`).

## Workflow

1. **Identify the input file.** The user usually provides a path or filename. If
   they only give a name, look for it in the working directory or ask for the
   path. Confirm the file exists before running.

2. **Locate the bundled script.** The script lives in the `scripts/` folder of
   this skill (`scripts/transcribe_srt.py`). Resolve its absolute path — it must
   be executed directly, not reimplemented.

3. **Run the script** with `uv run --with openai-whisper` so the dependency is
   provisioned on demand without touching the project environment:

   ```bash
   uv run --with openai-whisper python <path>/scripts/transcribe_srt.py <audio> \
     [--language es] [--model medium] [--output-dir <dir>]
   ```

   - `<audio>` — path to the audio/video file (required)
   - `--language` — spoken language code. Default `es`. Omit for auto-detect by
     passing `--language auto` (the script maps it to `None`).
   - `--model` — whisper model size: `tiny`, `base`, `small`, `medium`, `large`.
     Default `medium`. Prefer a smaller model for short/fast jobs, `large` for
     accuracy.
   - `--output-dir` — directory for the generated `.srt`. Default is the current
     working directory.

   The first run downloads the requested whisper model (default `medium`,
   ~1.5 GB); warn the user this can take a while.

4. **Report the result.** The script prints the absolute path of the generated
   `.srt` file. Confirm to the user that the file was created, and mention the
   location and the transcription language/model used. If transcription fails
   (e.g. missing input file, unsupported format), relay the error clearly.

## Output

- **Filename:** same basename as the input audio, with a `.srt` extension
  (e.g. `episode-03.mp3` → `episode-03.srt`).
- **Location:** same folder as the audio, or the directory passed via
  `--output-dir`.
- **Format:** standard SRT — numbered cues, `HH:MM:SS,mmm --> HH:MM:SS,mmm`
  timecodes, and the transcribed text for each segment.
