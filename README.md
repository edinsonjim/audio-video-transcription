# transcriptor-srt

A single-skill APM bundle for transcribing audio and video with faster-whisper.
It exports SRT subtitles by default, with optional JSONL and plain-text formats.
Designed for very large files (4–7 GB+): it transcribes in chunks, supports
resume after interruption, streams transcript assembly, and reports live
progress to the agent via a JSON status file and `[PROGRESS]` stdout markers.

## Contents

```
├── SKILL.md                    # skill instructions and activation metadata
├── scripts/
│   └── transcribe_srt.py       # CLI wrapper around faster-whisper
├── apm.yml                     # APM metadata and target configuration
└── README.md
```

## Usage

Install this package into any project (adds the skill to the runtime dirs):

    apm install <owner>/transcriptor-srt-skill

Or add it to the consumer's `apm.yml`:

```yaml
dependencies:
  apm:
    - <owner>/transcriptor-srt-skill
```

Then ask your agent to transcribe an audio file, e.g.:

    Transcribe episode-03.mp3 to SRT subtitles

    Transcribe episode-03.mp3 to JSONL for searching the transcript

The skill runs the bundled script via `uv run --with faster-whisper --with numpy`,
so no permanent project dependency is required. On first use, faster-whisper
downloads the selected model (default `medium`, ~1.5 GB). `ffmpeg`/`ffprobe`
must be installed on `PATH`.

## Script

```
python scripts/transcribe_srt.py AUDIO [--language es] [--model medium] [--format srt]
                      [--output-dir .]
                      [--chunk-secs 300] [--chunk-overlap 15] [--vad]
                      [--beam-size 5] [--word-timestamps]
                      [--status-json <file>] [--resume] [--force] [--quiet]
```

- `AUDIO` — path to the audio/video file to transcribe (required)
- `--language` — spoken language code (default: `es`); `auto` detects on the
  first chunk
- `--model` — whisper model size: tiny/base/small/medium/large/turbo (default:
  `medium`)
- `--format` — `srt` (default), `jsonl`, `txt`, or `all` (writes all three)
- `--output-dir` — directory for generated transcript file(s) (default: current dir)
- `--chunk-secs` / `--chunk-overlap` — per-chunk window and overlap (default:
  `300`/`15`); keeps RAM bounded on multi-GB inputs
- `--vad` — silence filtering (default on)
- `--status-json` — live JSON status file the agent polls (default:
  `<output>.status.json`)
- `--resume` — reuse completed chunk checkpoints; `--force` ignores them

Output: the selected `.srt`, `.jsonl`, or `.txt` transcript, with the same
basename as the input. JSONL contains one segment per line with absolute start/end
times, text, and chunk number; it is intended for batch searching and retrieval
from long recordings. `--format all` writes every format. Checkpoints remain
under `<basename>.srt.chunks/` for resume compatibility. Progress markers on
stdout: `[PROGRESS]`, `[DONE] output=<...>`, `[ERROR] msg=<...>`.

## Development

The root `SKILL.md` and `scripts/` directory are the single-skill source bundle.
Generated runtime dirs (`.agents/`, `.claude/`, `.cursor/`, `.opencode/`,
`.github/agents/`) are gitignored and created on demand:

    apm install --target copilot,claude,cursor,opencode

Check for drift with `apm audit`.
