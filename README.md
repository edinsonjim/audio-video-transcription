# transcriptor-srt

APM package (library) with a reusable skill for transcribing audio files to SRT
subtitles using faster-whisper. Designed for very large files (4–7 GB+): it
transcribes in memory-bounded chunks, supports resume after interruption, and
reports live progress to the agent via a JSON status file and `[PROGRESS]`
stdout markers.

## Contents

```
.apm/
└── skills/
    └── transcriptor-srt/
        ├── SKILL.md                    # trigger + workflow
        └── scripts/
            └── transcribe_srt.py       # CLI wrapper around whisper -> SRT
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

The skill runs the bundled script via `uv run --with faster-whisper --with numpy`,
so no permanent project dependency is required. On first use, faster-whisper
downloads the selected model (default `medium`, ~1.5 GB). `ffmpeg`/`ffprobe`
must be installed on `PATH`.

## Script

```
python transcribe_srt.py AUDIO [--language es] [--model medium] [--output-dir .]
                      [--chunk-secs 300] [--chunk-overlap 15] [--vad]
                      [--beam-size 5] [--word-timestamps]
                      [--status-json <file>] [--resume] [--force] [--quiet]
```

- `AUDIO` — path to the audio/video file to transcribe (required)
- `--language` — spoken language code (default: `es`); `auto` detects on the
  first chunk
- `--model` — whisper model size: tiny/base/small/medium/large/turbo (default:
  `medium`)
- `--output-dir` — directory for the generated `.srt` (default: current dir)
- `--chunk-secs` / `--chunk-overlap` — per-chunk window and overlap (default:
  `300`/`15`); keeps RAM bounded on multi-GB inputs
- `--vad` — silence filtering (default on)
- `--status-json` — live JSON status file the agent polls (default:
  `<output>.status.json`)
- `--resume` — reuse completed chunk checkpoints; `--force` ignores them

Output: an `.srt` file with the same basename as the input, written next to the
audio or into `--output-dir`. Progress markers on stdout: `[PROGRESS]`,
`[DONE] output=<...>`, `[ERROR] msg=<...>`.

## Development

Only the source under `.apm/` is versioned. Generated runtime dirs
(`.agents/`, `.claude/`, `.cursor/`, `.opencode/`, `.github/agents/`) are
gitignored and created on demand:

    apm install --target copilot,claude,cursor,opencode

Check for drift with `apm audit`.
