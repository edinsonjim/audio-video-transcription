# transcriptor-srt

APM package (library) with a reusable skill for transcribing audio files to SRT
subtitles using OpenAI Whisper.

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

The skill runs the bundled script via `uv run --with openai-whisper`, so no
permanent project dependency is required. On first use, whisper downloads the
selected model (default `medium`, ~1.5 GB).

## Script

```
python transcribe_srt.py AUDIO [--language es] [--model medium] [--output-dir .]
```

- `AUDIO` — path to the audio/video file to transcribe (required)
- `--language` — spoken language code (default: `es`)
- `--model` — whisper model size: tiny/base/small/medium/large (default: `medium`)
- `--output-dir` — directory for the generated `.srt` (default: current dir)

Output: an `.srt` file with the same basename as the input, written next to the
audio or into `--output-dir`.

## Development

Only the source under `.apm/` is versioned. Generated runtime dirs
(`.agents/`, `.claude/`, `.cursor/`, `.opencode/`, `.github/agents/`) are
gitignored and created on demand:

    apm install --target copilot,claude,cursor,opencode

Check for drift with `apm audit`.
