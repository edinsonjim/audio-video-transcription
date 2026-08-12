#!/usr/bin/env python3
import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
MODEL_ALIASES = {"large": "large-v3"}
MIN_AUDIO_SECS = 0.1


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe an audio/video file to an SRT subtitle file using "
            "faster-whisper. Very large files (multi-GB) are processed in "
            "memory-bounded chunks with progress reporting and resume support."
        )
    )
    parser.add_argument("audio", help="Path to the audio/video file to transcribe")
    parser.add_argument(
        "--language",
        default="es",
        help="Spoken language code (e.g. es, en, fr). Use 'auto' for auto-detect on the first chunk. Default: es",
    )
    parser.add_argument(
        "--model",
        default="medium",
        help="Whisper model size: tiny, base, small, medium, large (large-v3), turbo. Default: medium",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where the generated .srt file will be written. Default: current directory",
    )
    parser.add_argument(
        "--chunk-secs",
        type=float,
        default=300.0,
        help="Seconds of audio transcribed per chunk. Bigger chunks use more RAM. Default: 300",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=float,
        default=15.0,
        help="Seconds of extra context decoded after each chunk boundary. Default: 15",
    )
    parser.add_argument(
        "--vad",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Filter silence before transcription (much faster on audio with pauses). Default: enabled",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding. Default: 5",
    )
    parser.add_argument(
        "--word-timestamps",
        action="store_true",
        help="Enable word-level timestamps (slower; implies VAD). Default: off",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device: cpu or cuda. Default: cpu",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="Quantization: int8, float32, float16, etc. Default: int8 on CPU, float16 otherwise",
    )
    parser.add_argument(
        "--status-json",
        default=None,
        help="Path for the live JSON status file the agent can poll. Default: <output>.status.json",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed chunk checkpoints so an interrupted run continues",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore existing checkpoints and re-transcribe every chunk",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print [PROGRESS]/[DONE]/[ERROR] machine markers to stdout",
    )
    return parser.parse_args()


class StatusReporter:
    def __init__(self, args):
        self.path = args.status_json
        self.quiet = args.quiet
        self.start_time = time.time()
        self.base = {
            "input": os.path.abspath(args.audio),
            "model": args.model,
            "device": args.device,
            "compute_type": args.compute_type or ("int8" if args.device == "cpu" else "float16"),
        }
        self.state = "running"
        self.stage = "init"
        self.percent = 0.0
        self.chunk_index = 0
        self.chunk_total = 0
        self.eta_secs = None
        self.language = None
        self.output = None
        self.chunks_completed = 0
        self._last_marker = 0.0

    def update(self, force=False):
        now = time.time()
        self._write_json(now)
        throttle_ok = (now - self._last_marker) >= 5.0
        if force or throttle_ok:
            self._print_marker(now)
            self._last_marker = now

    def _print_marker(self, now):
        chunk_str = f" chunk={self.chunk_index}/{self.chunk_total}" if self.chunk_total else ""
        eta_str = f" eta={int(self.eta_secs)}s" if self.eta_secs is not None else ""
        print(
            f"[PROGRESS] stage={self.stage} pct={self.percent:.1f}{chunk_str}"
            f"{eta_str} elapsed={int(now - self.start_time)}s",
            flush=True,
        )

    def _write_json(self, now):
        if self.path is None:
            return
        payload = dict(self.base)
        payload.update(
            {
                "state": self.state,
                "stage": self.stage,
                "percent": round(self.percent, 1),
                "chunk_index": self.chunk_index,
                "chunk_total": self.chunk_total,
                "chunks_completed": self.chunks_completed,
                "language": self.language,
                "elapsed_secs": round(now - self.start_time, 1),
                "eta_secs": None if self.eta_secs is None else round(self.eta_secs, 1),
            }
        )
        if self.output:
            payload["output"] = self.output
        tmp_path = self.path + f".{os.getpid()}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        os.replace(tmp_path, self.path)

    def info(self, msg):
        if not self.quiet:
            print(f"[INFO] {msg}", flush=True)

    def done(self, output):
        self.state = "done"
        self.stage = "done"
        self.percent = 100.0
        self.output = output
        self.update(force=True)
        print(f"[DONE] output={output}", flush=True)

    def error(self, msg):
        self.state = "error"
        self.stage = "error"
        self.update(force=True)
        print(f"[ERROR] msg={msg}", flush=True)

    def interrupted(self):
        self.state = "interrupted"
        self.stage = "interrupted"
        self.update(force=True)


def probe_duration(path):
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe not found in PATH (install ffmpeg)")
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {proc.stderr.strip()}")
    try:
        return float(proc.stdout.strip())
    except ValueError:
        raise RuntimeError(f"could not read duration for {path}")


def extract_chunk(path, start_s, dur_s):
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found in PATH (install ffmpeg)")
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{dur_s:.3f}",
        "-i",
        path,
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-f",
        "f32le",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed extracting audio: {proc.stderr.decode('utf-8', 'replace').strip()}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def fmt_srt_ts(seconds):
    total_ms = int(round(seconds * 1000))
    hours, rem = divmod(total_ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(segments, out_path):
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{fmt_srt_ts(seg['start'])} --> {fmt_srt_ts(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    tmp_fd, tmp_path = tempfile.mkstemp(
        prefix=os.path.basename(out_path) + ".", suffix=".tmp",
        dir=os.path.dirname(out_path) or ".",
    )
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        os.replace(tmp_path, out_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def chunk_dir_for(output_path):
    return output_path + ".chunks"


def chunk_file_for(chunk_dir, idx):
    return os.path.join(chunk_dir, f"chunk_{idx:05d}.json")


def load_checkpoint(chunk_dir, idx):
    path = chunk_file_for(chunk_dir, idx)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as fh:
                ckpt = json.load(fh)
            if isinstance(ckpt, list):
                return {"language": None, "segments": ckpt}
            return ckpt
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
    return None


def save_checkpoint(chunk_dir, idx, language, segments):
    os.makedirs(chunk_dir, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=chunk_dir, suffix=".tmp")
    with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
        json.dump({"version": 1, "language": language, "segments": segments}, fh, ensure_ascii=False)
    os.replace(tmp_path, chunk_file_for(chunk_dir, idx))


def clean_text(text):
    return " ".join(text.split())


def transcribe_chunk(model, pcm, language, args):
    segments_iter, info = model.transcribe(
        audio=pcm,
        language=language,
        task="transcribe",
        beam_size=args.beam_size,
        vad_filter=args.vad,
        word_timestamps=args.word_timestamps,
        log_progress=False,
    )
    segments = []
    for seg in segments_iter:
        text = clean_text(seg.text)
        if text:
            segments.append({"start": float(seg.start), "end": float(seg.end), "text": text})
    return segments, info


def run(args, reporter):
    if not os.path.exists(args.audio):
        raise FileNotFoundError(args.audio)

    reporter.info("Probing media duration ...")
    reporter.stage = "probing"
    reporter.update(force=False)
    duration = probe_duration(args.audio)
    if duration < MIN_AUDIO_SECS:
        raise RuntimeError(f"audio too short or unreadable: {duration:.3f}s")

    model_name = args.model
    if model_name in MODEL_ALIASES:
        model_name = MODEL_ALIASES[model_name]
    compute_type = args.compute_type or ("int8" if args.device == "cpu" else "float16")

    reporter.info(f"Loading model '{model_name}' ({args.device}/{compute_type}) ...")
    reporter.base["model"] = model_name
    reporter.base["device"] = args.device
    reporter.base["compute_type"] = compute_type
    reporter.stage = "model_load"
    reporter.update(force=True)
    model = WhisperModel(model_name, device=args.device, compute_type=compute_type)

    language = None if args.language == "auto" else args.language

    chunk_secs = max(1.0, args.chunk_secs)
    overlap = max(0.0, args.chunk_overlap)
    total_chunks = math.ceil(duration / chunk_secs)
    reporter.language = language
    reporter.chunk_total = total_chunks

    out_base = os.path.splitext(os.path.basename(args.audio))[0]
    output_path = os.path.join(args.output_dir, out_base + ".srt")
    chunk_dir = chunk_dir_for(output_path)
    if args.status_json is None:
        reporter.path = output_path + ".status.json"

    merged = []
    covered = 0.0
    chunk_times = []
    reporter.stage = "chunk"
    reporter.info(f"Starting {total_chunks} chunk(s) of {chunk_secs:.0f}s ...")

    for idx in range(total_chunks):
        reporter.chunk_index = idx + 1
        start = idx * chunk_secs
        window_dur = min(chunk_secs + overlap, duration - start)

        checkpoint = None if args.force else load_checkpoint(chunk_dir, idx)
        start_ts = time.time()
        if checkpoint is None:
            pcm = extract_chunk(args.audio, start, window_dur)
            if pcm.size == 0:
                segments, info = [], None
            else:
                segments, info = transcribe_chunk(model, pcm, language, args)
                if language is None and info is not None:
                    language = info.language
                    reporter.language = language
                    reporter.info(f"Auto-detected language: {language}")
            checkpoint = segments
            save_checkpoint(chunk_dir, idx, language, segments)
            reporter.info(f"Chunk {idx + 1}/{total_chunks} transcribed ({len(segments)} segments)")
        else:
            segments = checkpoint["segments"]
            if language is None and checkpoint.get("language"):
                language = checkpoint["language"]
                reporter.language = language
            reporter.info(f"Chunk {idx + 1}/{total_chunks} restored from checkpoint")
        chunk_times.append(time.time() - start_ts)

        for seg in segments:
            g_start = start + seg["start"]
            g_end = start + seg["end"]
            if g_end <= covered:
                continue
            merged.append({"start": max(g_start, covered), "end": g_end, "text": seg["text"]})
            if g_end > covered:
                covered = g_end

        reporter.chunks_completed = idx + 1
        reporter.percent = min(start + chunk_secs, duration) / duration * 100.0
        if len(chunk_times) >= 2:
            avg = sum(chunk_times) / len(chunk_times)
            reporter.eta_secs = avg * (total_chunks - reporter.chunks_completed)
        reporter.update(force=True)

    reporter.percent = 100.0
    reporter.stage = "merging"
    reporter.update(force=True)

    os.makedirs(args.output_dir, exist_ok=True)
    reporter.stage = "writing"
    reporter.update(force=True)
    write_srt(merged, output_path)

    reporter.language = language
    reporter.done(os.path.abspath(output_path))
    return 0


def main(argv=None):
    def _handle_signal(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    args = parse_args()
    reporter = StatusReporter(args)
    try:
        return run(args, reporter)
    except KeyboardInterrupt:
        reporter.interrupted()
        print("[ERROR] msg=interrupted by signal; rerun with --resume to continue", flush=True)
        return 130
    except Exception as exc:
        reporter.error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())