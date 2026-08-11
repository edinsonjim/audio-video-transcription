import argparse
import os

import whisper
from whisper.utils import get_writer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe an audio/video file to an SRT subtitle file using OpenAI Whisper."
    )
    parser.add_argument("audio", help="Path to the audio/video file to transcribe")
    parser.add_argument(
        "--language",
        default="es",
        help="Spoken language code (e.g. es, en, fr). Use 'auto' for auto-detect. Default: es",
    )
    parser.add_argument(
        "--model",
        default="medium",
        help="Whisper model size: tiny, base, small, medium, large. Default: medium",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where the generated .srt file will be written. Default: current directory",
    )
    return parser.parse_args()


def generate_srt(args):
    if not os.path.exists(args.audio):
        print(f"Error: input file not found: {args.audio}")
        return 1

    language = None if args.language == "auto" else args.language

    print("--- Loading Model... ---")
    model = whisper.load_model(args.model)

    print(f"--- Transcribing to SRT: {args.audio} ---")
    result = model.transcribe(args.audio, language=language, fp16=False)

    output_directory = args.output_dir
    os.makedirs(output_directory, exist_ok=True)
    srt_writer = get_writer("srt", output_directory)

    srt_writer(result, args.audio)

    base = os.path.splitext(os.path.basename(args.audio))[0]
    output_file = os.path.join(output_directory, base + ".srt")
    print(f"--- Success! Subtitle file generated: {os.path.abspath(output_file)} ---")
    return 0


if __name__ == "__main__":
    sys_exit_code = generate_srt(parse_args())
    raise SystemExit(sys_exit_code)
