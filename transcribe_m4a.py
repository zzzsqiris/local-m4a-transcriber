#!/usr/bin/env python3
"""Offline Chinese/English M4A transcription with speaker diarization."""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import warnings
from pathlib import Path
from typing import Any, Iterable

# Keep third-party caches inside the project/user-selected cache instead of
# writing to a possibly read-only home configuration directory.
os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib")
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在本机将中英混合 M4A 转成带讲话人和时间戳的 TXT。"
    )
    parser.add_argument("input", type=Path, help="输入 .m4a 文件")
    parser.add_argument("-o", "--output", type=Path, help="输出 TXT 路径")
    parser.add_argument(
        "--model",
        default="small",
        help="Whisper 模型（默认 small；更准确可用 medium 或 large-v3）",
    )
    parser.add_argument("--speakers", type=int, help="已知的准确讲话人数")
    parser.add_argument("--min-speakers", type=int, help="最少讲话人数")
    parser.add_argument("--max-speakers", type=int, help="最多讲话人数")
    parser.add_argument("--batch-size", type=int, default=4, help="识别批量大小（默认 4）")
    parser.add_argument("--cache-dir", type=Path, help="模型缓存目录")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="仅使用已经下载的模型，不连接模型仓库",
    )
    parser.add_argument(
        "--no-timestamps", action="store_true", help="TXT 中不显示时间戳"
    )
    return parser.parse_args()


def _value(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def format_time(seconds: float) -> str:
    total_ms = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    return f"{minutes:02d}:{secs:02d}.{millis:03d}"


def label_speakers(segments: Iterable[Any]) -> list[dict[str, Any]]:
    """Rename arbitrary model speaker IDs by order of first appearance."""
    labels: dict[str, str] = {}
    normalized: list[dict[str, Any]] = []
    for segment in segments:
        raw = str(_value(segment, "speaker", "UNKNOWN"))
        if raw not in labels:
            labels[raw] = f"讲话人{len(labels) + 1}"
        normalized.append(
            {
                "speaker": labels[raw],
                "start": float(_value(segment, "start", 0.0)),
                "end": float(_value(segment, "end", 0.0)),
                "text": str(_value(segment, "text", "")).strip(),
            }
        )
    return normalized


def merge_segments(segments: Iterable[Any]) -> list[dict[str, Any]]:
    """Join adjacent segments from the same speaker for a cleaner transcript."""
    merged: list[dict[str, Any]] = []
    for current in label_speakers(segments):
        if not current["text"]:
            continue
        if merged and merged[-1]["speaker"] == current["speaker"]:
            merged[-1]["end"] = current["end"]
            merged[-1]["text"] += " " + current["text"]
        else:
            merged.append(current)
    return merged


def render_txt(segments: Iterable[Any], timestamps: bool = True) -> str:
    lines: list[str] = []
    for segment in merge_segments(segments):
        if timestamps:
            span = f"[{format_time(segment['start'])} - {format_time(segment['end'])}]"
            lines.append(f"{span} {segment['speaker']}: {segment['text']}")
        else:
            lines.append(f"{segment['speaker']}: {segment['text']}")
    return "\n\n".join(lines) + "\n"


def validate_environment(source: Path, args: argparse.Namespace) -> None:
    if not source.is_file():
        raise RuntimeError(f"找不到输入文件：{source}")
    if source.suffix.lower() != ".m4a":
        raise RuntimeError("输入文件必须是 .m4a 格式。")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("未安装 ffmpeg。macOS 请运行：brew install ffmpeg")
    if args.speakers is not None and (args.min_speakers or args.max_speakers):
        raise RuntimeError("--speakers 不能与 --min-speakers/--max-speakers 同时使用。")
    for name in ("speakers", "min_speakers", "max_speakers"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise RuntimeError("讲话人数必须大于 0。")
    if args.batch_size < 1:
        raise RuntimeError("--batch-size 必须大于 0。")


def get_hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")


def transcribe_local(source: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    # Homebrew currently provides FFmpeg 8 while TorchCodec 0.7 probes 4-7.
    # We pass decoded audio arrays to pyannote, so its decoder is intentionally unused.
    warnings.filterwarnings(
        "ignore",
        message=".*torchcodec is not installed correctly.*",
        category=UserWarning,
        module=r"pyannote\.audio\.core\.io",
    )
    try:
        import torch
        import whisperx
        from whisperx.diarize import DiarizationPipeline
    except ImportError as exc:
        raise RuntimeError("缺少本地模型依赖，请先运行：pip install -r requirements.txt") from exc

    token = get_hf_token()
    if not token and not args.offline:
        raise RuntimeError(
            "首次下载说话人模型需要 HF_TOKEN。下载完成后可用 --offline 断网运行。"
        )

    # This tool intentionally uses local inference only and disables pyannote metrics.
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    cache_dir = str(args.cache_dir.expanduser().resolve()) if args.cache_dir else None
    device = "cpu"  # WhisperX documents CPU mode for macOS.

    print(f"1/3 加载本地 Whisper 模型：{args.model}")
    model = whisperx.load_model(
        args.model,
        device,
        compute_type="int8",
        download_root=cache_dir,
    )
    print("2/3 正在本地识别中英混合语音……")
    audio = whisperx.load_audio(str(source))
    result = model.transcribe(audio, batch_size=args.batch_size)
    del model
    gc.collect()

    print("3/3 正在本地区分讲话人……")
    diarizer = DiarizationPipeline(
        token=token,
        device=device,
        cache_dir=cache_dir,
    )
    diarization = diarizer(
        audio,
        num_speakers=args.speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )
    assigned = whisperx.diarize.assign_word_speakers(diarization, result)
    del diarizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return list(assigned.get("segments", []))


def main() -> int:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output = (args.output or source.with_suffix(".txt")).expanduser().resolve()

    try:
        validate_environment(source, args)
        output.parent.mkdir(parents=True, exist_ok=True)
        segments = transcribe_local(source, args)
        if not segments:
            raise RuntimeError("模型没有返回可用的语音片段。")
        output.write_text(
            render_txt(segments, timestamps=not args.no_timestamps), encoding="utf-8"
        )
        print(f"完成：{output}")
        return 0
    except (OSError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"处理失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
