#!/usr/bin/env python
"""Compare normalizer stats from up to 3 sources.

Sources are auto-detected:
  - Path ending in .txt              → runtime log (Python repr from lerobot_policy.py)
  - Existing local directory         → checkpoint directory (safetensors files)
  - HuggingFace "owner/repo" id      → dataset (meta/stats.json) or checkpoint

Usage:
    python src/lerobot/scripts/clare/compare_normalizer_stats.py \\
        <source_a> <source_b> [source_c] \\
        [--label-a LABEL] [--label-b LABEL] [--label-c LABEL] \\
        [--type-a TYPE] [--type-b TYPE] [--type-c TYPE] \\
        [--proc {pre,post,both}]

Source types (for --type-* override):
    txt          Runtime log file with Python repr of norm stats dict
    checkpoint   Local dir or HF Hub repo with normalizer safetensors
    dataset      HF Hub dataset repo with meta/stats.json

Examples:
    python src/lerobot/scripts/clare/compare_normalizer_stats.py \\
        outputs/lora_runtime_norm.txt \\
        outputs/checkpoints/continuallearning/dit_lora_seed1000 \\
        continuallearning/real_0_put_bowl_filtered \\
        --label-a runtime_txt --label-b ckpt --label-c dataset
"""
import argparse
import ast
import json
import re
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


MATCH_THRESHOLD = 1e-6


# ── Source type detection ──────────────────────────────────────────────────────

def detect_source_type(path_or_id: str, type_hint: str | None = None) -> str:
    """Return one of 'txt', 'checkpoint', 'dataset'."""
    if type_hint is not None:
        return type_hint
    p = Path(path_or_id)
    if p.suffix == ".txt" and p.exists():
        return "txt"
    if p.exists() and p.is_dir():
        return "checkpoint"
    # HuggingFace Hub id: exactly one slash, no local existence
    parts = path_or_id.split("/")
    if len(parts) == 2 and not path_or_id.startswith("."):
        return _probe_hub_type(path_or_id)
    print(
        f"  WARNING: cannot auto-detect type for {path_or_id!r},"
        " defaulting to 'checkpoint'",
        file=sys.stderr,
    )
    return "checkpoint"


def _probe_hub_type(repo_id: str) -> str:
    """Probe Hub to decide 'dataset' vs 'checkpoint'."""
    try:
        from huggingface_hub import list_repo_files
        for fname in list_repo_files(repo_id, repo_type="dataset"):
            if fname == "meta/stats.json":
                return "dataset"
        return "dataset"
    except Exception:
        pass
    try:
        from huggingface_hub import list_repo_files
        for fname in list_repo_files(repo_id, repo_type="model"):
            if "normalizer_processor" in fname and fname.endswith(".safetensors"):
                return "checkpoint"
    except Exception:
        pass
    return "checkpoint"


# ── Loaders ────────────────────────────────────────────────────────────────────

def _is_hub_id(path_or_id: str) -> bool:
    p = Path(path_or_id)
    if p.exists():
        return False
    parts = path_or_id.split("/")
    return len(parts) == 2 and not path_or_id.startswith(".")


def _find_normalizer_file_local(
    pretrained_dir: Path,
) -> tuple[Path | None, Path | None]:
    pre = list(
        pretrained_dir.glob(
            "policy_preprocessor_step_*_normalizer_processor.safetensors"
        )
    )
    post = list(
        pretrained_dir.glob(
            "policy_postprocessor_step_*_unnormalizer_processor.safetensors"
        )
    )
    return (pre[0] if pre else None, post[0] if post else None)


def _find_normalizer_file_hub(repo_id: str) -> tuple[str | None, str | None]:
    from huggingface_hub import list_repo_files
    pre_pat = re.compile(
        r"policy_preprocessor_step_\d+_normalizer_processor\.safetensors"
    )
    post_pat = re.compile(
        r"policy_postprocessor_step_\d+_unnormalizer_processor\.safetensors"
    )
    pre_file = post_file = None
    for fname in list_repo_files(repo_id):
        if pre_pat.match(fname):
            pre_file = fname
        if post_pat.match(fname):
            post_file = fname
    return pre_file, post_file


def load_normalizer_state(
    path_or_id: str, label: str
) -> dict[str, dict[str, torch.Tensor]]:
    """Load pre/post normalizer safetensors from local dir or HF Hub."""
    result: dict[str, dict[str, torch.Tensor]] = {"pre": {}, "post": {}}

    if _is_hub_id(path_or_id):
        from huggingface_hub import hf_hub_download
        pre_fname, post_fname = _find_normalizer_file_hub(path_or_id)
        if pre_fname is None:
            print(
                f"  [{label}] WARNING: no normalizer_processor file"
                f" in Hub repo {path_or_id!r}"
            )
        else:
            local = hf_hub_download(repo_id=path_or_id, filename=pre_fname)
            result["pre"] = load_file(local)
            print(
                f"  [{label}] loaded pre-normalizer from Hub:"
                f" {path_or_id}/{pre_fname}"
            )
        if post_fname is not None:
            local = hf_hub_download(repo_id=path_or_id, filename=post_fname)
            result["post"] = load_file(local)
            print(
                f"  [{label}] loaded post-normalizer from Hub:"
                f" {path_or_id}/{post_fname}"
            )
    else:
        pretrained_dir = Path(path_or_id)
        if not pretrained_dir.exists():
            print(
                f"  [{label}] ERROR: path does not exist: {pretrained_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        pre_path, post_path = _find_normalizer_file_local(pretrained_dir)
        if pre_path is None:
            print(
                f"  [{label}] WARNING: no normalizer_processor file"
                f" in {pretrained_dir}"
            )
        else:
            result["pre"] = load_file(str(pre_path))
            print(f"  [{label}] loaded pre-normalizer: {pre_path}")
        if post_path is not None:
            result["post"] = load_file(str(post_path))
            print(f"  [{label}] loaded post-normalizer: {post_path}")

    return result


def load_dataset_stats(dataset_repo_id: str) -> dict[str, torch.Tensor]:
    """Load meta/stats.json from HF dataset repo; flatten to key→tensor."""
    from huggingface_hub import hf_hub_download
    local = hf_hub_download(
        repo_id=dataset_repo_id,
        filename="meta/stats.json",
        repo_type="dataset",
    )
    with open(local) as f:
        raw = json.load(f)

    flat: dict[str, torch.Tensor] = {}
    for feature, stats in raw.items():
        for stat_name, values in stats.items():
            if isinstance(values, (list, float, int)):
                try:
                    flat[f"{feature}.{stat_name}"] = torch.tensor(
                        values, dtype=torch.float32
                    ).flatten()
                except (ValueError, TypeError):
                    pass
    print(
        f"  [dataset] loaded stats from {dataset_repo_id}/meta/stats.json"
        f" ({len(flat)} keys)"
    )
    return flat


def _flatten_stats_dict(nested: dict) -> dict[str, torch.Tensor]:
    """Flatten {feature: {stat: value}} → {feature.stat: tensor}."""
    flat: dict[str, torch.Tensor] = {}
    for feature, stats in nested.items():
        if not isinstance(stats, dict):
            continue
        for stat_name, value in stats.items():
            key = f"{feature}.{stat_name}"
            if isinstance(value, (int, float)):
                flat[key] = torch.tensor([value], dtype=torch.float32)
            elif isinstance(value, list):
                try:
                    flat[key] = torch.tensor(
                        value, dtype=torch.float32
                    ).flatten()
                except (ValueError, TypeError):
                    pass
    return flat


def _strip_numpy_arrays(text: str) -> str:
    """Replace numpy array(...) literals with plain Python lists."""
    result = []
    i = 0
    while i < len(text):
        if text[i : i + 6] == "array(":
            i += 6
            # Skip to opening bracket
            while i < len(text) and text[i] != "[":
                i += 1
            bracket_start = i
            depth = 0
            while i < len(text):
                if text[i] == "[":
                    depth += 1
                elif text[i] == "]":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        break
                i += 1
            bracket_end = i
            result.append(text[bracket_start:bracket_end])
            # Skip trailing ", dtype=float32)" etc.
            while i < len(text) and text[i] != ")":
                i += 1
            i += 1
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


def load_txt_log(path_or_id: str) -> dict[str, dict[str, torch.Tensor]]:
    """Parse a lerobot_policy runtime norm log file.

    Expects lines containing:
        "preprocessor step NormalizerProcessorStep stats: {DICT}"
        "postprocessor step UnnormalizerProcessorStep stats: {DICT}"
    """
    text = Path(path_or_id).read_text()

    # Strip inline file references like "lerobot_policy.py:216" that the
    # rich/logging formatter inserts at the right margin of each log line.
    clean_text = re.sub(r"\s+\w[\w.]+\.py:\d+", " ", text)

    def extract_dict_after(marker: str) -> dict | None:
        idx = clean_text.find(marker)
        if idx == -1:
            return None
        start = clean_text.find("{", idx)
        if start == -1:
            return None
        depth = 0
        i = start
        while i < len(clean_text):
            if clean_text[i] == "{":
                depth += 1
            elif clean_text[i] == "}":
                depth -= 1
                if depth == 0:
                    raw_dict = clean_text[start : i + 1]
                    break
            i += 1
        else:
            return None

        cleaned = _strip_numpy_arrays(raw_dict)
        cleaned = re.sub(r"\s+", " ", cleaned)
        try:
            return ast.literal_eval(cleaned)
        except (ValueError, SyntaxError) as e:
            print(
                f"  WARNING: failed to parse dict from {path_or_id!r}: {e}",
                file=sys.stderr,
            )
            return None

    pre_dict = extract_dict_after(
        "preprocessor step NormalizerProcessorStep stats:"
    )
    post_dict = extract_dict_after(
        "postprocessor step UnnormalizerProcessorStep stats:"
    )

    result: dict[str, dict[str, torch.Tensor]] = {"pre": {}, "post": {}}
    if pre_dict is not None:
        result["pre"] = _flatten_stats_dict(pre_dict)
        print(
            f"  [txt] parsed pre-normalizer stats"
            f" ({len(result['pre'])} keys) from {path_or_id}"
        )
    else:
        print(
            f"  [txt] WARNING: no preprocessor stats found in {path_or_id}",
            file=sys.stderr,
        )
    if post_dict is not None:
        result["post"] = _flatten_stats_dict(post_dict)
        print(
            f"  [txt] parsed post-normalizer stats"
            f" ({len(result['post'])} keys) from {path_or_id}"
        )
    return result


def load_source(
    path_or_id: str,
    label: str,
    type_hint: str | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    src_type = detect_source_type(path_or_id, type_hint)
    print(f"  [{label}] detected type: {src_type!r}  ({path_or_id})")
    if src_type == "txt":
        return load_txt_log(path_or_id)
    elif src_type == "dataset":
        flat = load_dataset_stats(path_or_id)
        return {"pre": flat, "post": {}}
    else:
        return load_normalizer_state(path_or_id, label)


# ── Value display helpers ──────────────────────────────────────────────────────

def _val_summary(t: torch.Tensor | None, width: int = 28) -> str:
    """Compact one-line summary of a tensor."""
    if t is None:
        return "(missing)".center(width)
    t = t.float().flatten()
    if t.numel() == 1:
        s = f"{t.item():.6g}"
    elif t.numel() <= 3:
        s = "[" + ", ".join(f"{v:.4g}" for v in t.tolist()) + "]"
    else:
        s = (
            f"mean={t.mean().item():.4g}"
            f" [{t.min().item():.4g},{t.max().item():.4g}]"
        )
    return s[:width].ljust(width)


# ── Comparison ─────────────────────────────────────────────────────────────────

def _status(diffs: list[float | None]) -> str:
    valid = [d for d in diffs if d is not None]
    if not valid:
        return "N/A"
    if all(d <= MATCH_THRESHOLD for d in valid):
        return "MATCH"
    if any(d <= MATCH_THRESHOLD for d in valid):
        return "PARTIAL"
    return "DIFFER"


def compare_sources(
    dicts: list[dict[str, torch.Tensor]],
    labels: list[str],
    proc_name: str,
) -> int:
    """Print multi-source comparison table; return count of non-MATCH keys."""
    n = len(dicts)
    all_keys = sorted(set().union(*[d.keys() for d in dicts]))
    if not all_keys:
        print(f"  (no keys in {proc_name})")
        return 0

    VAL_W = 28
    KEY_W = 50
    SHAPE_W = 14

    header = f"{'KEY':<{KEY_W}}  {'SHAPE':<{SHAPE_W}}"
    for lbl in labels:
        header += f"  {lbl[:VAL_W]:<{VAL_W}}"
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    for i, j in pairs:
        col = f"{labels[i][:4]}-{labels[j][:4]}"
        header += f"  {col:>10}"
    header += f"  {'STATUS':>8}"
    print("  " + header)
    print("  " + "-" * len(header))

    mismatch_count = 0

    for key in all_keys:
        tensors = [d.get(key) for d in dicts]

        shape_str = ""
        for t in tensors:
            if t is not None:
                shape_str = str(list(t.float().flatten().shape))
                break

        row = f"{key:<{KEY_W}}  {shape_str:<{SHAPE_W}}"
        for t in tensors:
            row += f"  {_val_summary(t, VAL_W)}"

        diff_vals: list[float | None] = []
        for i, j in pairs:
            ta, tb = tensors[i], tensors[j]
            if ta is None or tb is None:
                diff_vals.append(None)
                row += f"  {'n/a':>10}"
            else:
                taf = ta.float().flatten()
                tbf = tb.float().flatten()
                if taf.shape != tbf.shape:
                    diff_vals.append(None)
                    row += f"  {'shape?':>10}"
                else:
                    d = torch.max(torch.abs(taf - tbf)).item()
                    diff_vals.append(d)
                    row += f"  {d:>10.3e}"

        status = _status(diff_vals)
        if status != "MATCH":
            mismatch_count += 1
        row += f"  {status:>8}"
        print("  " + row)

    return mismatch_count


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare normalizer stats from up to 3 sources"
            " (txt log, checkpoint, HF dataset)."
        )
    )
    parser.add_argument(
        "source_a", help="Source A: .txt log, local dir, or HF repo_id"
    )
    parser.add_argument(
        "source_b", help="Source B: .txt log, local dir, or HF repo_id"
    )
    parser.add_argument(
        "source_c", nargs="?", default=None, help="Source C (optional)"
    )
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--label-c", default="C")
    parser.add_argument(
        "--type-a",
        default=None,
        choices=["txt", "checkpoint", "dataset"],
        help="Override auto-detect for source A",
    )
    parser.add_argument(
        "--type-b",
        default=None,
        choices=["txt", "checkpoint", "dataset"],
        help="Override auto-detect for source B",
    )
    parser.add_argument(
        "--type-c",
        default=None,
        choices=["txt", "checkpoint", "dataset"],
        help="Override auto-detect for source C",
    )
    parser.add_argument(
        "--proc",
        default="both",
        choices=["pre", "post", "both"],
        help="Which processor section to show (default: both)",
    )
    args = parser.parse_args()

    sources_cfg = [
        (args.source_a, args.label_a, args.type_a),
        (args.source_b, args.label_b, args.type_b),
    ]
    if args.source_c is not None:
        sources_cfg.append((args.source_c, args.label_c, args.type_c))

    print("\n=== Loading sources ===")
    all_states: list[dict[str, dict[str, torch.Tensor]]] = []
    labels: list[str] = []
    for path_or_id, label, type_hint in sources_cfg:
        print(f"\nLoading [{label}]: {path_or_id}")
        state = load_source(path_or_id, label, type_hint)
        all_states.append(state)
        labels.append(label)

    total_mismatch = 0

    sections = []
    if args.proc in ("pre", "both"):
        sections.append(("PREPROCESSOR (normalizer)", "pre"))
    if args.proc in ("post", "both"):
        sections.append(("POSTPROCESSOR (unnormalizer)", "post"))

    for proc_name, key in sections:
        dicts = [s[key] for s in all_states]
        if not any(dicts):
            continue
        sep = "=" * 100
        print(f"\n{sep}")
        print(f"  {proc_name}")
        print(
            f"  Sources: {', '.join(labels)}"
            f"   (MATCH threshold={MATCH_THRESHOLD:.0e})"
        )
        print(sep)
        n = compare_sources(dicts, labels, proc_name)
        total_mismatch += n

    sep = "=" * 100
    print(f"\n{sep}")
    if total_mismatch == 0:
        print("  RESULT: ALL MATCH across all sources")
    else:
        print(f"  RESULT: {total_mismatch} KEY(S) with PARTIAL or DIFFER status")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
