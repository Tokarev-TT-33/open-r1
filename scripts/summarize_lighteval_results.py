#!/usr/bin/env python3
"""
Summarize evaluation outputs into a single comparison report (Markdown + CSV + JSON).

Sources:
- LightEval outputs: `results_*.json`
- Self-consistency outputs: `math500_self_consistency_*.json`

The output is meant to be a practical "one table" view to compare:
- quality (pass@1, any-of-k, majority vote)
- efficiency (wall time if available)
and includes the local file path for each experiment.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Row:
    kind: str  # "lighteval" | "self_consistency"
    run_path: str
    model_name: str
    task_key: str
    effective_num_docs: Optional[int]
    total_eval_seconds: Optional[float]
    # generation parameters
    max_new_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    use_beam_search: Optional[bool]
    beam_width: Optional[int]
    length_penalty: Optional[float]
    seed: Optional[int]
    # metrics (commonly present in math_500)
    math_pass_1_1: Optional[float]
    math_pass_1_1_stderr: Optional[float]
    math_pass_1_4: Optional[float]
    math_pass_1_4_stderr: Optional[float]
    math_maj_4: Optional[float]
    math_maj_4_stderr: Optional[float]
    # self-consistency metrics (when available)
    sc_k: Optional[int]
    sc_pass1_first: Optional[float]
    sc_any_of_k: Optional[float]
    sc_majority: Optional[float]


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def _load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_task_key(data: Dict[str, Any]) -> str:
    results = data.get("results") or {}
    # Prefer a concrete task key rather than "all"
    for k in results.keys():
        if k != "all":
            return k
    return "all"


def _extract_effective_num_docs(data: Dict[str, Any], task_name: str) -> Optional[int]:
    cfg_tasks = data.get("config_tasks") or {}
    # config_tasks uses key "lighteval|math_500" (no seed suffix)
    # task_name might be "lighteval|math_500|0"
    base = task_name.rsplit("|", 1)[0] if task_name.count("|") >= 2 else task_name
    t = cfg_tasks.get(base)
    if isinstance(t, dict):
        return _safe_int(t.get("effective_num_docs"))
    return None


def _infer_strategy(gen: Dict[str, Any]) -> str:
    if gen.get("use_beam_search"):
        bw = gen.get("beam_width")
        return f"beam_w{bw}" if bw is not None else "beam"
    # "greedy" in your sweep is implemented as temperature=1e-6, top_p=1.0, top_k=0
    temp = gen.get("temperature")
    top_p = gen.get("top_p")
    top_k = gen.get("top_k")
    if temp is not None and float(temp) <= 1e-5 and (top_p == 1.0 or top_p is None) and (top_k == 0 or top_k is None):
        return "greedy"
    if top_p is not None and top_p != 1.0:
        return "top_p"
    if top_k is not None and top_k != 0:
        return "top_k"
    return "sample"


def _extract_row(p: Path) -> Optional[Row]:
    try:
        data = _load_json(p)
    except Exception:
        return None

    cfg = data.get("config_general") or {}
    gen = cfg.get("generation_parameters") or {}
    task_key = _extract_task_key(data)
    results = (data.get("results") or {}).get(task_key, (data.get("results") or {}).get("all", {})) or {}

    row = Row(
        kind="lighteval",
        run_path=str(p),
        model_name=str(cfg.get("model_name") or ""),
        task_key=task_key,
        effective_num_docs=_extract_effective_num_docs(data, task_key),
        total_eval_seconds=_safe_float(cfg.get("total_evaluation_time_secondes")),
        max_new_tokens=_safe_int(gen.get("max_new_tokens")),
        temperature=_safe_float(gen.get("temperature")),
        top_p=_safe_float(gen.get("top_p")),
        top_k=_safe_int(gen.get("top_k")),
        use_beam_search=gen.get("use_beam_search") if gen.get("use_beam_search") is None else bool(gen.get("use_beam_search")),
        beam_width=_safe_int(gen.get("beam_width")),
        length_penalty=_safe_float(gen.get("length_penalty")),
        seed=_safe_int(gen.get("seed")),
        math_pass_1_1=_safe_float(results.get("math_pass@1:1_samples")),
        math_pass_1_1_stderr=_safe_float(results.get("math_pass@1:1_samples_stderr")),
        math_pass_1_4=_safe_float(results.get("math_pass@1:4_samples")),
        math_pass_1_4_stderr=_safe_float(results.get("math_pass@1:4_samples_stderr")),
        math_maj_4=_safe_float(results.get("math_maj@4_samples")),
        math_maj_4_stderr=_safe_float(results.get("math_maj@4_samples_stderr")),
        sc_k=None,
        sc_pass1_first=None,
        sc_any_of_k=None,
        sc_majority=None,
    )
    return row


def _scan_results(root: Path) -> List[Path]:
    return sorted(root.rglob("results_*.json"))


def _scan_self_consistency(root: Path) -> List[Path]:
    return sorted(root.rglob("math500_self_consistency_*.json"))


def _extract_sc_row(p: Path) -> Optional[Row]:
    try:
        data = _load_json(p)
    except Exception:
        return None

    summ = data.get("summary") or {}
    cfg = summ.get("config") or {}
    runtime = summ.get("runtime") or {}
    metrics = summ.get("metrics") or {}

    # Older runs may store the top-level dict directly (legacy). Support that too.
    if not summ and ("task" in data and "metrics" in data and "config" in data):
        cfg = data.get("config") or {}
        metrics = data.get("metrics") or {}
        summ = {"task": data.get("task"), "num_problems": data.get("num_problems")}
        runtime = {}

    return Row(
        kind="self_consistency",
        run_path=str(p),
        model_name=str(cfg.get("model") or ""),
        task_key=str(summ.get("task") or "math_500"),
        effective_num_docs=_safe_int(summ.get("num_problems")),
        total_eval_seconds=_safe_float(runtime.get("wall_time_seconds")),
        max_new_tokens=_safe_int(cfg.get("max_new_tokens")),
        temperature=_safe_float(cfg.get("temperature")),
        top_p=_safe_float(cfg.get("top_p")),
        top_k=_safe_int(cfg.get("top_k")),
        use_beam_search=False,
        beam_width=None,
        length_penalty=None,
        seed=_safe_int(cfg.get("seed")),
        # Map to lighteval-like slots for easy sorting/comparison
        math_pass_1_1=_safe_float(metrics.get("pass@1_first_sample")),
        math_pass_1_1_stderr=None,
        math_pass_1_4=None,
        math_pass_1_4_stderr=None,
        math_maj_4=None,
        math_maj_4_stderr=None,
        sc_k=_safe_int(cfg.get("num_samples")),
        sc_pass1_first=_safe_float(metrics.get("pass@1_first_sample")),
        sc_any_of_k=_safe_float(metrics.get("pass@1_any_of_k")),
        sc_majority=_safe_float(metrics.get("self_consistency_majority_vote")),
    )


def _fmt(x: Optional[float], ndigits: int = 4) -> str:
    if x is None:
        return "-"
    return f"{x:.{ndigits}f}"


def _seconds_to_hms(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m > 0:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def _write_csv(rows: List[Row], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        if rows:
            w.writeheader()
            for r in rows:
                w.writerow(asdict(r))


def _write_json(rows: List[Row], out_json: Path, meta: Dict[str, Any]) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "rows": [asdict(r) for r in rows]}
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_md(rows: List[Row], out_md: Path, meta: Dict[str, Any]) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)

    # Group by model+task and show key configs
    lines: List[str] = []
    lines.append(f"# 评测结果对比报告（lighteval + self-consistency）\n")
    lines.append(f"- 生成时间：{meta.get('generated_at')}")
    lines.append(f"- 扫描目录：`{meta.get('scan_root')}`")
    lines.append(f"- 共找到结果文件数：{meta.get('num_files')}\n")

    # Table header
    lines.append("|kind|strategy|task|docs|max_new_tokens|temp|top_p|top_k|beam_width|pass@1(1)|pass@1(4)|maj@4(math)|SC_k|SC_majority|SC_any_of_k|time|run_path|")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for r in rows:
        strategy = _infer_strategy(
            {
                "use_beam_search": r.use_beam_search,
                "beam_width": r.beam_width,
                "temperature": r.temperature,
                "top_p": r.top_p,
                "top_k": r.top_k,
            }
        )
        lines.append(
            "|{kind}|{strategy}|{task}|{docs}|{mxt}|{temp}|{top_p}|{top_k}|{bw}|{p11}±{p11e}|{p14}±{p14e}|{m4}±{m4e}|{sck}|{scm}|{sca}|{t}|`{path}`|".format(
                kind=r.kind,
                strategy=strategy,
                task=r.task_key,
                docs=r.effective_num_docs if r.effective_num_docs is not None else "-",
                mxt=r.max_new_tokens if r.max_new_tokens is not None else "-",
                temp=_fmt(r.temperature, 6) if r.temperature is not None else "-",
                top_p=_fmt(r.top_p, 4) if r.top_p is not None else "-",
                top_k=r.top_k if r.top_k is not None else "-",
                bw=r.beam_width if r.beam_width is not None else "-",
                p11=_fmt(r.math_pass_1_1, 4),
                p11e=_fmt(r.math_pass_1_1_stderr, 4),
                p14=_fmt(r.math_pass_1_4, 4),
                p14e=_fmt(r.math_pass_1_4_stderr, 4),
                m4=_fmt(r.math_maj_4, 4),
                m4e=_fmt(r.math_maj_4_stderr, 4),
                sck=r.sc_k if r.sc_k is not None else "-",
                scm=_fmt(r.sc_majority, 4),
                sca=_fmt(r.sc_any_of_k, 4),
                t=_seconds_to_hms(r.total_eval_seconds),
                path=r.run_path,
            )
        )

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-root", type=str, default="data/evals", help="Directory to scan for evaluation result jsons")
    ap.add_argument("--out-dir", type=str, default="data/evals/comparisons", help="Output directory for reports")
    ap.add_argument("--filter-task-contains", type=str, default="math_500", help="Keep only task_key containing this substring")
    ap.add_argument("--filter-model-contains", type=str, default="Qwen3-4B-Math-220k-GRPO-full_v1_checkpoint-2300", help="Keep only model_name containing this substring")
    ap.add_argument("--sort-by", type=str, default="total_eval_seconds", choices=["total_eval_seconds", "math_pass_1_1", "math_pass_1_4", "max_new_tokens"], help="Sort key")
    args = ap.parse_args()

    scan_root = Path(args.scan_root)
    out_dir = Path(args.out_dir)
    paths = _scan_results(scan_root)
    sc_paths = _scan_self_consistency(scan_root)
    rows: List[Row] = []
    for p in paths:
        row = _extract_row(p)
        if row is None:
            continue
        if args.filter_task_contains and args.filter_task_contains not in row.task_key:
            continue
        if args.filter_model_contains and args.filter_model_contains not in row.model_name:
            continue
        rows.append(row)

    # Add self-consistency rows (they don't follow lighteval schema)
    for p in sc_paths:
        row = _extract_sc_row(p)
        if row is None:
            continue
        if args.filter_task_contains and args.filter_task_contains not in row.task_key:
            continue
        if args.filter_model_contains and args.filter_model_contains not in row.model_name:
            continue
        rows.append(row)

    # Sort
    rows.sort(key=lambda r: getattr(r, args.sort_by) if getattr(r, args.sort_by) is not None else -1, reverse=(args.sort_by in {"math_pass_1_1", "math_pass_1_4"}))

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    meta = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "scan_root": str(scan_root),
        "num_files": len(paths) + len(sc_paths),
        "num_rows_after_filter": len(rows),
        "filters": {
            "task_contains": args.filter_task_contains,
            "model_contains": args.filter_model_contains,
        },
    }

    out_md = out_dir / f"lighteval_comparison_{stamp}.md"
    out_csv = out_dir / f"lighteval_comparison_{stamp}.csv"
    out_json = out_dir / f"lighteval_comparison_{stamp}.json"

    _write_md(rows, out_md, meta)
    if rows:
        _write_csv(rows, out_csv)
    _write_json(rows, out_json, meta)

    print(f"[OK] Wrote:\n- {out_md}\n- {out_csv if rows else '(csv skipped: no rows)'}\n- {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


