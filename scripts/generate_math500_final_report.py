#!/usr/bin/env python3
"""
Generate a single consolidated report for all MATH-500 experiments:
- lighteval results_*.json
- self_consistency outputs (math500_self_consistency_*.json)

Writes a Markdown report with one unified table + links to all source files.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Entry:
    kind: str  # lighteval | self_consistency
    name: str
    docs: Optional[int]
    wall_time_s: Optional[float]
    max_new_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    beam_width: Optional[int]
    # metrics (unified)
    pass1_1: Optional[float]
    pass1_4: Optional[float]
    sc_majority: Optional[float]
    sc_any_of_k: Optional[float]
    sc_k: Optional[int]
    path: str


def load_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def safe_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def fmt(x: Optional[float], nd: int = 4) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def sec_to_hms(x: Optional[float]) -> str:
    if x is None:
        return "-"
    s = int(x)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h:
        return f"{h}h{m:02d}m{sec:02d}s"
    if m:
        return f"{m}m{sec:02d}s"
    return f"{sec}s"


def pick_strategy_name(gen: Dict[str, Any]) -> str:
    if gen.get("use_beam_search"):
        bw = gen.get("beam_width")
        return f"beam_w{bw}" if bw else "beam"
    temp = gen.get("temperature")
    top_p = gen.get("top_p")
    top_k = gen.get("top_k")
    if temp is not None and float(temp) <= 1e-5 and (top_p in (None, 1.0)) and (top_k in (None, 0)):
        return "greedy"
    if top_p is not None and top_p != 1.0:
        return "top_p"
    if top_k is not None and top_k != 0:
        return "top_k"
    return "sample"


def parse_lighteval(p: Path) -> Optional[Entry]:
    d = load_json(p)
    cfg = d.get("config_general") or {}
    gen = cfg.get("generation_parameters") or {}
    # task key
    task_key = "all"
    for k in (d.get("results") or {}).keys():
        if k != "all":
            task_key = k
            break
    # docs
    docs = None
    base = task_key.rsplit("|", 1)[0] if task_key.count("|") >= 2 else task_key
    tcfg = (d.get("config_tasks") or {}).get(base)
    if isinstance(tcfg, dict):
        docs = safe_int(tcfg.get("effective_num_docs"))
    res = (d.get("results") or {}).get("all", {}) or {}
    name = pick_strategy_name(gen)
    return Entry(
        kind="lighteval",
        name=name,
        docs=docs,
        wall_time_s=safe_float(cfg.get("total_evaluation_time_secondes")),
        max_new_tokens=safe_int(gen.get("max_new_tokens")),
        temperature=safe_float(gen.get("temperature")),
        top_p=safe_float(gen.get("top_p")),
        top_k=safe_int(gen.get("top_k")),
        beam_width=safe_int(gen.get("beam_width")) if gen.get("use_beam_search") else None,
        pass1_1=safe_float(res.get("math_pass@1:1_samples")),
        pass1_4=safe_float(res.get("math_pass@1:4_samples")),
        sc_majority=None,
        sc_any_of_k=None,
        sc_k=None,
        path=str(p),
    )


def parse_sc(p: Path) -> Optional[Entry]:
    d = load_json(p)
    summ = d.get("summary") or {}
    cfg = summ.get("config") or {}
    rt = summ.get("runtime") or {}
    met = summ.get("metrics") or {}
    return Entry(
        kind="self_consistency",
        name=f"self_consistency_k{safe_int(cfg.get('num_samples'))}",
        docs=safe_int(summ.get("num_problems")),
        wall_time_s=safe_float(rt.get("wall_time_seconds")),
        max_new_tokens=safe_int(cfg.get("max_new_tokens")),
        temperature=safe_float(cfg.get("temperature")),
        top_p=safe_float(cfg.get("top_p")),
        top_k=safe_int(cfg.get("top_k")),
        beam_width=None,
        # Map to lighteval-like names for easier comparison:
        pass1_1=safe_float(met.get("pass@1_first_sample")),
        pass1_4=None,
        sc_majority=safe_float(met.get("self_consistency_majority_vote")),
        sc_any_of_k=safe_float(met.get("pass@1_any_of_k")),
        sc_k=safe_int(cfg.get("num_samples")),
        path=str(p),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan-root", default="data/evals", type=str)
    ap.add_argument("--out", default="data/evals/reports/math500_sampling_report_final.md", type=str)
    args = ap.parse_args()

    root = Path(args.scan_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    entries: List[Entry] = []
    for p in root.rglob("results_*.json"):
        e = parse_lighteval(p)
        if e and "math_500" in e.path:
            entries.append(e)
    for p in root.rglob("math500_self_consistency_*.json"):
        e = parse_sc(p)
        if e:
            entries.append(e)

    # Keep only full-500 runs for conclusions, but list all links
    full = [e for e in entries if e.docs == 500]

    # pick best accuracy among full runs:
    best_effect = None
    for e in full:
        # prefer self-consistency majority, else pass1_1
        score = e.sc_majority if e.sc_majority is not None else e.pass1_1
        if score is None:
            continue
        if best_effect is None or score > best_effect[0]:
            best_effect = (score, e)

    best_efficiency = None
    for e in full:
        score = e.sc_majority if e.sc_majority is not None else e.pass1_1
        if score is None or e.wall_time_s is None or e.wall_time_s <= 0:
            continue
        eff = score / e.wall_time_s  # score per second
        if best_efficiency is None or eff > best_efficiency[0]:
            best_efficiency = (eff, e)

    lines: List[str] = []
    lines.append("## MATH-500 采样/解码策略最终汇总报告\n")
    lines.append(f"生成时间：{dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"扫描目录：`{root}`")
    lines.append(f"输出文件：`{out}`\n")

    if best_effect:
        lines.append(f"- **效果最好**：`{best_effect[1].name}`（score={fmt(best_effect[0])}）")
    if best_efficiency:
        lines.append(f"- **性价比最高（score/秒）**：`{best_efficiency[1].name}`（{best_efficiency[0]:.6g} /s）")
    lines.append("")

    lines.append("### 统一总表（含 lighteval + self-consistency）")
    lines.append("|kind|name|docs|max_new_tokens|temp|top_p|top_k|beam_w|pass@1(1)|pass@1(4)|SC_majority|SC_any_of_k|time|path|")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for e in sorted(entries, key=lambda x: (x.docs or 0, x.kind, x.name, x.path)):
        lines.append(
            "|{kind}|{name}|{docs}|{mxt}|{temp}|{top_p}|{top_k}|{bw}|{p11}|{p14}|{scm}|{sca}|{t}|`file://{path}`|".format(
                kind=e.kind,
                name=e.name,
                docs=e.docs if e.docs is not None else "-",
                mxt=e.max_new_tokens if e.max_new_tokens is not None else "-",
                temp=fmt(e.temperature, 6) if e.temperature is not None else "-",
                top_p=fmt(e.top_p, 4) if e.top_p is not None else "-",
                top_k=e.top_k if e.top_k is not None else "-",
                bw=e.beam_width if e.beam_width is not None else "-",
                p11=fmt(e.pass1_1),
                p14=fmt(e.pass1_4),
                scm=fmt(e.sc_majority),
                sca=fmt(e.sc_any_of_k),
                t=sec_to_hms(e.wall_time_s),
                path=e.path,
            )
        )

    lines.append("\n### 说明")
    lines.append("- lighteval 的 `math_pass@1:4_samples` = PassAtK(k=1,n=4)，属于 **any-of-4**，不是多数投票。")
    lines.append("- self-consistency 文件里的 `pass@1_first_sample` 与 lighteval 的 `math_pass@1:1_samples` 可对齐（同抽取+sympy 校验），但 `SC_majority` 是额外指标。")
    lines.append("- 旧版 self-consistency 输出可能没有 `runtime.wall_time_seconds`；新版本脚本已补充记录。")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[OK] wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


