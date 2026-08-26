#!/usr/bin/env python3
"""Read what a gym eval run or reverify actually produced.

    python3 rewards.py                      compare every run in results/
    python3 rewards.py --rows FILE.jsonl    per-row detail for one run
    python3 rewards.py --dir some/other     look somewhere else

WHY THIS EXISTS
`gym eval reverify` prints its Key metrics block to stdout -- in the middle of
about twenty seconds of Ray startup chatter and a spray of GCS shutdown warnings
that look like errors. Live, in front of a room, that block is gone off the top
of the terminal before you have finished the sentence.

It also writes the numbers to disk, and nothing tells you that. This reads them
back, so Lab 4's four-mode sweep is one command at the end instead of scrollback
archaeology.
"""
import argparse
import glob
import json
import os
import sys

# The metric names Gym uses. We search for these rather than assuming the shape
# of the file, because the aggregate JSON nests differently depending on whether
# it came from `eval run` or `eval reverify`.
WANT = ["mean/reward", "pass@1/accuracy", "pass@1/no_answer"]


def find_metrics(obj, out=None, depth=0):
    """Pull metric keys out of an arbitrarily nested dict."""
    if out is None:
        out = {}
    if depth > 6 or not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            # keep the shortest path to each metric name
            if k in WANT and k not in out:
                out[k] = v
            elif k.startswith("pass@1[") and "accuracy" in k and "pass@1/accuracy" not in out:
                out["pass@1/accuracy"] = v
            elif k.startswith("pass@1[") and "no_answer" in k and "pass@1/no_answer" not in out:
                out["pass@1/no_answer"] = v
        elif isinstance(v, dict):
            find_metrics(v, out, depth + 1)
    return out


def label(path):
    """rv_lenient_boxed_aggregate_metrics.json -> lenient_boxed"""
    b = os.path.basename(path)
    b = b.replace("_aggregate_metrics.json", "")
    for p in ("rv_", "mcqa_", "triage_"):
        if b.startswith(p):
            b = b[len(p):]
    return b or "(run)"


def compare(d):
    files = sorted(glob.glob(os.path.join(d, "*_aggregate_metrics.json")))
    if not files:
        sys.exit(
            f"no *_aggregate_metrics.json in {d}/\n"
            "Run an eval or a reverify first -- and check you passed --output."
        )

    rows = []
    for f in files:
        try:
            m = find_metrics(json.load(open(f)))
        except (json.JSONDecodeError, OSError) as e:
            rows.append((label(f), None, f"unreadable: {e}"))
            continue
        rows.append((label(f), m, None))

    w = max(len(r[0]) for r in rows) + 2
    print()
    print(f"  {'RUN'.ljust(w)}{'REWARD':>9}{'ACCURACY':>11}{'NO_ANSWER':>12}")
    print("  " + "-" * (w + 32))
    for name, m, err in rows:
        if err:
            print(f"  {name.ljust(w)}  {err}")
            continue
        r = m.get("mean/reward")
        a = m.get("pass@1/accuracy")
        n = m.get("pass@1/no_answer")
        print(f"  {name.ljust(w)}"
              f"{'-' if r is None else f'{r:.3f}':>9}"
              f"{'-' if a is None else f'{a:.1f}%':>11}"
              f"{'-' if n is None else f'{n:.1f}%':>12}")
    print()
    print("  Higher reward is better. A high no_answer means the grader could not")
    print("  read the response at all -- which is a different problem from a wrong")
    print("  answer, and has a different fix.")
    print()


def rows_detail(path):
    if not os.path.exists(path):
        sys.exit(f"no such file: {path}")
    print()
    print(f"  {'ROW':>4}{'REWARD':>9}   {'EXPECTED':<12}{'EXTRACTED':<12}")
    print("  " + "-" * 42)
    n = hits = 0
    total = 0.0
    for i, line in enumerate(open(path)):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        n += 1
        rw = r.get("reward")
        if isinstance(rw, (int, float)):
            total += rw
            hits += 1 if rw and rw > 0 else 0
        exp = r.get("expected_answer")
        got = r.get("extracted_answer")
        # support_triage and other envs use different fields
        if exp is None and got is None:
            got = r.get("parsed")
        flag = ""
        if exp is not None and got is not None:
            flag = "  hit" if str(exp) == str(got) else "  MISS"
        print(f"  {i:>4}{'-' if rw is None else f'{rw:.2f}':>9}   "
              f"{str(exp)[:11]:<12}{str(got)[:11]:<12}{flag}")
    if n:
        print("  " + "-" * 42)
        print(f"  {n} rows, mean reward {total / n:.3f}, {hits} scored above zero")
    print()
    print("  Read the rows, not the mean. A row that recovers as the WRONG answer")
    print("  was never a hard question the model failed -- it answered confidently")
    print("  and got it wrong, and a strict grader scores that the same as a right")
    print("  answer in the wrong format.")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results", help="where the run outputs are")
    ap.add_argument("--rows", metavar="FILE.jsonl", help="per-row detail for one run")
    a = ap.parse_args()
    rows_detail(a.rows) if a.rows else compare(a.dir)
