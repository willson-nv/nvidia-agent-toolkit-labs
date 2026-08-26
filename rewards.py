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

def flatten(obj, prefix="", out=None):
    """Every numeric leaf in the file, keyed by its full dotted path.

    We do not assume a shape. `eval run` and `eval reverify` nest their
    aggregate metrics differently, and the keys may or may not be prefixed with
    the agent name -- so we flatten everything and then search.
    """
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            flatten(v, f"{prefix}[{i}]", out)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix] = obj
    return out


def _first(flat, *predicates):
    """First value whose key matches any predicate, in order of preference.

    Returns None only when nothing matched. Note the explicit `is not None`
    checks -- a reward of 0.0 is a real, meaningful result and must not be
    treated as 'not found'. That bug hid a genuine 0.0 in the first version.
    """
    for pred in predicates:
        for k, v in flat.items():
            if pred(k.lower()):
                return v
    return None


def find_metrics(obj):
    flat = flatten(obj)
    return {
        "mean/reward": _first(
            flat,
            lambda k: k.endswith("mean/reward"),
            lambda k: "mean/reward" in k,
            lambda k: k.endswith("/reward") or k == "reward",
        ),
        "pass@1/accuracy": _first(
            flat,
            lambda k: "accuracy" in k and "pass@1" in k and "majority" not in k,
            lambda k: "accuracy" in k and "majority" not in k,
            lambda k: "accuracy" in k,
        ),
        "pass@1/no_answer": _first(
            flat,
            lambda k: "no_answer" in k and "pass@1" in k,
            lambda k: "no_answer" in k,
        ),
        "_flat": flat,
    }


def label(path):
    """rv_lenient_boxed_aggregate_metrics.json -> lenient_boxed"""
    b = os.path.basename(path)
    b = b.replace("_aggregate_metrics.json", "")
    for p in ("rv_", "mcqa_", "triage_"):
        if b.startswith(p):
            b = b[len(p):]
    return b or "(run)"


def show_raw(d):
    """Print every numeric key in every metrics file, so a shape change is
    diagnosable in one command instead of a round trip."""
    files = sorted(glob.glob(os.path.join(d, "*_aggregate_metrics.json")))
    if not files:
        sys.exit(f"no *_aggregate_metrics.json in {d}/")
    for f in files:
        print(f"\n=== {os.path.basename(f)} ===")
        try:
            flat = flatten(json.load(open(f)))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  unreadable: {e}")
            continue
        if not flat:
            print("  (no numeric values at all — file may be empty or a list)")
        for k, v in flat.items():
            print(f"  {k:<58} {v}")
    print("\nIf the table shows dashes, one of these key names is what it "
          "should be matching. Send this output.\n")


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
    if all(r[1] and r[1].get("mean/reward") is None for r in rows if r[1]):
        print("  Every value is a dash, which means the metric names in these files are")
        print("  not the ones this script looks for. Run:")
        print()
        print("      python3 rewards.py --raw")
        print()
        return
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
    ap.add_argument("--raw", action="store_true",
                    help="dump every numeric key found, for when the table shows dashes")
    a = ap.parse_args()
    if a.raw:
        show_raw(a.dir)
    elif a.rows:
        rows_detail(a.rows)
    else:
        compare(a.dir)
