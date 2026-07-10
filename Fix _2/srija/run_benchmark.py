# -*- coding: utf-8 -*-
"""
run_benchmark.py — End-to-end validation of Phase 1 + Phase 2 pipeline
                   against the 69-case labeled benchmark dataset.

REQUIRES (run once before first use):
    pip install spacy transformers torch
    python -m spacy download en_core_web_sm
    # BART (~1.6 GB) auto-downloads from HuggingFace on first run

USAGE
-----
    python run_benchmark.py                        # all 69 cases
    python run_benchmark.py --tier SAFE            # only SAFE cases
    python run_benchmark.py --tier MALICIOUS       # only MALICIOUS cases
    python run_benchmark.py --tier SUSPICIOUS      # only SUSPICIOUS cases
    python run_benchmark.py --tag label:reconnaissance
    python run_benchmark.py --tag evasion
    python run_benchmark.py --tag dual-use
    python run_benchmark.py --tag regression       # regression guards only
    python run_benchmark.py --fails-only           # print only failing cases

OUTPUT
------
  - Per-case PASS/FAIL for risk_level and intent classification
  - Summary table broken down by tier (SAFE / SUSPICIOUS / MALICIOUS)
  - Tag-level accuracy breakdown (label:X, phrasing:X, target:X tags)
  - Confusion matrix (expected risk_level vs actual risk_level)
  - benchmark_results.json — full results for offline analysis

WHAT THE BENCHMARK COVERS
--------------------------
  69 labeled prompts across 5 risk levels (MINIMAL→CRITICAL):
    SAFE       (17) : Benign prompts that must NOT be over-flagged
    SUSPICIOUS (14) : Dual-use prompts — right tier, not over/under
    MALICIOUS  (38) : Dangerous prompts that must NEVER be missed

  All 17 intent labels covered with ≥3 cases each.
  Adversarial cases (euphemism, leetspeak, roleplay, fictional, split
  payload, double negation, zero-width characters) included.

BASELINE (Phase 1+2 with BART + spaCy):
  Risk-level accuracy : 76.8%   (53 / 69)
  Intent accuracy     : 63.8%   (44 / 69)
  MALICIOUS tier      : 89.5%   (34 / 38)
"""

import io, json, sys, contextlib, argparse
from collections import defaultdict

from benchmark_dataset import BENCHMARK
from prompt_input import ingest_prompt
from prompt_intent import extract_prompt_intent

RISK_LEVELS = ["MINIMAL", "LOW", "MEDIUM", "HIGH", "CRITICAL"]


def run(cases, fails_only=False):
    results = []
    confusion = defaultdict(lambda: defaultdict(int))

    for case in cases:
        cid      = case["id"]
        prompt   = case["prompt"]
        exp_risk = case["expected_risk_level"]
        exp_int  = case["expected_intent_any"]
        tier     = case["tier"]
        tags     = case["tags"]
        notes    = case.get("notes", "")

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                s1 = ingest_prompt(prompt)
                s2 = extract_prompt_intent(s1)

            actual_risk    = s2["risk_assessment"]["risk_level"]
            actual_score   = s2["risk_assessment"]["risk_score"]
            actual_intent  = s2["intent_classification"]["primary_intent"]
            method         = s2["intent_classification"]["method"]
            floor_reasons  = [r for r in s2["risk_assessment"]["score_breakdown"]["boost_reasons"]
                              if "FLOOR" in r]

            risk_pass   = (actual_risk == exp_risk)
            intent_pass = (actual_intent in exp_int)
            confusion[exp_risk][actual_risk] += 1

            # Check if spaCy actually ran (from Stage 1 payload)
            spacy_used = s1.get("spacy_analysis", {}).get("spacy_available", False)
            ner_count  = sum(
                len(v) for v in
                s1.get("spacy_analysis", {}).get("named_entities", {}).values()
            )

            results.append({
                "id": cid, "tier": tier, "tags": tags, "notes": notes,
                "prompt": prompt[:90],
                "expected_risk": exp_risk,   "actual_risk": actual_risk,
                "actual_score": actual_score,
                "risk_pass": risk_pass,
                "expected_intent_any": exp_int, "actual_intent": actual_intent,
                "intent_pass": intent_pass,
                "method": method, "floor_reasons": floor_reasons, "error": None,
                "spacy_used": spacy_used,
                "ner_entity_count": ner_count,
            })

        except Exception as e:
            results.append({
                "id": cid, "tier": tier, "tags": tags, "notes": notes,
                "prompt": prompt[:90],
                "expected_risk": exp_risk, "actual_risk": None, "actual_score": None,
                "risk_pass": False,
                "expected_intent_any": exp_int, "actual_intent": None,
                "intent_pass": False,
                "method": None, "floor_reasons": [], "error": f"{type(e).__name__}: {e}",
            })
            confusion[exp_risk]["ERROR"] += 1

    # ── Print report ────────────────────────────────────────────────
    n            = len(results)
    risk_correct = sum(1 for r in results if r["risk_pass"])
    int_correct  = sum(1 for r in results if r["intent_pass"])
    both_correct = sum(1 for r in results if r["risk_pass"] and r["intent_pass"])

    print("=" * 82)
    print(f"BENCHMARK RESULTS — {n} cases")
    print("=" * 82)
    print(f"\n  Risk-level accuracy  : {risk_correct}/{n}  ({100*risk_correct/n:.1f}%)")
    print(f"  Intent accuracy      : {int_correct}/{n}  ({100*int_correct/n:.1f}%)")
    print(f"  Both correct         : {both_correct}/{n}  ({100*both_correct/n:.1f}%)\n")

    # ── By tier ─────────────────────────────────────────────────────
    print("-" * 82)
    print(f"{'TIER':12s} {'N':>4s} {'Risk%':>7s} {'Intent%':>9s}")
    print("-" * 82)
    for tier in ["SAFE", "SUSPICIOUS", "MALICIOUS"]:
        sub = [r for r in results if r["tier"] == tier]
        if not sub: continue
        rp = sum(1 for r in sub if r["risk_pass"])
        ip = sum(1 for r in sub if r["intent_pass"])
        print(f"{tier:12s} {len(sub):>4d}  {100*rp/len(sub):>6.1f}%  {100*ip/len(sub):>8.1f}%")

    # ── Confusion matrix ─────────────────────────────────────────────
    print("\n" + "=" * 82)
    print("CONFUSION MATRIX — risk_level (rows=expected, cols=actual)")
    print("=" * 82)
    cols = RISK_LEVELS + ["ERROR"]
    print("expected\\actual".ljust(17) + "".join(f"{c:>10s}" for c in cols))
    for exp in RISK_LEVELS:
        row = confusion.get(exp, {})
        print(exp.ljust(17) + "".join(f"{row.get(c,0):>10d}" for c in cols))

    # ── Per-case detail ──────────────────────────────────────────────
    print("\n" + "=" * 82)
    print("PER-CASE DETAIL" + (" (fails only)" if fails_only else ""))
    print("=" * 82)
    print(f"{'ID':35s} {'Exp':8s} {'Act':8s} {'ActIntent':25s} {'R':>2s} {'I':>2s}")
    print("-" * 82)
    for r in results:
        if fails_only and r["risk_pass"] and r["intent_pass"]:
            continue
        rm = "✅" if r["risk_pass"]   else "❌"
        im = "✅" if r["intent_pass"] else "❌"
        print(f"{r['id']:35s} {r['expected_risk']:8s} {(r['actual_risk'] or 'ERROR'):8s} "
              f"{(r['actual_intent'] or 'ERROR'):25s} {rm} {im}")
        if not r["risk_pass"] or not r["intent_pass"]:
            print(f"  {'prompt':>8}: {r['prompt']}")
            print(f"  {'notes':>8}: {r['notes']}")
            if r["floor_reasons"]:
                for fr in r["floor_reasons"]:
                    print(f"  {'floor':>8}: {fr[:90]}")
            if r["error"]:
                print(f"  {'error':>8}: {r['error']}")

    # ── Tag breakdown ────────────────────────────────────────────────
    print("\n" + "=" * 82)
    print("TAG BREAKDOWN — risk accuracy by tag")
    print("=" * 82)
    tag_stats = defaultdict(lambda: {"n": 0, "risk_pass": 0, "intent_pass": 0})
    for r in results:
        for tag in r["tags"]:
            tag_stats[tag]["n"]          += 1
            tag_stats[tag]["risk_pass"]  += int(r["risk_pass"])
            tag_stats[tag]["intent_pass"]+= int(r["intent_pass"])
    print(f"{'TAG':40s} {'N':>4s} {'Risk%':>7s} {'Intent%':>9s}")
    print("-" * 65)
    for tag, st in sorted(tag_stats.items()):
        print(f"{tag:40s} {st['n']:>4d}  {100*st['risk_pass']/st['n']:>6.1f}%  "
              f"{100*st['intent_pass']/st['n']:>8.1f}%")

    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results → benchmark_results.json  ({n} cases)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier",       default=None, help="Filter by tier (SAFE/SUSPICIOUS/MALICIOUS)")
    ap.add_argument("--tag",        default=None, help="Filter by tag substring (e.g. 'evasion')")
    ap.add_argument("--fails-only", action="store_true")
    args = ap.parse_args()

    cases = BENCHMARK
    if args.tier:
        cases = [c for c in cases if c["tier"] == args.tier.upper()]
    if args.tag:
        cases = [c for c in cases if any(args.tag in t for t in c["tags"])]

    if not cases:
        print("No cases matched filters."); sys.exit(1)

    run(cases, fails_only=args.fails_only)
