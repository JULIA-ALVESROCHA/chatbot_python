#!/usr/bin/env python3
"""Regenerate results_report.md from an existing results.json (no re-scoring)."""
import argparse, json, importlib.util, sys

def load_eval():
    spec = importlib.util.spec_from_file_location("le", "lumie_eval.py")
    le = importlib.util.module_from_spec(spec); sys.modules["le"] = le
    spec.loader.exec_module(le); return le

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="eval/results.json")
    ap.add_argument("--report", default="eval/results_report.md")
    a = ap.parse_args()
    le = load_eval()
    with open(a.results, encoding="utf-8") as f:
        data = json.load(f)
    seed_results = data["per_seed"]
    model = data.get("config", {}).get("model", "gpt-4o-mini")
    agg = le.aggregate(seed_results)
    report = le.render_report(agg, model)
    with open(a.report, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report regenerated -> {a.report}")

if __name__ == "__main__":
    main()