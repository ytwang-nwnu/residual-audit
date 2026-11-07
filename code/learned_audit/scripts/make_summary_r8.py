"""PREREG_R8 summary for learned detectors: S3 (test-only calibration) and S4 (label-free).

Reads frozen s1/s2 runs (baselines) and new s3/s4 runs. Writes only summary_r8/. Never touches
summary/ or run_selection.json. Same AUC / pooled / recovery definitions as make_summary.py.
"""
import csv, glob, json, os, re, statistics as st
import numpy as np

RES = "/data1/users/yangrui/results/idea67b/exp/prl_v2/round7"
OUT = os.path.join(RES, "summary_r8"); os.makedirs(OUT, exist_ok=True)
RUN = re.compile(r"^\d{4}-\d{2}-\d{2}_v(\d+)_(m1|m2|m3p|m4|m5)_(s1|s2|s3|s4)_fold(\d+)_seed(\d+)$")
ACTS = ["SQUAT", "Lunges", "Plank"]; METHODS = ["m1", "m2", "m3p", "m4", "m5"]
SEEDS = {"m1": [0, 1, 2], "m3p": [0, 1, 2], "m2": [0], "m4": [0], "m5": [0]}
EXCLUDE_SUBSTR = ("pilot", "_v49")  # smoke runs


import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import make_summary as MS                      # frozen selection rule + VOID_RULES, read-only use
from metrics_r7 import auc_mw, auc_of
MS.RUN_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}_v(\d+)_(m1|m2|m3p|m4|m5)_(s0|s1|s2|s3|s4)_fold(\d+)_seed(\d+)$")


def auc(err, cor):
    return auc_mw(err, cor)


def load_runs():
    """Same selection rule as make_summary (void rules by commit, highest version wins),
    extended to s3/s4. Returns {(method, split, fold, seed): rows} with typed rows."""
    runs, _sel = MS.select_runs()
    return {k: v["rows"] for k, v in runs.items()}


def pooled_auc(runs, meth, cond, act, seed):
    rows = []
    split = cond.split("_")[0]
    for f in range(4):
        if (meth, split, f, seed) in runs:
            rows += [r for r in runs[(meth, split, f, seed)] if r["split"] == cond and r["exercise"] == act]
    if not rows: return float("nan"), 0
    err = [float(r["score"]) for r in rows if r["label"] == 1]; cor = [float(r["score"]) for r in rows if r["label"] == 0]
    return auc(err, cor), len(rows)


def recovery_k(runs, meth, prefix, raw_auc, all_auc, act, seed, k):
    """mean over draws of (AUC_k − raw) / (all − raw)."""
    split = prefix
    draws = set()
    for f in range(4):
        if (meth, split, f, seed) in runs:
            draws |= {r["draw"] for r in runs[(meth, split, f, seed)] if r["split"] == "%s_k%d" % (prefix, k)}
    vals = []
    for d in sorted(draws):
        rows = []
        for f in range(4):
            if (meth, split, f, seed) in runs:
                rows += [r for r in runs[(meth, split, f, seed)] if r["split"] == "%s_k%d" % (prefix, k) and r["draw"] == d and r["exercise"] == act]
        a = auc([float(r["score"]) for r in rows if r["label"] == 1], [float(r["score"]) for r in rows if r["label"] == 0])
        vals.append((a - raw_auc) / (all_auc - raw_auc))
    return float(np.mean(vals)) if vals else float("nan")


def main():
    runs = load_runs()
    have = sorted({(k[0], k[1]) for k in runs})
    print("runs present:", have)
    table = []
    for meth in METHODS:
        for act in ACTS:
            per_seed = []
            for seed in SEEDS[meth]:
                if (meth, "s3", 0, seed) not in runs and (meth, "s4", 0, seed) not in runs: continue
                a1, n1 = pooled_auc(runs, meth, "s1", act, seed)
                a2, _ = pooled_auc(runs, meth, "s2_all", act, seed)
                a3raw, _ = pooled_auc(runs, meth, "s3_raw", act, seed)
                a3, _ = pooled_auc(runs, meth, "s3_all", act, seed)
                a4, _ = pooled_auc(runs, meth, "s4_all", act, seed)
                row = dict(method=meth, action=act, seed=seed, auc_s1=a1, auc_s2all=a2, auc_s3raw=a3raw, auc_s3all=a3, auc_s4all=a4,
                           d_s2=a2 - a1, d_s3=a3 - a3raw, d_s4=a4 - a1)
                row["rec_s3_k2"] = recovery_k(runs, meth, "s3", a3raw, a3, act, seed, 2) if not np.isnan(a3) else float("nan")
                row["rec_s4_k2"] = recovery_k(runs, meth, "s4", a1, a4, act, seed, 2) if not np.isnan(a4) else float("nan")
                per_seed.append(row)
            if not per_seed: continue
            agg = {k: st.mean(r[k] for r in per_seed) for k in per_seed[0] if k not in ("seed", "method", "action")}
            agg.update(method=meth, action=act, n_seeds=len(per_seed))
            agg["failing_cell"] = agg["d_s2"] >= 0.05          # R7 rule: mean over seeds
            if not agg["failing_cell"]:
                agg["rec_s3_k2"] = agg["rec_s4_k2"] = float("nan")
            table.append(agg)
    # verdicts
    def cells(pred):
        return [r for r in table if r["failing_cell"] and pred(r)]
    fail = [r for r in table if r["failing_cell"]]
    h24_ok = [r for r in fail if r["d_s3"] >= 0.5 * r["d_s2"]]
    h26_ok = [r for r in fail if r["d_s4"] >= 0.5 * r["d_s2"]]
    def band(n, tot):
        return "支持" if n == tot else ("部分支持" if n * 2 >= tot else "不支持")
    h25 = {m: any((r["d_s2"] - r["d_s3"]) >= 0.02 for r in fail if r["method"] == m) for m in {r["method"] for r in fail}}
    h27 = {m: sum(r["d_s4"] < r["d_s2"] for r in table if r["method"] == m) for m in {r["method"] for r in table}}
    check = dict(methods_done=sorted({r["method"] for r in table}), n_failing_cells=len(fail),
                 H24=band(len(h24_ok), len(fail)), H24_cells_ok=len(h24_ok),
                 H25_training_centring_contributes=h25, H26=band(len(h26_ok), len(fail)), H26_cells_ok=len(h26_ok),
                 H27_actions_with_dS4_below_dS2=h27)
    with open(os.path.join(OUT, "s3_s4_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in table for k in r}, key=lambda x: (x not in ("method","action"), x))); w.writeheader(); w.writerows(table)
    json.dump(check, open(os.path.join(OUT, "prereg_check_r8.json"), "w"), indent=2, ensure_ascii=False)
    import pandas as pd
    pd.set_option("display.width", 200)
    print(pd.DataFrame(table)[["method", "action", "n_seeds", "auc_s1", "d_s2", "d_s3", "d_s4", "rec_s3_k2", "rec_s4_k2", "failing_cell"]].round(3).to_string(index=False))
    print(json.dumps(check, ensure_ascii=False))


if __name__ == "__main__":
    main()
