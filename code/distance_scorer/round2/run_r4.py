"""R4: GAITEX leave-one-person O versus D2. Stops if the ratio self-check fails."""
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "/data1/users/yangrui/projects/idea67b/experiment/code")
sys.path.insert(0, "/data1/users/yangrui/projects/idea67b/experiment/code/prl_v2/round2")
import common as C
import gaitex_cells_api as G

OUT = os.path.join(C.EXP, "r4")
os.makedirs(OUT, exist_ok=True)
WANT = {"rd": 2.979155849879391, "rgs": 1.141}


def estimate_fn():
    env = {"np": np, "sq": lambda d: (np.linalg.norm(d, axis=-1) ** 2).mean()}
    return C.load_function("/data1/users/yangrui/projects/idea67b/experiment/code/gaitex_decomp.py", "estimate", env)


def ratio_of(cells, estimate):
    subs = sorted({s for (s, lab) in cells if lab == 1 and len(cells[(s, 1)]) >= 2})
    Vs, mist, _ = estimate(cells, subs)
    med = float(np.median([m["m2"] for m in mist]))
    return float(Vs / med), float(Vs), med, len(mist), len(subs)


def expand(cells):
    reps = {}
    for (sub, lab), xs in cells.items():
        for i, x in enumerate(xs):
            reps[(sub, int(lab), i)] = x
    return reps


def run_e1(reps):
    subs = sorted({k[0] for k in reps})
    def keys(sub, lab=None):
        return [k for k in reps if k[0] == sub and (lab is None or k[1] == lab)]
    oscore = {}
    for k, x in reps.items():
        own = [reps[q] for q in reps if q[0] == k[0] and q[1] == 1 and q != k]
        if own:
            oscore[k] = C.score_mse(x, np.mean(own, 0))
    per = []
    store = {"O": [], "D2": []}
    for h in subs:
        others = [s for s in subs if s != h]
        kh = keys(h)
        if not any(k[1] == 1 for k in kh):
            continue
        labs = np.array([k[1] for k in kh])
        cal = [oscore[k] for s in others for k in keys(s, 1) if k in oscore]
        tau = C.tau_of(cal)
        sh = np.array([oscore[k] for k in kh])
        dec = sh > tau
        per.append(dict(ref="O", newcomer=h, n_correct=int((labs == 1).sum()), n_error=int((labs != 1).sum()),
                        tau=float(tau), fpr=float(dec[labs == 1].mean()) if (labs == 1).any() else float("nan"),
                        tpr=float(dec[labs != 1].mean()) if (labs != 1).any() else float("nan"),
                        auc=C.auc(sh[labs != 1], sh[labs == 1])))
        store["O"].append((sh, labs))
        def mean_people(people):
            xs = [reps[k] for s in people for k in keys(s, 1)]
            return np.mean(xs, 0)
        ref_h = mean_people(others)
        cal_scores = []
        for c in others:
            rest = [s for s in others if s != c]
            if not rest:
                continue
            ref_c = mean_people(rest)
            cal_scores += [C.score_mse(reps[k], ref_c) for k in keys(c, 1)]
        tau = C.tau_of(cal_scores)
        sh = np.array([C.score_mse(reps[k], ref_h) for k in kh])
        dec = sh > tau
        per.append(dict(ref="D2", newcomer=h, n_correct=int((labs == 1).sum()), n_error=int((labs != 1).sum()),
                        tau=float(tau), fpr=float(dec[labs == 1].mean()) if (labs == 1).any() else float("nan"),
                        tpr=float(dec[labs != 1].mean()) if (labs != 1).any() else float("nan"),
                        auc=C.auc(sh[labs != 1], sh[labs == 1])))
        store["D2"].append((sh, labs))
    pooled = []
    for kind, parts in store.items():
        sc = np.concatenate([s for s, _ in parts])
        lab = np.concatenate([L for _, L in parts])
        pooled.append(dict(ref=kind, n=int(len(sc)), n_people=len(parts), auc=C.auc(sc[lab != 1], sc[lab == 1])))
    return pd.DataFrame(per), {r["ref"]: r["auc"] for r in pooled}, pooled


def plot(per, ses):
    fig, ax = plt.subplots(figsize=(6, 4))
    rng = np.random.default_rng(0)
    for i, ref in enumerate(("O", "D2")):
        sub = per[per.ref == ref]
        x = i + rng.uniform(-0.08, 0.08, len(sub))
        ax.scatter(x, sub.fpr, s=16)
    ax.axhline(0.10, ls="--", color="0.4", lw=1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["%s O" % ses, "%s D2" % ses])
    ax.set_ylabel("false positive rate")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fpr_%s.svg" % ses))
    fig.savefig(os.path.join(OUT, "fpr_%s.png" % ses), dpi=150)
    plt.close(fig)


def main():
    t0 = C.now_iso()
    t_start = time.time()
    note = {
        "script": "/data1/users/yangrui/projects/idea67b/experiment/code/gaitex_cells_api.py",
        "cache": G.CACHE,
        "wearing": "build_cells uses one ik_*.mot (the first glob hit that is not orientationErrors) and one timestamps_*.csv per subject directory. Segments in that file are the repetitions. It does not pool repeated sensor-wearing sessions beyond that one file.",
        "representation": "Observed IMU inverse-kinematics angles after dropping unobserved upper-limb coordinates. This is the series behind ledger ratios 2.979 and 1.141. There is no second position trajectory in this cache.",
    }
    json.dump(note, open(os.path.join(OUT, "source_note.json"), "w"), indent=2)
    estimate = estimate_fn()
    cells = G.build_cells(sessions=("rd", "rgs"))
    diffs = []
    ok = True
    import csv
    published = {}
    for ses, fn in (("rd", "/data1/users/yangrui/projects/idea67b/experiment/runs/gaitex/gaitex_decomp_rd_observed.csv"),
                    ("rgs", "/data1/users/yangrui/projects/idea67b/experiment/runs/gaitex/gaitex_decomp_rgs_observed.csv")):
        for row in csv.DictReader(open(fn)):
            if row["mistake"] == "(all)":
                published[ses] = float(row["ratio"])
        got, Vs, med, n_cells, n_sub = ratio_of(cells[ses], estimate)
        d = abs(got - published[ses])
        diffs.append(dict(session=ses, ratio=got, published=published[ses], abs_diff=d, equal=got == published[ses],
                          V_s=Vs, median_m2=med, n_cells=n_cells, n_subjects=n_sub))
        if got != published[ses]:
            ok = False
    json.dump({"passed": ok, "diffs": diffs}, open(os.path.join(OUT, "selfcheck.json"), "w"), indent=2)
    print("R4 SELFCHECK", ok, diffs, flush=True)
    if not ok:
        C.update_manifest(os.path.abspath(__file__), [os.path.join(OUT, "selfcheck.json"), os.path.join(OUT, "source_note.json")], t0, C.now_iso())
        sys.exit(2)
    deltas = {}
    for ses in ("rd", "rgs"):
        reps = expand(cells[ses])
        per, aucs, pooled = run_e1(reps)
        per.to_csv(os.path.join(OUT, "per_person_%s.csv" % ses), index=False)
        json.dump(pooled, open(os.path.join(OUT, "pooled_%s.json" % ses), "w"), indent=2)
        deltas[ses] = aucs["O"] - aucs["D2"]
        plot(per, ses)
        print(ses, aucs, deltas[ses], flush=True)
    h9a = "支持" if deltas["rd"] >= 0.05 and deltas["rgs"] >= 0.05 else "不支持"
    h9b = "支持" if deltas["rd"] > deltas["rgs"] else "不支持"
    if any(np.isnan(v) for v in deltas.values()):
        h9a = h9b = "无法判定"
    json.dump({"H9a": h9a, "H9b": h9b, "delta": deltas}, open(os.path.join(OUT, "prereg_check.json"), "w"), indent=2)
    t1 = C.now_iso()
    C.update_manifest(os.path.abspath(__file__),
                      [os.path.join(OUT, f) for f in
                       ["source_note.json", "selfcheck.json", "per_person_rd.csv", "per_person_rgs.csv",
                        "pooled_rd.json", "pooled_rgs.json", "fpr_rd.svg", "fpr_rd.png", "fpr_rgs.svg", "fpr_rgs.png",
                        "prereg_check.json"]],
                      t0, t1)
    print("R4 DONE", h9a, h9b, deltas, "seconds", round(time.time() - t_start, 3))


if __name__ == "__main__":
    main()
