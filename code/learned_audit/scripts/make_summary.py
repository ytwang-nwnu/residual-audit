"""Aggregate round7 runs -> per-method outputs + summary, per PREREG_R7.

Reads runs/*/scores.csv + config.yaml + commit.txt under the results root.
Writes:
  <method>/pooled.csv, per_person.csv, recovery.csv, metrics.json, prereg_check.json
  summary/prereg_check.json, summary/s0_split_audit.json,
  summary/run_selection.json (which run dir feeds each (method, split, fold,
  seed) cell, which were voided / superseded / ignored, and why),
  summary/m1_s0_repro.json (M1 repo-split reproduction numbers).

PREREG_R7_amend1 (M4 LITEMV, M5 ConvTran): the same per-method aggregation
code path produces m4/ and m5/; H20-H23 are judged for m4/m5 only and
APPENDED to summary/prereg_check.json (keys H20..H23 + "amend1"); H16-H19
keep their original method set (m1, m2, m3p) and logic, so their blocks and
summary/run_selection.json (original methods) are unchanged. The amend1 run
selection goes to summary/run_selection_amend1.json.

Run selection rule (code, not file deletion; nothing under runs/ is removed):
  1. only dirs named YYYY-MM-DD_vNN_<method>_<split>_fold<h>_seed<s> exactly
     (e.g. the *_pilot dir is ignored) that contain config.yaml, scores.csv
     and commit.txt;
  2. runs whose commit.txt is listed in VOID_RULES for that method are void
     (reason recorded) and never used;
  3. if several runs remain for one cell, the highest version number vNN is
     used and the others are recorded as superseded.
No literal result numbers live in this file; every number is computed from
the selected runs' scores.csv.
"""
import csv, glob, json, os, re, sys
from collections import defaultdict
import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import ec3d  # noqa: E402
from metrics_r7 import auc_mw, auc_of, rates  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = yaml.safe_load(open(os.path.join(HERE, "..", "configs", "common.yaml")))
ROOT = CFG["results_root"]
DELTA_THR = 0.05   # prereg: H16/H18 threshold on delta AUC_L
REC_THR = 0.5      # prereg: H18 k=2 recovery threshold
K_STAR_THR = 0.8   # prereg: recovery k*(0.8)
EXIT_AUC = 0.6     # prereg: S2-all pooled AUC exit rule
H19_RANGE = 0.3    # prereg: detection-rate range threshold
H23_RANGE = 0.3    # amend1: FALSE-ALARM-rate range threshold (H23)

ORIG_METHODS = ("m1", "m2", "m3p")   # PREREG_R7: H16-H19
AMEND1_METHODS = ("m4", "m5")        # PREREG_R7_amend1: H20-H23

RUN_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}_v(\d+)_(m1|m2|m3p|m4|m5)_(s0|s1|s2)_fold(\d+)_seed(\d+)$")

# Runs produced by these code commits are void for the given method.
VOID_RULES = [
    dict(method="m1",
         # 49f9388 only added scripts/make_manifest.py on top of 5490935 (it was
         # committed while the first M1 matrix was still running, so its last two
         # runs carry that hash); src/m1_zhao.py is byte-identical at both.
         commits=["5490935231b2eb81f2bfd3c06f6a7fde2923bc22",
                  "49f9388e2a5f3c9ac8f4a53ca8e2e15d57825781"],
         reason=("score saturation: score_items computed 1 - exp(logp[c] - lse) in "
                 "float, which with the model's large logits collapses to exactly 0 "
                 "or 1 (1-3 unique score values per action), so AUC was dominated by "
                 "ties. Fixed in commit c902914 (within-action log-odds in log space, "
                 "a strictly monotone transform of the pre-registered 1 - p(correct); "
                 "pred unchanged). These runs also have the batch-1 BatchNorm defect "
                 "below. The whole M1 matrix was re-run.")),
    dict(method="m1",
         # c902914 (log-odds score), 2e3bf5c (run_matrix guard; runs v28-v46,
         # an interrupted batch), 1ec9d5b (pred_class column; runs v47-v73).
         commits=["c9029149c57f190b6f8a4797a84bfbabb30a8b3b",
                  "2e3bf5c8ed42a7820a406fc10a50ea93bfb70052",
                  "1ec9d5b800cc43d720082408aab4288e4c10ed65"],
         reason=("inference ran with BatchNorm skipped: score_items fed one rep at a "
                 "time and zhao22/models.py GCN_class.forward guards both BatchNorm1d "
                 "layers with `if b > 1`, while the repo tests with "
                 "batch_size=len(data_test). On un-normalized activations |log p| ~ 1e4 "
                 "(root cause of the saturation) and S0 12-class accuracy was 0.25/0.36 "
                 "(seed 0/1) vs 0.875/0.920 with full-batch inference on the same "
                 "trained models. Fixed in commit 89d7974 (one batch per condition, "
                 "score/pred formulas unchanged) and the whole M1 matrix was re-run.")),
]

# External reference value for summary/m1_s0_repro.json (NOT a computed
# result): Zhao et al. 2022 (ACCV) report 90.9% mistake identification
# accuracy on EC3D with the repo split; figure as given in the task brief.
PAPER_M1_REPORTED_ACC = 0.909


def void_reason(method, commit):
    for rule in VOID_RULES:
        if rule["method"] == method and any(commit.startswith(c) for c in rule["commits"]):
            return rule["reason"]
    return None


def parse_rows(scp):
    rows = list(csv.DictReader(open(scp)))
    for r in rows:
        r["label"] = int(r["label"])
        r["pred"] = int(r["pred"])
        r["score"] = float(r["score"])
        r["k"] = int(r["k"]) if r["k"] != "" else None
        r["draw"] = int(r["draw"]) if r["draw"] != "" else None
        if "pred_class" in r:
            r["pred_class"] = int(r["pred_class"]) if r["pred_class"] != "" else None
    return rows


def select_runs():
    """Apply the run selection rule.

    -> runs {(method, split, fold, seed): {run_dir, rows, subjects_test, commit, version}},
       selection {selected, voided, superseded, ignored}
    """
    candidates = defaultdict(list)
    sel = dict(rule=(select_runs.__doc__ or "").strip(),
               selected={}, voided=[], superseded=[], ignored=[])
    for rd in sorted(glob.glob(os.path.join(ROOT, "runs", "*"))):
        name = os.path.basename(rd)
        m = RUN_NAME.match(name)
        cfgp, scp, cmp_ = (os.path.join(rd, "config.yaml"), os.path.join(rd, "scores.csv"),
                           os.path.join(rd, "commit.txt"))
        if not m:
            sel["ignored"].append(dict(run=name, why="name does not match the run pattern"))
            continue
        if not (os.path.exists(cfgp) and os.path.exists(scp) and os.path.exists(cmp_)):
            sel["ignored"].append(dict(run=name, why="missing config.yaml / scores.csv / commit.txt"))
            continue
        c = yaml.safe_load(open(cfgp))
        commit = open(cmp_).read().strip()
        key = (c["method"], c["split"], int(c["fold"]), int(c["seed"]))
        assert key == (m.group(2), m.group(3), int(m.group(4)), int(m.group(5))), \
            "dir name / config mismatch: " + name
        vr = void_reason(c["method"], commit)
        if vr is not None:
            sel["voided"].append(dict(run=name, commit=commit, reason=vr))
            continue
        candidates[key].append(dict(run_dir=rd, name=name, version=int(m.group(1)), commit=commit))
    runs = {}
    for key, lst in candidates.items():
        lst.sort(key=lambda d: d["version"])
        chosen = lst[-1]
        for other in lst[:-1]:
            sel["superseded"].append(dict(
                run=other["name"], commit=other["commit"],
                superseded_by=chosen["name"],
                why="a later version exists for the same (method, split, fold, seed)"))
        rows = parse_rows(os.path.join(chosen["run_dir"], "scores.csv"))
        runs[key] = dict(run_dir=chosen["run_dir"], rows=rows, commit=chosen["commit"],
                         version=chosen["version"],
                         subjects_test=sorted({r["person"] for r in rows}))
        sel["selected"]["|".join(map(str, key))] = dict(run=chosen["name"], commit=chosen["commit"])
    sel["void_rules"] = VOID_RULES
    return runs, sel


def _is_amend1_run(name):
    return any("_{}_".format(m) in name for m in AMEND1_METHODS)


def split_selection(sel):
    """Partition the selection record into (original methods, amend1 methods)
    by run name, preserving entry order, so summary/run_selection.json keeps
    exactly its PREREG_R7 content and the m4/m5 record goes to
    summary/run_selection_amend1.json."""
    orig = dict(rule=sel["rule"], selected={}, voided=[], superseded=[], ignored=[])
    am1 = dict(rule=sel["rule"], methods=list(AMEND1_METHODS),
               selected={}, voided=[], superseded=[], ignored=[])
    for key, v in sel["selected"].items():
        (am1 if key.split("|")[0] in AMEND1_METHODS else orig)["selected"][key] = v
    for lst in ("voided", "superseded", "ignored"):
        for e in sel[lst]:
            (am1 if _is_amend1_run(e["run"]) else orig)[lst].append(e)
    orig["void_rules"] = sel["void_rules"]
    am1["void_rules"] = [r for r in sel["void_rules"] if r["method"] in AMEND1_METHODS]
    return orig, am1


def rows_of(run_rows, split_name, act=None, k=None, draw=None):
    out = [r for r in run_rows if r["split"] == split_name]
    if act is not None:
        out = [r for r in out if r["exercise"] == act]
    if k is not None:
        out = [r for r in out if r["k"] == k]
    if draw is not None:
        out = [r for r in out if r["draw"] == draw]
    return out


def mean(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.mean(x)) if x else float("nan")


def s0_repro(runs, segs):
    """M1 repo-split (S0) reproduction: 12-class accuracy (pred_class vs the
    pickle's fine label) and binary error/correct accuracy per seed, plus the
    paper's reported figure for reference."""
    import m1_zhao  # fine-label -> 12-class mapping (torch import only)
    out = dict(
        note=("M1 S0 = repo split (train subjects 0-2, test subject 3 in sorted order); "
              "acc12 = fraction of test reps whose 12-class argmax equals the fine "
              "label class; acc_binary = fraction whose error/correct decision (argmax "
              "!= action's correct class) equals the binary label."),
        paper_reported_mistake_identification_acc=PAPER_M1_REPORTED_ACC,
        per_seed={}, runs={})
    lab_of = {s["id"]: (s["act"], s["lab"]) for s in segs}
    for (m, split, f, seed), pack in sorted(runs.items()):
        if m != "m1" or split != "s0":
            continue
        rows = rows_of(pack["rows"], "s0")
        test_subs = sorted({r["person"] for r in rows})
        acc_bin = mean([float(r["pred"] == r["label"]) for r in rows])
        has_cls = all(r.get("pred_class") is not None for r in rows)
        acc12 = (mean([float(r["pred_class"] == m1_zhao.class_of(*lab_of[r["id"]])) for r in rows])
                 if has_cls else None)
        per_act = {}
        for act in sorted({r["exercise"] for r in rows}):
            ra = [r for r in rows if r["exercise"] == act]
            per_act[act] = dict(
                n=len(ra),
                acc12=(mean([float(r["pred_class"] == m1_zhao.class_of(*lab_of[r["id"]])) for r in ra])
                       if has_cls else None),
                acc_binary=mean([float(r["pred"] == r["label"]) for r in ra]),
                auc=auc_of(ra))
        out["per_seed"][str(seed)] = dict(n_test=len(rows), test_subjects=test_subs,
                                          acc12=acc12, acc_binary=acc_bin, per_action=per_act)
        out["runs"][str(seed)] = os.path.basename(pack["run_dir"])
    if out["per_seed"]:
        out["mean_over_seeds"] = dict(
            acc12=mean([v["acc12"] for v in out["per_seed"].values()]),
            acc_binary=mean([v["acc_binary"] for v in out["per_seed"].values()]))
        out["gap_to_paper"] = dict(
            acc12_minus_paper=out["mean_over_seeds"]["acc12"] - PAPER_M1_REPORTED_ACC
            if np.isfinite(out["mean_over_seeds"]["acc12"]) else None,
            acc_binary_minus_paper=out["mean_over_seeds"]["acc_binary"] - PAPER_M1_REPORTED_ACC)
    return out


def verdict_of(n_true, n_methods):
    """PREREG_R7 / amend1 verdict scheme: all -> 支持, all but one -> 部分支持."""
    return ("支持" if n_true == n_methods
            else "部分支持" if n_true == n_methods - 1
            else "不支持")


def main():
    runs, selection = select_runs()
    selection, selection_amend1 = split_selection(selection)
    methods_all = sorted({k[0] for k in runs})
    # PREREG_R7 method set (H16-H19) and amend1 method set (H20-H23), kept apart
    methods = [m for m in methods_all if m in ORIG_METHODS]
    methods_amend1 = [m for m in methods_all if m in AMEND1_METHODS]
    seeds = sorted({k[3] for k in runs if k[0] in ORIG_METHODS})
    segs = ec3d.load_segments(CFG["data_pickle"])
    acts = ec3d.actions(segs)
    subs = ec3d.subjects(segs)
    print("methods:", methods, "amend1 methods:", methods_amend1, "seeds:", seeds,
          "acts:", acts, "subs:", subs)
    print("selected runs:", len(runs), "voided:", len(selection["voided"]),
          "superseded:", len(selection["superseded"]), "ignored:", len(selection["ignored"]),
          "| amend1 selected:", len(selection_amend1["selected"]),
          "ignored:", len(selection_amend1["ignored"]))

    summary = {"H16": {}, "H17": {}, "H18": {}, "H19": {}, "exited": {}}
    amend1 = {"methods": methods_amend1, "exited": {}}
    s0_audit = {}
    per_method = {}

    for m in methods + methods_amend1:
        folds = sorted({k[2] for k in runs if k[0] == m})
        m_seeds = sorted({k[3] for k in runs if k[0] == m})
        m_acts = acts

        # per-method row buffers (kept inside per_method[m]; the earlier version
        # re-created these lists per method but wrote them after the loop, so
        # only the last method's csvs were non-empty)
        pooled_rows, person_rows, rec_rows = [], [], []
        metrics = {"method": m, "seeds": m_seeds, "acts": {},
                   "runs_used": sorted(os.path.basename(v["run_dir"]) for k, v in runs.items() if k[0] == m)}

        for act in m_acts:
            auc_s1, auc_s2all = {}, {}
            for seed in m_seeds:
                r1, r2 = [], []
                for f in folds:
                    if (m, "s1", f, seed) in runs:
                        r1 += rows_of(runs[(m, "s1", f, seed)]["rows"], "s1", act)
                    if (m, "s2", f, seed) in runs:
                        r2 += rows_of(runs[(m, "s2", f, seed)]["rows"], "s2_all", act)
                auc_s1[seed] = auc_of(r1)
                auc_s2all[seed] = auc_of(r2)
                pooled_rows.append([m, "s1", act, seed, auc_s1[seed], len(r1)])
                pooled_rows.append([m, "s2_all", act, seed, auc_s2all[seed], len(r2)])
            d_auc = {s: auc_s2all[s] - auc_s1[s] for s in m_seeds}
            metrics["acts"][act] = dict(
                auc_s1_per_seed=auc_s1, auc_s2all_per_seed=auc_s2all,
                auc_s1=mean(auc_s1.values()), auc_s2all=mean(auc_s2all.values()),
                delta_auc_L_per_seed=d_auc, delta_auc_L=mean(d_auc.values()))

        # s0 pooled (m1 only)
        for seed in m_seeds:
            for f in folds:
                if (m, "s0", f, seed) in runs:
                    for act in m_acts:
                        r = rows_of(runs[(m, "s0", f, seed)]["rows"], "s0", act)
                        pooled_rows.append([m, "s0", act, seed, auc_of(r), len(r)])

        # per-person metrics + working point (s1 and s2_all; s0 too if present)
        wp = defaultdict(dict)  # wp[(split, act, seed)][person] = (tpr,fpr)
        for split_name, tag in (("s1", "s1"), ("s2_all", "s2"), ("s0", "s0")):
            for seed in m_seeds:
                for f in folds:
                    if (m, tag, f, seed) not in runs:
                        continue
                    for act in m_acts:
                        for sub in subs:
                            r = [x for x in rows_of(runs[(m, tag, f, seed)]["rows"], split_name, act)
                                 if x["person"] == sub]
                            if not r:
                                continue
                            tpr, fpr = rates(r)
                            wp[(split_name, act, seed)][sub] = (tpr, fpr)
                            person_rows.append([m, split_name, act, sub, seed,
                                                sum(1 for x in r if x["label"] == 0),
                                                sum(1 for x in r if x["label"] == 1),
                                                auc_of(r), tpr, fpr])

        # ranges over the 4 held-out persons
        for act in m_acts:
            for split_name in ("s1", "s2_all"):
                ranges_tpr, ranges_fpr = [], []
                for seed in m_seeds:
                    d = wp.get((split_name, act, seed), {})
                    if len(d) == 4:
                        tprs = [v[0] for v in d.values()]
                        fprs = [v[1] for v in d.values()]
                        ranges_tpr.append(max(tprs) - min(tprs))
                        ranges_fpr.append(max(fprs) - min(fprs))
                metrics["acts"][act]["tpr_range_" + split_name] = mean(ranges_tpr)
                metrics["acts"][act]["fpr_range_" + split_name] = mean(ranges_fpr)

        # recovery: per (act, k, seed, draw)
        for act in m_acts:
            rec_by_k = {}
            for k in (1, 2, 4):
                per_seed = {}
                for seed in m_seeds:
                    vals = []
                    # draws live in s2 runs; assume same draw count across folds
                    n_draws = 0
                    for f in folds:
                        if (m, "s2", f, seed) in runs:
                            dr = {r["draw"] for r in rows_of(runs[(m, "s2", f, seed)]["rows"], "s2_k%d" % k, act)}
                            n_draws = max(n_draws, len(dr))
                    for d_i in range(n_draws):
                        s2k_ids, s2k_rows = set(), []
                        s1_rows, s2a_rows = [], []
                        for f in folds:
                            if (m, "s2", f, seed) not in runs:
                                continue
                            rr = runs[(m, "s2", f, seed)]["rows"]
                            part = rows_of(rr, "s2_k%d" % k, act, draw=d_i)
                            s2k_rows += part
                            s2k_ids |= {x["id"] for x in part}
                            if (m, "s1", f, seed) in runs:
                                s1_rows += [x for x in rows_of(runs[(m, "s1", f, seed)]["rows"], "s1", act)
                                            if x["id"] in s2k_ids]
                            s2a_rows += [x for x in rows_of(rr, "s2_all", act)
                                         if x["id"] in s2k_ids]
                        a1 = auc_of(s1_rows)
                        a2k = auc_of(s2k_rows)
                        a2a = auc_of(s2a_rows)
                        den = a2a - a1
                        rec = (a2k - a1) / den if den != 0 else float("nan")
                        vals.append(rec)
                        rec_rows.append([m, act, k, seed, d_i, rec])
                    per_seed[seed] = mean(vals)
                rec_by_k[k] = mean(per_seed.values())
                metrics["acts"][act].setdefault("recovery_per_seed", {})[k] = per_seed
                metrics["acts"][act].setdefault("recovery", {})[k] = rec_by_k[k]
            ks = sorted(rec_by_k)
            k_star = next((k for k in ks if rec_by_k[k] >= K_STAR_THR), ">4")
            metrics["acts"][act]["k_star_08"] = k_star

        # ---- prereg checks per method ----
        acts_d = {a: metrics["acts"][a]["delta_auc_L"] for a in m_acts}
        s2all_auc = {a: metrics["acts"][a]["auc_s2all"] for a in m_acts}
        exited = all(v < EXIT_AUC for v in s2all_auc.values())

        squat_d = acts_d.get("SQUAT", float("nan"))
        plank_d = acts_d.get("Plank", float("nan"))
        lunge_d = acts_d.get("Lunges", float("nan"))
        qual_acts = [a for a, v in acts_d.items() if v >= DELTA_THR]

        if m in ORIG_METHODS:
            # PREREG_R7 H16-H19 (unchanged)
            h16 = any(v >= DELTA_THR for v in acts_d.values())
            h17 = (squat_d >= lunge_d and squat_d >= plank_d
                   and plank_d <= lunge_d and plank_d <= squat_d)
            h18 = bool(qual_acts) and all(
                metrics["acts"][a]["recovery"][2] >= REC_THR for a in qual_acts)
            h19_acts = [a for a in m_acts
                        if metrics["acts"][a]["tpr_range_s1"] >= H19_RANGE
                        and metrics["acts"][a]["tpr_range_s2_all"] < metrics["acts"][a]["tpr_range_s1"]]
            h19 = bool(h19_acts)

            check = dict(
                exited=exited, s2all_auc=s2all_auc,
                H16=dict(satisfied=h16, delta_auc_L=acts_d, thr=DELTA_THR),
                H17=dict(satisfied=(None if exited else h17),
                         order=dict(SQUAT=squat_d, Lunges=lunge_d, Plank=plank_d)),
                H18=dict(satisfied=(None if exited else h18), qualifying_acts=qual_acts,
                         recovery_k2={a: metrics["acts"][a]["recovery"][2] for a in qual_acts},
                         thr=REC_THR),
                H19=dict(satisfied=(None if exited else h19), qualifying_acts=h19_acts,
                         tpr_range_s1={a: metrics["acts"][a]["tpr_range_s1"] for a in m_acts},
                         tpr_range_s2_all={a: metrics["acts"][a]["tpr_range_s2_all"] for a in m_acts},
                         thr=H19_RANGE),
                runs_used=metrics["runs_used"],
            )
            metrics["exit_rule"] = dict(
                triggered=exited,
                rule="all 3 acts s2_all pooled AUC < %.2f -> exits H17/H18/H19, H16 counted" % EXIT_AUC)
            summary["exited"][m] = exited
        else:
            # PREREG_R7_amend1 H20-H23 (m4, m5 only). H20-H22 mirror H16-H18;
            # H23 uses the FALSE-ALARM-rate (fpr) range across the 4 held-out
            # persons at the S1 working point, shrinking under S2-all.
            h20 = any(v >= DELTA_THR for v in acts_d.values())
            h21 = (squat_d >= lunge_d and squat_d >= plank_d
                   and plank_d <= lunge_d and plank_d <= squat_d)
            h22 = bool(qual_acts) and all(
                metrics["acts"][a]["recovery"][2] >= REC_THR for a in qual_acts)
            h23_acts = [a for a in m_acts
                        if metrics["acts"][a]["fpr_range_s1"] >= H23_RANGE
                        and metrics["acts"][a]["fpr_range_s2_all"] < metrics["acts"][a]["fpr_range_s1"]]
            h23 = bool(h23_acts)

            check = dict(
                exited=exited, s2all_auc=s2all_auc,
                H20=dict(satisfied=h20, delta_auc_L=acts_d, thr=DELTA_THR),
                H21=dict(satisfied=(None if exited else h21),
                         order=dict(SQUAT=squat_d, Lunges=lunge_d, Plank=plank_d)),
                H22=dict(satisfied=(None if exited else h22), qualifying_acts=qual_acts,
                         recovery_k2={a: metrics["acts"][a]["recovery"][2] for a in qual_acts},
                         thr=REC_THR),
                H23=dict(satisfied=(None if exited else h23), qualifying_acts=h23_acts,
                         fpr_range_s1={a: metrics["acts"][a]["fpr_range_s1"] for a in m_acts},
                         fpr_range_s2_all={a: metrics["acts"][a]["fpr_range_s2_all"] for a in m_acts},
                         thr=H23_RANGE),
                runs_used=metrics["runs_used"],
            )
            metrics["exit_rule"] = dict(
                triggered=exited,
                rule="all 3 acts s2_all pooled AUC < %.2f -> exits H21/H22/H23, H20 counted" % EXIT_AUC)
            amend1["exited"][m] = exited
        per_method[m] = dict(metrics=metrics, check=check, pooled_rows=pooled_rows,
                             person_rows=person_rows, rec_rows=rec_rows)

    # ---- write per-method outputs (csvs, metrics.json and prereg_check.json
    #      all come from the same per_method[m] pack, i.e. the same scores) ----
    for m, pack in per_method.items():
        od = os.path.join(ROOT, m)
        os.makedirs(od, exist_ok=True)
        with open(os.path.join(od, "pooled.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "split", "act", "seed", "pooled_auc", "n"])
            w.writerows(pack["pooled_rows"])
        with open(os.path.join(od, "per_person.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "split", "act", "person", "seed",
                        "n_correct", "n_error", "auc_within", "tpr", "fpr"])
            w.writerows(pack["person_rows"])
        with open(os.path.join(od, "recovery.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["method", "act", "k", "seed", "draw", "recovery"])
            w.writerows(pack["rec_rows"])
        json.dump(pack["metrics"], open(os.path.join(od, "metrics.json"), "w"),
                  indent=2, default=str)
        json.dump(pack["check"], open(os.path.join(od, "prereg_check.json"), "w"),
                  indent=2, default=str)
        # consistency guard: csv rows and metrics must agree on n_rows
        assert len(pack["pooled_rows"]) > 0, "no pooled rows for " + m

    # ---- summary prereg check (PREREG_R7: H16-H19 over m1, m2, m3p) ----
    for hname in ("H16", "H17", "H18", "H19"):
        sat = {m: per_method[m]["check"][hname]["satisfied"] for m in methods}
        n_true = sum(1 for v in sat.values() if v is True)
        verdict = verdict_of(n_true, len(methods))
        summary[hname] = dict(per_method=sat, n_satisfied=n_true,
                              n_methods=len(methods), verdict=verdict,
                              detail={m: per_method[m]["check"][hname] for m in methods})
    summary["methods"] = methods
    summary["seeds"] = seeds
    summary["seeds_per_method"] = {m: per_method[m]["metrics"]["seeds"] for m in methods}
    # ---- appended: PREREG_R7_amend1 H20-H23 over m4, m5 (only when runs exist) ----
    amend1_h = ("H20", "H21", "H22", "H23")
    if methods_amend1:
        for hname in amend1_h:
            sat = {m: per_method[m]["check"][hname]["satisfied"] for m in methods_amend1}
            n_true = sum(1 for v in sat.values() if v is True)
            summary[hname] = dict(per_method=sat, n_satisfied=n_true,
                                  n_methods=len(methods_amend1),
                                  verdict=verdict_of(n_true, len(methods_amend1)),
                                  detail={m: per_method[m]["check"][hname] for m in methods_amend1})
        amend1["seeds"] = sorted({k[3] for k in runs if k[0] in AMEND1_METHODS})
        amend1["seeds_per_method"] = {m: per_method[m]["metrics"]["seeds"] for m in methods_amend1}
        amend1["note"] = ("H20-H23 per PREREG_R7_amend1, judged over m4/m5 only; H23 uses the "
                          "false-alarm-rate range (H19 used the detection-rate range). "
                          "H16-H19 above keep the PREREG_R7 method set and are unchanged.")
        summary["amend1"] = amend1
    os.makedirs(os.path.join(ROOT, "summary"), exist_ok=True)
    json.dump(summary, open(os.path.join(ROOT, "summary", "prereg_check.json"), "w"),
              indent=2, default=str, ensure_ascii=False)
    json.dump(selection, open(os.path.join(ROOT, "summary", "run_selection.json"), "w"),
              indent=2, default=str, ensure_ascii=False)
    if methods_amend1:
        json.dump(selection_amend1,
                  open(os.path.join(ROOT, "summary", "run_selection_amend1.json"), "w"),
                  indent=2, default=str, ensure_ascii=False)

    # ---- s0 audit ----
    for (m, split, f, seed), pack in runs.items():
        if split != "s0":
            continue
        test_subs = pack["subjects_test"]
        train_subs = sorted(set(subs) - set(test_subs))
        s0_audit.setdefault(m, dict(
            train_subjects=train_subs, test_subjects=test_subs,
            n_runs=0, overlap=[], note=""))["n_runs"] += 1
    for m, a in s0_audit.items():
        a["overlap"] = sorted(set(a["train_subjects"]) & set(a["test_subjects"]))
        a["test_subject_in_train"] = bool(a["overlap"])
        a["note"] = ("repo split sets=[[0,1,2],[],[3]] -> test subject not in train"
                     if not a["overlap"] else "test subjects appear in train set")
    json.dump(s0_audit, open(os.path.join(ROOT, "summary", "s0_split_audit.json"), "w"),
              indent=2, ensure_ascii=False)

    # ---- m1 S0 reproduction ----
    rep = s0_repro(runs, segs)
    json.dump(rep, open(os.path.join(ROOT, "summary", "m1_s0_repro.json"), "w"),
              indent=2, default=str, ensure_ascii=False)
    print("summary written:", {h: summary[h]["verdict"] for h in ("H16", "H17", "H18", "H19")})
    if methods_amend1:
        print("amend1 summary written:", {h: summary[h]["verdict"] for h in amend1_h})


if __name__ == "__main__":
    main()
