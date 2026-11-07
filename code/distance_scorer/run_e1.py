"""E1: own reference versus demonstrator reference. Stops if the self-check fails."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/data1/users/yangrui/projects/idea67b/experiment/code/prl_v2")
import common as C
import prl_scheme12 as P

OUT = os.path.join(C.EXP, "e1")
os.makedirs(OUT, exist_ok=True)
EXPECTED_AUC = {
    ("Lunges", "positions"): 0.8862050456,
    ("Plank", "positions"): 0.9904761905,
    ("SQUAT", "positions"): 0.9675609756,
    ("Lunges", "bone angles"): 0.7630166398,
    ("Plank", "bone angles"): 0.9103896104,
    ("SQUAT", "bone angles"): 0.8485365854,
}
LEDGER_ORDER = {
    "positions": ["SQUAT", "Lunges", "Plank"],
    "bone angles": ["SQUAT", "Lunges", "Plank"],
}


def align_and_templates(reps):
    """Same selection and warp as prl_scheme12.align_all, also returning the template key."""
    aligned, templates = {}, {}
    for a in sorted({k[0] for k in reps}):
        cor = [k for k in reps if k[0] == a and k[2] == 1]
        best, bestd = None, 1e18
        for k in cor:
            d = 0.0
            for q in cor:
                if q == k:
                    continue
                d += P.align_to(reps[q], reps[k])[1]
            if d < bestd:
                bestd, best = d, k
        templates[a] = {"act": best[0], "sub": best[1], "lab": int(best[2]), "rep": int(best[3]),
                        "sum_mean_cost": float(bestd),
                        "n_correct_in_pool": len(cor),
                        "subjects_in_pool": sorted({k[1] for k in cor})}
        tmpl = reps[best]
        for k, x in reps.items():
            if k[0] == a:
                aligned[k] = P.align_to(x, tmpl)[0]
        print("aligned", a, "template", best, flush=True)
    return aligned, templates


def feats_of(aligned):
    pos = {k: P.flat(v) for k, v in aligned.items()}
    ang = {k: P.angles_of(v) for k, v in aligned.items()}
    return {"positions": pos, "bone angles": ang}


def o_table(feat):
    df = P.scores(feat)
    df["view"] = None
    return df


def rel_diff(a, b):
    return abs(a - b) / max(abs(b), 1e-30)


def main():
    t0 = C.now_iso()
    t_start = time.time()
    reps = P.load_reps()
    aligned, templates = align_and_templates(reps)
    feat = feats_of(aligned)
    lengths = {a: int(aligned[k].shape[0]) for a, k in ((a, next(k for k in aligned if k[0] == a)) for a in templates)}
    same_axis = {}
    for a in templates:
        lens = {int(aligned[k].shape[0]) for k in aligned if k[0] == a}
        same_axis[a] = {"n_lengths": len(lens), "length": sorted(lens)}
    leak = []
    subs = sorted({k[1] for k in aligned})
    for a, info in templates.items():
        leak.append({"act": a, "template_subject": info["sub"], "uses_every_subject": set(info["subjects_in_pool"]) == set(subs)})
    json.dump({"templates": templates, "same_time_axis": same_axis, "lengths": lengths,
               "alignment_leakage": True,
               "note": "align_all picks the template from every subject's correct repetitions, including the held-out person. O and D share this alignment. Not repaired."},
              open(os.path.join(OUT, "alignment.json"), "w"), indent=2, ensure_ascii=False)

    raw = pd.read_csv("/data1/users/yangrui/projects/idea67b/experiment/runs/prl_supplement/scheme1_scores_raw.csv")
    mine = []
    for view, f in feat.items():
        d = P.scores(f)
        d["view"] = view
        mine.append(d)
    mine = pd.concat(mine, ignore_index=True)
    m = mine.merge(raw, on=["act", "sub", "lab", "rep", "view"], suffixes=("_new", "_old"))
    if len(m) != len(raw) or len(m) != len(mine):
        selfcheck = {"passed": False, "reason": "row mismatch", "n_merged": int(len(m)),
                     "n_raw": int(len(raw)), "n_new": int(len(mine))}
        json.dump(selfcheck, open(os.path.join(OUT, "selfcheck.json"), "w"), indent=2)
        print("SELFCHECK FAIL rows", selfcheck)
        sys.exit(2)
    m["rel"] = [rel_diff(a, b) for a, b in zip(m.score_new, m.score_old)]
    max_rel = float(m["rel"].max())
    aucs = {}
    auc_ok = True
    for (act, view), exp in EXPECTED_AUC.items():
        sub = mine[(mine.act == act) & (mine.view == view)]
        got = C.auc(sub.loc[sub.lab != 1, "score"], sub.loc[sub.lab == 1, "score"])
        aucs["%s|%s" % (act, view)] = {"got": got, "expected": exp, "abs_diff": abs(got - exp)}
        if abs(got - exp) > 5e-11:
            auc_ok = False
    pos = mine[mine.view == "positions"]
    n_err = int((pos.lab != 1).sum())
    n_cor = int((pos.lab == 1).sum())
    passed = max_rel < 1e-9 and auc_ok and n_err == 251 and n_cor == 120 and len(raw) == 742
    selfcheck = {"passed": passed, "max_relative_diff": max_rel, "auc": aucs,
                 "n_error_positions": n_err, "n_correct_positions": n_cor, "n_score_rows": int(len(raw))}
    json.dump(selfcheck, open(os.path.join(OUT, "selfcheck.json"), "w"), indent=2)
    print("SELFCHECK", json.dumps(selfcheck), flush=True)
    if not passed:
        sys.exit(2)

    # cache alignment for later experiments
    np.savez_compressed(os.path.join(OUT, "aligned_pos.npz"),
                        **{"|".join(map(str, k)): v for k, v in aligned.items()})

    rows = []
    per = []
    subs = sorted({k[1] for k in aligned})
    # precompute O scores per view
    oscore = {}
    for view, f in feat.items():
        d = P.scores(f)
        oscore[view] = {(r.act, r.sub, int(r.lab), int(r.rep)): float(r.score) for r in d.itertuples(index=False)}

    def keys_of(act, sub, lab=None):
        out = []
        for k in aligned:
            if k[0] == act and k[1] == sub and (lab is None or int(k[2]) == lab):
                out.append(k)
        return out

    def medoid_key(act, sub):
        ks = keys_of(act, sub, 1)
        best, bestd = None, 1e300
        pos = feat["positions"]
        for k in ks:
            others = [pos[q] for q in ks if q != k]
            d = float(np.mean([np.mean((pos[k] - o) ** 2) for o in others]))
            if d < bestd:
                bestd, best = d, k
        return best

    medoids = {}
    for a in templates:
        for s in subs:
            medoids[(a, s)] = medoid_key(a, s)

    def ref_vec(view, key):
        return feat[view][key]

    def mean_correct(view, act, people):
        xs = []
        for s in people:
            for k in keys_of(act, s, 1):
                xs.append(feat[view][k])
        return np.mean(xs, 0)

    long_rows = []

    def add_eval(act, view, kind, demo, h, scores_h, tau, keys_h):
        labs = np.array([int(k[2]) for k in keys_h])
        sc = np.asarray(scores_h, float)
        dec = C.decide(sc, tau)
        cor = labs == 1
        err = ~cor
        fpr = float(dec[cor].mean()) if cor.any() else float("nan")
        tpr = float(dec[err].mean()) if err.any() else float("nan")
        ua = C.auc(sc[err], sc[cor])
        per.append(dict(act=act, view=view, ref=kind, demonstrator=demo or "", newcomer=h,
                        n_correct=int(cor.sum()), n_error=int(err.sum()), tau=tau,
                        fpr=fpr, tpr=tpr, auc=ua))
        for k, s, d in zip(keys_h, sc, dec):
            long_rows.append(dict(act=act, view=view, ref=kind, demonstrator=demo or "", newcomer=h,
                                  rep=int(k[3]), lab=int(k[2]), score=float(s), tau=float(tau),
                                  decision=int(d)))
        return sc, labs

    pooled_store = {("O", v, a): [] for v in C.VIEWS for a in templates}
    pooled_store.update({("D2", v, a): [] for v in C.VIEWS for a in templates})

    for view in C.VIEWS:
        for act in templates:
            for h in subs:
                others = [s for s in subs if s != h]
                kh = [k for k in aligned if k[0] == act and k[1] == h]
                # O
                cal = [oscore[view][(act, s, 1, int(k[3]))] for s in others for k in keys_of(act, s, 1)]
                tau = C.tau_of(cal)
                sh = [oscore[view][(act, h, int(k[2]), int(k[3]))] for k in kh]
                add_eval(act, view, "O", "", h, sh, tau, kh)
                pooled_store[("O", view, act)].append((kh, sh))
                # D2
                ref_h = mean_correct(view, act, others)
                cal_scores = []
                for c in others:
                    rest = [s for s in others if s != c]
                    ref_c = mean_correct(view, act, rest)
                    for k in keys_of(act, c, 1):
                        cal_scores.append(C.score_mse(feat[view][k], ref_c))
                tau = C.tau_of(cal_scores)
                sh = [C.score_mse(feat[view][k], ref_h) for k in kh]
                add_eval(act, view, "D2", "", h, sh, tau, kh)
                pooled_store[("D2", view, act)].append((kh, sh))
                # D1
                for p in others:
                    mk = medoids[(act, p)]
                    ref = ref_vec(view, mk)
                    cals = [s for s in subs if s not in (h, p)]
                    cal_scores = [C.score_mse(feat[view][k], ref) for s in cals for k in keys_of(act, s, 1)]
                    tau = C.tau_of(cal_scores)
                    sh = [C.score_mse(feat[view][k], ref) for k in kh]
                    add_eval(act, view, "D1", p, h, sh, tau, kh)

    per_df = pd.DataFrame(per)
    long_df = pd.DataFrame(long_rows)
    per_df.to_csv(os.path.join(OUT, "per_person.csv"), index=False)
    long_df.to_csv(os.path.join(OUT, "scores_long.csv"), index=False)

    pooled_rows = []
    for (kind, view, act), parts in pooled_store.items():
        sc, lab = [], []
        for kh, sh in parts:
            sc.extend(sh)
            lab.extend([int(k[2]) for k in kh])
        sc, lab = np.asarray(sc, float), np.asarray(lab)
        pooled_rows.append(dict(ref=kind, view=view, act=act, n=int(len(sc)),
                                n_correct=int((lab == 1).sum()), n_error=int((lab != 1).sum()),
                                auc=C.auc(sc[lab != 1], sc[lab == 1])))
    # D1 aggregate per act x view over 12 pairs
    d1 = per_df[per_df.ref == "D1"]
    for (act, view), g in d1.groupby(["act", "view"]):
        pooled_rows.append(dict(ref="D1", view=view, act=act, n=int(len(g)),
                                n_correct=int(g.n_correct.sum()), n_error=int(g.n_error.sum()),
                                auc_mean=float(g.auc.mean()), auc_min=float(g.auc.min()), auc_max=float(g.auc.max()),
                                fpr_mean=float(g.fpr.mean()), fpr_min=float(g.fpr.min()), fpr_max=float(g.fpr.max()),
                                tpr_mean=float(g.tpr.mean()), tpr_min=float(g.tpr.min()), tpr_max=float(g.tpr.max()),
                                auc=float("nan")))
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(os.path.join(OUT, "pooled.csv"), index=False)

    # rho with a2 estimate on this alignment
    estimate = C.a2_estimate()
    pairs = C.bone_pairs()
    rho_rows = []
    for view, transform in (
        ("positions", lambda X: X),
        ("bone angles", lambda X: C.angles_of_ec3d(X, pairs)),
    ):
        for act in templates:
            A = {}
            for k, X in aligned.items():
                if k[0] != act:
                    continue
                A.setdefault((k[1], int(k[2])), []).append(transform(X))
            Vs, mist, s2 = estimate(A, subs)
            med = float(np.median([m["m2"] for m in mist]))
            rho_rows.append(dict(view=view, act=act, V_s=Vs, sigma2_correct=s2, median_m2=med,
                                 rho=Vs / med, n_cells=len(mist)))
    rho = pd.DataFrame(rho_rows)
    rho.to_csv(os.path.join(OUT, "rho_realign.csv"), index=False)
    order_ok = {}
    for view, expect in LEDGER_ORDER.items():
        sub = rho[rho.view == view].sort_values("rho", ascending=False)
        got = sub.act.tolist()
        order_ok[view] = got == expect

    def cell(ref, view, act):
        r = pooled[(pooled.ref == ref) & (pooled.view == view) & (pooled.act == act)].iloc[0]
        return float(r.auc)

    deltas = []
    for view in C.VIEWS:
        for act in templates:
            d = cell("O", view, act) - cell("D2", view, act)
            deltas.append(dict(view=view, act=act, auc_O=cell("O", view, act), auc_D2=cell("D2", view, act), delta=d))
    ddf = pd.DataFrame(deltas)
    any_nan = bool(ddf.delta.isna().any())

    def both_one(pred):
        """pred(view) -> whether that ruler satisfies the claim."""
        flags = {v: bool(pred(v)) for v in C.VIEWS}
        if any_nan:
            return "无法判定", flags
        n = sum(flags.values())
        if n == 2:
            return "支持", flags
        if n == 1:
            return "部分支持", flags
        return "不支持", flags

    def arg_extreme(view, which):
        sub = ddf[ddf.view == view]
        if which == "max":
            i = sub.delta.idxmax()
        else:
            i = sub.delta.idxmin()
        top = float(sub.loc[i, "delta"])
        tied = sub.loc[np.isclose(sub.delta, top)].act.tolist()
        return tied

    h1 = "无法判定" if any_nan else ("支持" if bool((ddf.delta >= 0.05).any()) else "不支持")
    h2, h2f = both_one(lambda v: arg_extreme(v, "max") == ["SQUAT"])
    h3, h3f = both_one(lambda v: arg_extreme(v, "min") == ["Plank"])
    pos = pooled[(pooled.ref == "D2") & (pooled.view == "positions")]
    if pos.auc.isna().any():
        h4 = "无法判定"
    else:
        h4 = "支持" if pos.loc[pos.auc.idxmin(), "act"] == "SQUAT" and (pos.auc == pos.auc.min()).sum() == 1 else "不支持"
    check = {
        "H1": h1, "H2": h2, "H3": h3, "H4": h4,
        "H2_flags": h2f, "H3_flags": h3f,
        "deltas": deltas,
        "rho_order_matches_ledger": order_ok,
        "alignment_leakage": True,
        "same_time_axis": all(v["n_lengths"] == 1 for v in same_axis.values()),
    }
    json.dump(check, open(os.path.join(OUT, "prereg_check.json"), "w"), indent=2, ensure_ascii=False)
    t1 = C.now_iso()
    outs = [os.path.join(OUT, f) for f in
            ["alignment.json", "selfcheck.json", "scores_long.csv", "per_person.csv", "pooled.csv",
             "prereg_check.json", "rho_realign.csv", "aligned_pos.npz"]]
    C.update_manifest(os.path.abspath(__file__), outs, t0, t1)
    print("E1 DONE", t1, "seconds", round(time.time() - t_start, 3))
    print(json.dumps(check, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
