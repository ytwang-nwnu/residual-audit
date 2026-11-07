"""R2: 27 bone-pair angles. Stops if the 23-pair self-check fails."""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "/data1/users/yangrui/projects/idea67b/experiment/code")
sys.path.insert(0, "/data1/users/yangrui/projects/idea67b/experiment/code/prl_v2/round2")
import common as C
import prl_scheme12 as P
import prl_scheme2b as S2
from ec3d_common import align_to, dtw, load_reps, root_center

OUT = os.path.join(C.EXP, "r2")
os.makedirs(OUT, exist_ok=True)
CONDS = ("bilateral_50", "unilateral_50", "bilateral_10", "trunk_50")
DEGS = (5, 10, 15)
E1B_DEGS = (5, 10, 15, 20, 30)


def joint_names_ok():
    src = open("/data1/users/yangrui/projects/idea67b/experiment/code/ec3d_common.py", encoding="utf-8").read()
    need = ["1 Neck", "2 RSho", "3 RElb", "5 LSho", "6 LElb", "8 MidHip", "9 RHip", "10 RKnee", "12 LHip", "13 LKnee"]
    return {s: s in src for s in need}


def mechanism_note():
    return {
        "bilateral_50": "prl_scheme2b.apply_cond: bend_knee on the right (hip 9, knee 10, ankle 11, children 11,17,18) then the left (hip 12, knee 13, ankle 14, children 14,15,16). Axis is the cross product of (knee-hip) and (ankle-knee). Angle is the requested degrees. Frames from 0.25 to 0.75 of the repetition.",
        "trunk_50": "tilt_trunk: joints 0 through 7 rotate about the axis (left hip 12 minus right hip 9), origin at mid-hip 8, by the requested degrees, on frames from 0.25 to 0.75.",
    }


def fit_k(degs, Es):
    x = np.asarray(degs, float) ** 2
    y = np.asarray(Es, float)
    return float(np.dot(x, y) / np.dot(x, x))


def both_one(flags):
    n = sum(bool(v) for v in flags.values())
    if n == 2:
        return "支持"
    if n == 1:
        return "部分支持"
    return "不支持"


def load_aligned():
    z = np.load("/data1/users/yangrui/projects/idea67b/experiment/runs/prl_v2/e1/aligned_pos.npz")
    out = {}
    for key in z.files:
        act, sub, lab, rep = key.split("|")
        out[(act, sub, int(lab), int(rep))] = z[key]
    return out


def e1_angle(aligned, ang):
    subs = sorted({k[1] for k in aligned})
    acts = sorted({k[0] for k in aligned})
    pos = {k: v.reshape(len(v), -1) for k, v in aligned.items()}

    def keys(act, sub, lab=None):
        return [k for k in aligned if k[0] == act and k[1] == sub and (lab is None or k[2] == lab)]

    def medoid_key(act, sub):
        ks = keys(act, sub, 1)
        best, bestd = None, 1e300
        for k in ks:
            others = [pos[q] for q in ks if q != k]
            d = float(np.mean([np.mean((pos[k] - o) ** 2) for o in others]))
            if d < bestd:
                bestd, best = d, k
        return best

    def mean_correct(act, people):
        xs = [ang[k] for s in people for k in keys(act, s, 1)]
        return np.mean(xs, 0)

    oscore = {}
    for k, x in ang.items():
        own = [ang[q] for q in ang if q[0] == k[0] and q[1] == k[1] and q[2] == 1 and q != k]
        if not own:
            continue
        oscore[k] = C.score_mse(x, np.mean(own, 0))

    per, pooled = [], []
    for act in acts:
        store = {"O": [], "D2": []}
        for h in subs:
            others = [s for s in subs if s != h]
            kh = keys(act, h)
            cal = [oscore[k] for s in others for k in keys(act, s, 1)]
            tau = C.tau_of(cal)
            sh = np.array([oscore[k] for k in kh])
            labs = np.array([k[2] for k in kh])
            dec = sh > tau
            per.append(dict(act=act, ref="O", demonstrator="", newcomer=h, n_correct=int((labs == 1).sum()),
                            n_error=int((labs != 1).sum()), tau=tau,
                            fpr=float(dec[labs == 1].mean()), tpr=float(dec[labs != 1].mean()),
                            auc=C.auc(sh[labs != 1], sh[labs == 1])))
            store["O"].append((sh, labs))
            ref_h = mean_correct(act, others)
            cal_scores = []
            for c in others:
                rest = [s for s in others if s != c]
                ref_c = mean_correct(act, rest)
                cal_scores += [C.score_mse(ang[k], ref_c) for k in keys(act, c, 1)]
            tau = C.tau_of(cal_scores)
            sh = np.array([C.score_mse(ang[k], ref_h) for k in kh])
            dec = sh > tau
            per.append(dict(act=act, ref="D2", demonstrator="", newcomer=h, n_correct=int((labs == 1).sum()),
                            n_error=int((labs != 1).sum()), tau=tau,
                            fpr=float(dec[labs == 1].mean()), tpr=float(dec[labs != 1].mean()),
                            auc=C.auc(sh[labs != 1], sh[labs == 1])))
            store["D2"].append((sh, labs))
            for p in others:
                mk = medoid_key(act, p)
                ref = ang[mk]
                cals = [s for s in subs if s not in (h, p)]
                cal_scores = [C.score_mse(ang[k], ref) for s in cals for k in keys(act, s, 1)]
                tau = C.tau_of(cal_scores)
                sh = np.array([C.score_mse(ang[k], ref) for k in kh])
                dec = sh > tau
                per.append(dict(act=act, ref="D1", demonstrator=p, newcomer=h, n_correct=int((labs == 1).sum()),
                                n_error=int((labs != 1).sum()), tau=tau,
                                fpr=float(dec[labs == 1].mean()), tpr=float(dec[labs != 1].mean()),
                                auc=C.auc(sh[labs != 1], sh[labs == 1])))
        for kind, parts in store.items():
            sc = np.concatenate([s for s, _ in parts])
            lab = np.concatenate([L for _, L in parts])
            pooled.append(dict(act=act, ref=kind, n=int(len(sc)), auc=C.auc(sc[lab != 1], sc[lab == 1])))
    return pd.DataFrame(per), pd.DataFrame(pooled)


def verdicts(pos_auc, ang_auc):
    """pos_auc / ang_auc: dict (ref, act) -> auc"""
    deltas = []
    for view, src in (("positions", pos_auc), ("bone angles", ang_auc)):
        for act in ("Lunges", "Plank", "SQUAT"):
            d = src[("O", act)] - src[("D2", act)]
            deltas.append(dict(view=view, act=act, auc_O=src[("O", act)], auc_D2=src[("D2", act)], delta=d))
    ddf = pd.DataFrame(deltas)

    def tied(view, which):
        sub = ddf[ddf.view == view]
        val = sub.delta.max() if which == "max" else sub.delta.min()
        return sub.loc[np.isclose(sub.delta, val), "act"].tolist()

    flags2 = {v: tied(v, "max") == ["SQUAT"] for v in ("positions", "bone angles")}
    flags3 = {v: tied(v, "min") == ["Plank"] for v in ("positions", "bone angles")}
    pos = [pos_auc[("D2", a)] for a in ("Lunges", "Plank", "SQUAT")]
    h4 = "支持" if pos_auc[("D2", "SQUAT")] == min(pos) and pos.count(min(pos)) == 1 else "不支持"
    return {
        "H1": "支持" if bool((ddf.delta >= 0.05).any()) else "不支持",
        "H2": both_one(flags2),
        "H3": both_one(flags3),
        "H4": h4,
        "deltas": deltas,
    }


def rho_27(pairs):
    from ec3d_common import LABEL_NAME
    env = {"np": np, "LABEL_NAME": LABEL_NAME, "sq": lambda d: (np.linalg.norm(d, axis=-1) ** 2).mean()}
    estimate = C.load_function("/data1/users/yangrui/projects/idea67b/experiment/code/a2_ec3d_decomp_v2.py", "estimate", env)
    reps = load_reps()
    R = {k: root_center(v) for k, v in reps.items()}
    acts = sorted({k[0] for k in R})
    subs = sorted({k[1] for k in R})
    rows = []
    for a in acts:
        allc = [k for k in R if k[0] == a and k[2] == 1]
        D = np.zeros((len(allc), len(allc)))
        for i, j in __import__("itertools").combinations(range(len(allc)), 2):
            D[i, j] = D[j, i] = dtw(R[allc[i]], R[allc[j]])[0]
        tmpl = R[allc[int(D.sum(1).argmin())]]
        al = {k: align_to(R[k], tmpl) for k in R if k[0] == a}
        for view, take in (
            ("bone angles, all joints", lambda X: C.angles_27(X, pairs)[..., None]),
            ("bone angles, distal only", lambda X: C.angles_23(X, pairs)[:, [i for i, (j, u, v) in enumerate(pairs) if j in C.DISTAL]][..., None]),
        ):
            A = {}
            for k, X in al.items():
                A.setdefault((k[1], int(k[2])), []).append(take(X))
            Vs, mist, _ = estimate(A, subs)
            med = float(np.median([m["m2"] for m in mist]))
            rows.append(dict(act=a, view=view, V_s=float(Vs), median_m2=med, ratio=float(Vs / med), n_cells=len(mist)))
        print("rho", a, flush=True)
    return rows


def main():
    t0 = C.now_iso()
    t_start = time.time()
    names = joint_names_ok()
    json.dump({"joint_name_strings_found": names, "mechanism": mechanism_note(),
               "extra_pairs": [{"i0": a, "i1": b, "j0": c, "j1": d} for a, b, c, d in C.EXTRA],
               "distal_joints": C.DISTAL,
               "new_pairs_are_distal": False,
               "distal_rule": "A new pair is distal only if every endpoint of both bones is in the distal joint list. None of the four new bones qualify, so the distal set stays the original 23-pair distal subset."},
              open(os.path.join(OUT, "definition.json"), "w"), indent=2)
    if not all(names.values()):
        json.dump({"passed": False, "joint_names": names}, open(os.path.join(OUT, "selfcheck.json"), "w"), indent=2)
        sys.exit(2)

    aligned = load_aligned()
    pairs = C.bone_pairs_23()
    # self-check on one trajectory and then a sample of five
    keys = list(aligned)[:5]
    max_diff = 0.0
    for k in keys:
        a = P.angles_of(aligned[k])
        b = C.angles_23(aligned[k], pairs)
        max_diff = max(max_diff, float(np.max(np.abs(a - b))))
    passed = max_diff < 1e-9
    json.dump({"passed": passed, "max_abs_diff_23": max_diff, "n_checked": len(keys)},
              open(os.path.join(OUT, "selfcheck.json"), "w"), indent=2)
    print("R2 SELFCHECK", max_diff, passed, flush=True)
    if not passed:
        sys.exit(2)

    ang = {k: C.angles_27(v, pairs) for k, v in aligned.items()}
    # equivalence: 23-pair O pooled AUC versus saved E1
    ang23 = {k: C.angles_23(v, pairs) for k, v in aligned.items()}
    _, pooled23 = e1_angle(aligned, ang23)
    saved = pd.read_csv("/data1/users/yangrui/projects/idea67b/experiment/runs/prl_v2/e1/pooled.csv")
    eq = []
    eq_ok = True
    for _, r in pooled23.iterrows():
        s = saved[(saved.ref == r.ref) & (saved.view == "bone angles") & (saved.act == r.act)]
        if r.ref not in ("O", "D2") or len(s) != 1:
            continue
        d = abs(float(r.auc) - float(s.auc.iloc[0]))
        eq.append(dict(ref=r.ref, act=r.act, abs_diff=d))
        if d >= 1e-9:
            eq_ok = False
    sc = json.load(open(os.path.join(OUT, "selfcheck.json")))
    sc["protocol_equivalence_23"] = eq
    sc["protocol_equivalence_passed"] = eq_ok
    json.dump(sc, open(os.path.join(OUT, "selfcheck.json"), "w"), indent=2)
    print("R2 EQUIV", eq_ok, eq, flush=True)
    if not eq_ok:
        sys.exit(2)

    # E4 k on 27 angles
    rows = []
    order = []
    for k in aligned:
        if k[0] == "SQUAT" and k[2] == 1 and k[1] not in order:
            order.append(k[1])
    by = {}
    for k, X in aligned.items():
        if k[0] == "SQUAT" and k[2] == 1:
            by.setdefault(k[1], []).append(X)
    for cond in CONDS:
        for deg in DEGS:
            for sub in order:
                xs = by[sub]
                bent = [S2.apply_cond(X, cond, deg) for X in xs]
                mu_c = np.mean([C.angles_27(X, pairs) for X in xs], 0)
                mu_b = np.mean([C.angles_27(Y, pairs) for Y in bent], 0)
                rows.append(dict(sub=sub, condition=cond, deg=deg, n=len(xs), E=float(np.mean((mu_b - mu_c) ** 2))))
    edf = pd.DataFrame(rows)
    edf.to_csv(os.path.join(OUT, "e4_energies.csv"), index=False)
    krows = []
    for sub in order:
        for cond in CONDS:
            g = edf[(edf["sub"] == sub) & (edf.condition == cond)].sort_values("deg")
            krows.append(dict(sub=sub, condition=cond, k=fit_k(g.deg, g.E), n=int(g["n"].iloc[0])))
    kdf = pd.DataFrame(krows)
    kdf.to_csv(os.path.join(OUT, "k_person.csv"), index=False)
    med = kdf.groupby("condition", as_index=False).k.median()
    med.to_csv(os.path.join(OUT, "k_median.csv"), index=False)

    def kr(name):
        return float(med.loc[med.condition == name, "k"].iloc[0])

    ratio_tb = kr("trunk_50") / kr("bilateral_50")
    ratios = {
        "unilateral_50/bilateral_50": kr("unilateral_50") / kr("bilateral_50"),
        "bilateral_10/bilateral_50": kr("bilateral_10") / kr("bilateral_50"),
        "trunk_50/bilateral_50": ratio_tb,
    }
    json.dump(ratios, open(os.path.join(OUT, "k_ratios.json"), "w"), indent=2)

    per, pooled = e1_angle(aligned, ang)
    per.to_csv(os.path.join(OUT, "e1_per_person.csv"), index=False)
    pooled.to_csv(os.path.join(OUT, "e1_pooled.csv"), index=False)
    old = pd.read_csv("/data1/users/yangrui/projects/idea67b/experiment/runs/prl_v2/e1/pooled.csv")
    pos, ang_auc = {}, {}
    for act in ("Lunges", "Plank", "SQUAT"):
        for ref in ("O", "D2"):
            pos[(ref, act)] = float(old[(old.ref == ref) & (old.view == "positions") & (old.act == act)].auc.iloc[0])
            ang_auc[(ref, act)] = float(pooled[(pooled.ref == ref) & (pooled.act == act)].auc.iloc[0])
    h = verdicts(pos, ang_auc)

    print("rho start", flush=True)
    rho_rows = rho_27(pairs)
    pd.DataFrame(rho_rows).to_csv(os.path.join(OUT, "rho.csv"), index=False)
    plank = [r for r in rho_rows if r["act"] == "Plank" and r["view"] == "bone angles, all joints"][0]["ratio"]
    squat_auc = ang_auc[("O", "SQUAT")]
    h7a = "支持" if 0.8 <= ratio_tb <= 1.25 else "不支持"
    h7b = "支持" if plank < 0.9572 else "不支持"
    h7c = "支持" if squat_auc > 0.8485365854 else "不支持"
    fig = "图 1 的骨夹角值可保留，须注明角度集合" if 0.4786 <= plank <= 1.9144 else "图 1 改用 27 对的值，并注明 23 对的结果"
    # E1b bone angles, O and D2, using the new taus
    e1b = []
    subs = sorted({k[1] for k in aligned})
    correct = [k for k in aligned if k[0] == "SQUAT" and k[2] == 1]
    for kind in ("O", "D2"):
        meta = per[(per.ref == kind) & (per.act == "SQUAT")]
        for hname in subs:
            row = meta[meta.newcomer == hname].iloc[0]
            tau, fpr0 = float(row.tau), float(row.fpr)
            hk = [k for k in correct if k[1] == hname]
            others = [s for s in subs if s != hname]
            for cond in CONDS:
                for deg in E1B_DEGS:
                    hits = []
                    for k in hk:
                        if kind == "O":
                            ref = np.mean([ang[q] for q in hk if q != k], 0)
                        else:
                            ref = np.mean([ang[q] for q in correct if q[1] in others], 0)
                        Y = S2.apply_cond(aligned[k], cond, deg)
                        hits.append(C.score_mse(C.angles_27(Y, pairs), ref) > tau)
                    det = float(np.mean(hits))
                    e1b.append(dict(ref=kind, newcomer=hname, condition=cond, deg=deg, n=len(hk),
                                    detection=det, net=det - fpr0))
    e1b = pd.DataFrame(e1b)
    e1b.to_csv(os.path.join(OUT, "e1b_person.csv"), index=False)
    curve = []
    for (kind, cond, deg), g in e1b.groupby(["ref", "condition", "deg"]):
        curve.append(dict(ref=kind, condition=cond, deg=int(deg), n_people=int(g.newcomer.nunique()),
                          detection_mean=float(g.detection.mean()), net_mean=float(g.net.mean()),
                          net_min=float(g.net.min()), net_max=float(g.net.max())))
    pd.DataFrame(curve).to_csv(os.path.join(OUT, "e1b_curve.csv"), index=False)

    outv = {"H7a": h7a, "H7b": h7b, "H7c": h7c, "figure": fig,
            "k_trunk_over_bilateral": ratio_tb, "plank_rho": plank, "squat_auc_O": squat_auc,
            "H1": h["H1"], "H2": h["H2"], "H3": h["H3"], "H4": h["H4"], "deltas": h["deltas"]}
    json.dump(outv, open(os.path.join(OUT, "prereg_check.json"), "w"), indent=2)
    t1 = C.now_iso()
    C.update_manifest(os.path.abspath(__file__),
                      [os.path.join(OUT, f) for f in
                       ["definition.json", "selfcheck.json", "e4_energies.csv", "k_person.csv", "k_median.csv",
                        "k_ratios.json", "e1_per_person.csv", "e1_pooled.csv", "rho.csv", "e1b_person.csv",
                        "e1b_curve.csv", "prereg_check.json"]],
                      t0, t1)
    print("R2 DONE", h7a, h7b, h7c, fig, "seconds", round(time.time() - t_start, 3))


def load_aligned():
    z = np.load("/data1/users/yangrui/projects/idea67b/experiment/runs/prl_v2/e1/aligned_pos.npz")
    out = {}
    for key in z.files:
        act, sub, lab, rep = key.split("|")
        out[(act, sub, int(lab), int(rep))] = z[key]
    return out


if __name__ == "__main__":
    main()
