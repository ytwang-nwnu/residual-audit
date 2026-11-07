"""B1 (PREREG_R8b): uncertainty of ΔAUC = AUC_O − AUC_D2.

GAITEX: person-cluster bootstrap (B=2000, seed 0) over held-out persons' frozen score vectors.
EC3D: leave-one-person-out jackknife of Δ (4 persons). Writes only to runs/prl_v2/round8/b1.
"""
import importlib.util, json, os, sys, time
import numpy as np
import pandas as pd

CODE = "/data1/users/yangrui/projects/idea67b/experiment/code"
sys.path.insert(0, CODE); sys.path.insert(0, os.path.join(CODE, "prl_v2/round2"))
import common as C
import gaitex_cells_api as G
import prl_scheme12 as P
spec = importlib.util.spec_from_file_location("k1", os.path.join(CODE, "prl_v2/round3/run_k1.py"))
K1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(K1)

EXP = "/data1/users/yangrui/projects/idea67b/experiment/runs/prl_v2"
OUT = os.path.join(EXP, "round8", "b1"); os.makedirs(OUT, exist_ok=True)
B, SEED = 2000, 0


def person_scores_gaitex(reps):
    """Per held-out person: (labels, O scores, D2 scores). Mirrors K1.gaitex_od exactly."""
    subs = sorted({k[0] for k in reps})
    keys = lambda sub, lab=None: [k for k in reps if k[0] == sub and (lab is None or k[1] == lab)]
    oscore = {}
    for k, x in reps.items():
        own = [reps[q] for q in reps if q[0] == k[0] and q[1] == 1 and q != k]
        if own: oscore[k] = C.score_mse(x, np.mean(own, 0))
    out = {}
    for h in subs:
        kh = keys(h)
        if not any(k[1] == 1 for k in kh): continue
        labs = np.array([k[1] for k in kh])
        sO = np.array([oscore[k] for k in kh])
        others = [s for s in subs if s != h]
        ref_h = np.mean([reps[k] for s in others for k in keys(s, 1)], 0)
        sD = np.array([C.score_mse(reps[k], ref_h) for k in kh])
        out[h] = (labs, sO, sD)
    return out


def person_scores_ec3d(feat, act, subs):
    keys_all = list(feat)
    keys = lambda sub, lab=None: [k for k in keys_all if k[0] == act and k[1] == sub and (lab is None or k[2] == lab)]
    oscore = {}
    for k, x in feat.items():
        if k[0] != act: continue
        own = [feat[q] for q in keys_all if q[0] == act and q[1] == k[1] and q[2] == 1 and q != k]
        if own: oscore[k] = float(np.mean((x - np.mean(own, 0)) ** 2))
    out = {}
    for h in subs:
        kh = keys(h); labs = np.array([k[2] for k in kh])
        sO = np.array([oscore[k] for k in kh])
        ref_h = np.mean([feat[k] for s in subs if s != h for k in keys(s, 1)], 0)
        sD = np.array([float(np.mean((feat[k] - ref_h) ** 2)) for k in kh])
        out[h] = (labs, sO, sD)
    return out


def pooled_delta(ps, people):
    labs = np.concatenate([ps[p][0] for p in people]); sO = np.concatenate([ps[p][1] for p in people]); sD = np.concatenate([ps[p][2] for p in people])
    aO = K1.auc(sO[labs != 1], sO[labs == 1]); aD = K1.auc(sD[labs != 1], sD[labs == 1])
    return aO, aD, aO - aD


def frozen_people(ses):
    """Persons that entered the frozen K1/paper analysis (RGS: 17, 'ziri' lacked IK then)."""
    df = pd.read_csv(os.path.join(EXP, "round3/k1/per_person.csv"))
    return set(df[(df.dataset == "GAITEX") & (df.act == ses) & (df.protocol == "O")].newcomer)


def expand_frozen(cells, ses):
    reps = K1.expand(cells[ses]); keep = frozen_people(ses)
    return {k: v for k, v in reps.items() if k[0] in keep}


def main():
    t_start = time.time()
    k1rec = pd.read_csv(os.path.join(EXP, "round3/k1/recovery.csv")).drop_duplicates(["dataset", "act", "ruler"])
    frozen = {(r.dataset, r.act, r.ruler): (float(r.auc_O), float(r.auc_D2)) for r in k1rec.itertuples()}
    rows, jack = [], []
    rng = np.random.default_rng(SEED)
    # GAITEX bootstrap
    cells = G.build_cells(sessions=("rd", "rgs"))
    for ses in ("rd", "rgs"):
        ps = person_scores_gaitex(expand_frozen(cells, ses)); people = sorted(ps)
        aO, aD, d = pooled_delta(ps, people)
        fO, fD = frozen[("GAITEX", ses, "imu")]
        assert abs(aO - fO) < 1e-9 and abs(aD - fD) < 1e-9, ("selfcheck failed", ses, aO, fO, aD, fD)
        boots = np.array([pooled_delta(ps, list(rng.choice(people, size=len(people), replace=True)))[2] for _ in range(B)])
        rows.append(dict(dataset="GAITEX", act=ses, ruler="imu", n_people=len(people), delta=d,
                         ci_lo=float(np.quantile(boots, 0.025)), ci_hi=float(np.quantile(boots, 0.975)), B=B))
        for p in people:
            jack.append(dict(dataset="GAITEX", act=ses, ruler="imu", left_out=p, delta=pooled_delta(ps, [q for q in people if q != p])[2]))
    # EC3D jackknife
    aligned = K1.load_aligned(); subs = sorted({k[1] for k in aligned})
    rulers = {"positions": {k: P.flat(v) for k, v in aligned.items()},
              "bone27": {k: C.angles_27(v, C.bone_pairs_23()) for k, v in aligned.items()}}
    for ruler, feat in rulers.items():
        for act in ("SQUAT", "Lunges", "Plank"):
            ps = person_scores_ec3d(feat, act, subs)
            aO, aD, d = pooled_delta(ps, subs); fO, fD = frozen[("EC3D", act, ruler)]
            assert abs(aO - fO) < 1e-9 and abs(aD - fD) < 1e-9, ("selfcheck failed", act, ruler)
            js = {p: pooled_delta(ps, [q for q in subs if q != p])[2] for p in subs}
            rows.append(dict(dataset="EC3D", act=act, ruler=ruler, n_people=4, delta=d,
                             jack_min=min(js.values()), jack_max=max(js.values()), B=0))
            for p, v in js.items(): jack.append(dict(dataset="EC3D", act=act, ruler=ruler, left_out=p, delta=v))
    df = pd.DataFrame(rows); jk = pd.DataFrame(jack)
    rd = df[(df.act == "rd")].iloc[0]; rgs = df[(df.act == "rgs")].iloc[0]
    h31 = rd.ci_lo > 0 and rgs.ci_lo > 0
    h32 = "支持" if rd.ci_lo > rgs.ci_hi else ("部分支持" if rd.delta > rgs.ci_hi else "不支持")
    # H33: EC3D sign + rho order per left-out person
    rho_order = {"positions": ["SQUAT", "Lunges", "Plank"], "bone27": ["SQUAT", "Lunges", "Plank"]}
    applicable = [("SQUAT", "positions"), ("Lunges", "positions"), ("SQUAT", "bone27"), ("Lunges", "bone27"), ("Plank", "bone27")]
    sign_ok = all((jk[(jk.dataset == "EC3D") & (jk.act == a) & (jk.ruler == r)].delta > 0).all() for a, r in applicable)
    order_ok = 0
    for p in subs:
        ok = True
        for r, acts in rho_order.items():
            vals = [float(jk[(jk.dataset == "EC3D") & (jk.ruler == r) & (jk.act == a) & (jk.left_out == p)].delta.iloc[0]) for a in acts]
            ok &= vals[0] >= vals[1] >= vals[2]
        order_ok += int(ok)
    check = {"H31": "支持" if h31 else "不支持", "H32": h32,
             "H33": "支持" if sign_ok and order_ok == 4 else "部分支持" if sign_ok else "不支持",
             "H33_sign_all_positive": bool(sign_ok), "H33_persons_preserving_rho_order": order_ok,
             "B": B, "seed": SEED, "note": "bootstrap resamples persons' frozen score vectors; references/thresholds not re-estimated"}
    df.to_csv(os.path.join(OUT, "delta_ci.csv"), index=False); jk.to_csv(os.path.join(OUT, "jackknife.csv"), index=False)
    json.dump(check, open(os.path.join(OUT, "prereg_check.json"), "w"), indent=2, ensure_ascii=False)
    print("B1 DONE", check["H31"], check["H32"], check["H33"], "sec", round(time.time() - t_start, 1))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
