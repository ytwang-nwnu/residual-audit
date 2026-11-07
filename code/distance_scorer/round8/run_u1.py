"""U1 (PREREG_R8b): label-free personal calibration for the distance scorer.

Same cells / k / draws / threshold rule as round3/run_k1.py, but the k reps are drawn from
ALL own reps (any label). Reads frozen K1 outputs for O/D2 AUC and labelled recovery.
Writes only to runs/prl_v2/round8/u1.
"""
import csv, hashlib, importlib.util, json, os, sys, time
import numpy as np
import pandas as pd

CODE = "/data1/users/yangrui/projects/idea67b/experiment/code"
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, "prl_v2/round2"))
import common as C
import gaitex_cells_api as G
import prl_scheme12 as P

spec = importlib.util.spec_from_file_location("k1", os.path.join(CODE, "prl_v2/round3/run_k1.py"))
K1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(K1)

EXP = "/data1/users/yangrui/projects/idea67b/experiment/runs/prl_v2"
OUT = os.path.join(EXP, "round8", "u1"); os.makedirs(OUT, exist_ok=True)
KS, N_REP, SEED = (1, 2, 3, 5), 50, 0


def eval_unlabelled(correct, error, chosen_all, people):
    """chosen_all[p]: indices into vstack(correct[p], error[p]). Returns pooled AUC, n, per-person."""
    mus, rem_c, rem_e = {}, {}, {}
    for p in people:
        allp = np.vstack([correct[p], error[p]])
        mus[p] = allp[chosen_all[p]].mean(axis=0)
        nc = len(correct[p])
        mask = np.ones(len(allp), bool); mask[chosen_all[p]] = False
        rem_c[p] = np.flatnonzero(mask[:nc]); rem_e[p] = np.flatnonzero(mask[nc:])
    chunks_s, chunks_y, person = [], [], {}
    for h in people:
        sc_c = K1.row_mse(correct[h][rem_c[h]], mus[h]); sc_e = K1.row_mse(error[h][rem_e[h]], mus[h])
        cal = [K1.row_mse(correct[c][rem_c[c]], mus[c]) for c in people if c != h]
        tau = float(np.quantile(np.concatenate(cal), 1.0 - C.ALPHA))
        fpr = float(np.mean(sc_c > tau)) if len(sc_c) else float("nan")
        tpr = float(np.mean(sc_e > tau)) if len(sc_e) else float("nan")
        person[h] = (fpr, tpr, K1.auc(sc_e, sc_c), int(len(sc_c)), int(len(sc_e)))
        if len(sc_c): chunks_s.append(sc_c); chunks_y.append(np.ones(len(sc_c)))
        if len(sc_e): chunks_s.append(sc_e); chunks_y.append(np.zeros(len(sc_e)))
    sc = np.concatenate(chunks_s); lab = np.concatenate(chunks_y)
    return K1.auc(sc[lab == 0], sc[lab == 1]), int(len(sc)), person


def eval_all_loo(correct, error, people):
    """U_all: own mean over ALL reps, leaving the scored item out."""
    chunks_s, chunks_y, person = [], [], {}
    def scores(p):
        allp = np.vstack([correct[p], error[p]]); n = len(allp); tot = allp.sum(0)
        mu_loo = (tot[None, :] - allp) / (n - 1)
        s = np.mean((allp - mu_loo) ** 2, axis=1)
        return s[:len(correct[p])], s[len(correct[p]):]
    S = {p: scores(p) for p in people}
    for h in people:
        sc_c, sc_e = S[h]
        cal = np.concatenate([S[c][0] for c in people if c != h])
        tau = float(np.quantile(cal, 1.0 - C.ALPHA))
        person[h] = (float(np.mean(sc_c > tau)), float(np.mean(sc_e > tau)) if len(sc_e) else float("nan"),
                     K1.auc(sc_e, sc_c), int(len(sc_c)), int(len(sc_e)))
        chunks_s += [sc_c, sc_e]; chunks_y += [np.ones(len(sc_c)), np.zeros(len(sc_e))]
    sc = np.concatenate(chunks_s); lab = np.concatenate(chunks_y)
    return K1.auc(sc[lab == 0], sc[lab == 1]), int(len(sc)), person


def frozen_people(ses):
    """Persons that entered the frozen K1/paper analysis (RGS: 17, 'ziri' lacked IK then)."""
    df = pd.read_csv(os.path.join(EXP, "round3/k1/per_person.csv"))
    return set(df[(df.dataset == "GAITEX") & (df.act == ses) & (df.protocol == "O")].newcomer)


def expand_frozen(cells, ses):
    reps = K1.expand(cells[ses]); keep = frozen_people(ses)
    return {k: v for k, v in reps.items() if k[0] in keep}


def main():
    t0 = C.now_iso(); t_start = time.time()
    aligned = K1.load_aligned(); subs = sorted({k[1] for k in aligned}); acts = ["Lunges", "Plank", "SQUAT"]
    pos = {k: P.flat(v) for k, v in aligned.items()}
    ang = {k: C.angles_27(v, C.bone_pairs_23()) for k, v in aligned.items()}
    rulers = {"positions": pos, "bone27": ang}
    k1rec = pd.read_csv(os.path.join(EXP, "round3/k1/recovery.csv"))
    od = {(r.dataset, r.act, r.ruler): (float(r.auc_O), float(r.auc_D2)) for r in k1rec.itertuples()}
    lab_rec = {(r.dataset, r.act, r.ruler, int(r.k)): float(r.recovery_mean) for r in k1rec.itertuples()}

    packs = {}  # (dataset, act) -> (people, {ruler: (correct, error)})
    for act in acts:
        d = {}
        for ruler, feat in rulers.items():
            correct, error = {}, {}
            for p in subs:
                ck = sorted([k for k in feat if k[0] == act and k[1] == p and k[2] == 1], key=lambda k: k[3])
                ek = sorted([k for k in feat if k[0] == act and k[1] == p and k[2] != 1], key=lambda k: k[3])
                correct[p] = np.stack([np.asarray(feat[k], float).reshape(-1) for k in ck])
                error[p] = np.stack([np.asarray(feat[k], float).reshape(-1) for k in ek]) if ek else np.zeros((0, correct[p].shape[1]))
            d[ruler] = (correct, error)
        packs[("EC3D", act)] = (subs, d)
    cells = G.build_cells(sessions=("rd", "rgs"))
    for ses in ("rd", "rgs"):
        reps = expand_frozen(cells, ses); people = sorted({k[0] for k in reps if k[1] == 1})
        correct, error, _ = K1.pack_gaitex(reps, people)
        packs[("GAITEX", ses)] = (people, {"imu": (correct, error)})

    rng = np.random.default_rng(SEED)
    pooled_rows, per_rows, all_rows = [], [], []
    order = [("EC3D", a) for a in acts] + [("GAITEX", s) for s in ("rd", "rgs")]
    for dataset, act in order:
        people, d = packs[(dataset, act)]
        n_all = {p: len(next(iter(d.values()))[0][p]) + len(next(iter(d.values()))[1][p]) for p in people}
        for ruler, (correct, error) in d.items():
            a, n, person = eval_all_loo(correct, error, people)
            all_rows.append(dict(dataset=dataset, act=act, ruler=ruler, protocol="U_all", pooled_auc=a, n=n))
            for h, (fpr, tpr, ua, nc, ne) in person.items():
                per_rows.append(dict(dataset=dataset, act=act, ruler=ruler, protocol="U_all", k="", newcomer=h,
                                     fpr=fpr, tpr=tpr, auc=ua, n_correct=nc, n_error=ne))
        for k in KS:
            for rep in range(N_REP):
                chosen = {p: rng.choice(n_all[p], size=k, replace=False) for p in people}
                for ruler, (correct, error) in d.items():
                    a, n, person = eval_unlabelled(correct, error, chosen, people)
                    pooled_rows.append(dict(dataset=dataset, act=act, ruler=ruler, k=k, rep=rep, pooled_auc=a, n=n,
                                            n_err_drawn=int(sum(int((chosen[p] >= len(correct[p])).sum()) for p in people))))
    pooled = pd.DataFrame(pooled_rows)
    rec_rows = []
    for (dataset, act, ruler, k), g in pooled.groupby(["dataset", "act", "ruler", "k"], sort=False):
        aO, aD = od[(dataset, act, ruler)]; delta = aO - aD; aucs = g.pooled_auc.to_numpy(float)
        ratios = (aucs - aD) / delta if delta >= 0.05 else np.full(len(aucs), np.nan)
        rec_rows.append(dict(dataset=dataset, act=act, ruler=ruler, k=int(k), auc_mean=float(aucs.mean()),
                             auc_O=aO, auc_D2=aD, delta=float(delta), applicable=bool(delta >= 0.05),
                             recovery_unlabelled=float(np.nanmean(ratios)),
                             recovery_labelled_k1=lab_rec.get((dataset, act, ruler, int(k)), float("nan")),
                             mean_err_reps_drawn=float(g.n_err_drawn.mean())))
    for r in all_rows:
        aO, aD = od[(r["dataset"], r["act"], r["ruler"])]; delta = aO - aD
        r.update(auc_O=aO, auc_D2=aD, delta=delta, recovery=(r["pooled_auc"] - aD) / delta if delta >= 0.05 else float("nan"))
    rec = pd.DataFrame(rec_rows); allr = pd.DataFrame(all_rows)

    def cellrec(ds, act, ruler, k, col="recovery_unlabelled"):
        return float(rec[(rec.dataset == ds) & (rec.act == act) & (rec.ruler == ruler) & (rec.k == k)][col].iloc[0])
    h28 = {s: cellrec("GAITEX", s, "imu", 1) for s in ("rd", "rgs")}
    ec5 = [("SQUAT", "positions"), ("Lunges", "positions"), ("SQUAT", "bone27"), ("Lunges", "bone27"), ("Plank", "bone27")]
    gaps = {f"{a}/{r}": cellrec("EC3D", a, r, 2, "recovery_labelled_k1") - cellrec("EC3D", a, r, 2) for a, r in ec5}
    lower = sum(v > 0 for v in gaps.values())
    squat_gaps = sorted(gaps, key=gaps.get, reverse=True)[:2]
    h30_cells = allr[allr.delta >= 0.05]; h30_n = int((h30_cells.recovery >= 0.8).sum())
    check = {
        "H28": "支持" if all(v >= 0.5 for v in h28.values()) else "不支持", "H28_k1_recovery": h28,
        "H29": ("支持" if lower >= 4 and all(s.startswith("SQUAT") for s in squat_gaps) else
                "部分支持" if lower >= 4 else "不支持"), "H29_gaps_labelled_minus_unlabelled_k2": gaps,
        "H30": "支持" if h30_n >= 4 else "不支持", "H30_n_cells_Uall_ge_0.8": h30_n,
        "U_all": allr.to_dict("records"), "seed": SEED, "n_reps": N_REP,
    }
    pooled.to_csv(os.path.join(OUT, "pooled.csv"), index=False)
    rec.to_csv(os.path.join(OUT, "recovery.csv"), index=False)
    allr.to_csv(os.path.join(OUT, "u_all.csv"), index=False)
    pd.DataFrame(per_rows).to_csv(os.path.join(OUT, "per_person_uall.csv"), index=False)
    json.dump(check, open(os.path.join(OUT, "prereg_check.json"), "w"), indent=2, ensure_ascii=False)
    print("U1 DONE", check["H28"], check["H29"], check["H30"], "sec", round(time.time() - t_start, 1))
    print(rec[rec.applicable][["dataset", "act", "ruler", "k", "recovery_unlabelled", "recovery_labelled_k1"]].to_string(index=False))
    print(allr[["dataset", "act", "ruler", "pooled_auc", "recovery"]].to_string(index=False))


if __name__ == "__main__":
    main()
