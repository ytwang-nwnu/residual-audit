"""K1: how many teacher-confirmed correct reps recover detection.

Stops if the recomputed O/D2 pooled AUC differs from the landed files by 1e-9 or more.
Does not modify earlier experiments. One draw of k correct reps is shared across rulers.
"""
import csv
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

CODE = "/data1/users/yangrui/projects/idea67b/experiment/code"
sys.path.insert(0, os.path.join(CODE, "prl_v2/round2"))
sys.path.insert(0, CODE)
import common as C
import gaitex_cells_api as G
import prl_scheme12 as P

ROOT = "/data1/users/yangrui/projects/idea67b/experiment/runs/prl_v2/round3"
EXP = "/data1/users/yangrui/projects/idea67b/experiment/runs/prl_v2"
OUT = os.path.join(ROOT, "k1")
os.makedirs(OUT, exist_ok=True)
KS = (1, 2, 3, 5)
N_REP = 50
APPLICABLE = [
    ("EC3D", "SQUAT", "positions"),
    ("EC3D", "Lunges", "positions"),
    ("EC3D", "SQUAT", "bone27"),
    ("EC3D", "Lunges", "bone27"),
    ("EC3D", "Plank", "bone27"),
    ("GAITEX", "rd", "imu"),
    ("GAITEX", "rgs", "imu"),
]


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def auc(err, cor):
    return C.auc(err, cor)


def row_mse(stack, mu):
    if len(stack) == 0:
        return np.zeros(0, dtype=float)
    return np.mean((stack - mu) ** 2, axis=1)


def load_aligned():
    z = np.load(os.path.join(EXP, "e1/aligned_pos.npz"))
    out = {}
    for key in z.files:
        act, sub, lab, rep = key.split("|")
        out[(act, sub, int(lab), int(rep))] = z[key]
    return out


def expand(cells):
    reps = {}
    for (sub, lab), xs in cells.items():
        for i, x in enumerate(xs):
            reps[(sub, int(lab), i)] = np.asarray(x, dtype=np.float64)
    return reps


def read_csv_auc(path):
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[(row["ref"], row.get("view", ""), row["act"])] = float(row["auc"])
    return out


def gaitex_od(reps):
    """Same O/D2 loop as round2/run_r4.py run_e1."""
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
        per.append(dict(protocol="O", newcomer=h, n_correct=int((labs == 1).sum()),
                        n_error=int((labs != 1).sum()), fpr=float(dec[labs == 1].mean()),
                        tpr=float(dec[labs != 1].mean()) if (labs != 1).any() else float("nan"),
                        auc=auc(sh[labs != 1], sh[labs == 1])))
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
        per.append(dict(protocol="D2", newcomer=h, n_correct=int((labs == 1).sum()),
                        n_error=int((labs != 1).sum()), fpr=float(dec[labs == 1].mean()),
                        tpr=float(dec[labs != 1].mean()) if (labs != 1).any() else float("nan"),
                        auc=auc(sh[labs != 1], sh[labs == 1])))
        store["D2"].append((sh, labs))
    pooled = {}
    for kind, parts in store.items():
        sc = np.concatenate([s for s, _ in parts])
        lab = np.concatenate([L for _, L in parts])
        pooled[kind] = dict(n=int(len(sc)), n_people=len(parts), auc=auc(sc[lab != 1], sc[lab == 1]))
    return per, pooled


def ec3d_od(feat, act, subs):
    """Same O/D2 loop as run_e1.py / round2 e1_angle, for one action."""
    keys_all = list(feat)

    def keys(sub, lab=None):
        return [k for k in keys_all if k[0] == act and k[1] == sub and (lab is None or k[2] == lab)]

    oscore = {}
    for k, x in feat.items():
        if k[0] != act:
            continue
        own = [feat[q] for q in keys_all if q[0] == act and q[1] == k[1] and q[2] == 1 and q != k]
        if own:
            oscore[k] = float(np.mean((x - np.mean(own, 0)) ** 2))
    per = []
    store = {"O": [], "D2": []}
    for h in subs:
        others = [s for s in subs if s != h]
        kh = keys(h)
        labs = np.array([k[2] for k in kh])
        cal = [oscore[k] for s in others for k in keys(s, 1)]
        tau = C.tau_of(cal)
        sh = np.array([oscore[k] for k in kh])
        dec = sh > tau
        per.append(dict(protocol="O", newcomer=h, n_correct=int((labs == 1).sum()),
                        n_error=int((labs != 1).sum()), fpr=float(dec[labs == 1].mean()),
                        tpr=float(dec[labs != 1].mean()) if (labs != 1).any() else float("nan"),
                        auc=auc(sh[labs != 1], sh[labs == 1])))
        store["O"].append((sh, labs))
        xs = [feat[k] for s in others for k in keys(s, 1)]
        ref_h = np.mean(xs, 0)
        cal_scores = []
        for c in others:
            rest = [s for s in others if s != c]
            ref_c = np.mean([feat[k] for s in rest for k in keys(s, 1)], 0)
            cal_scores += [float(np.mean((feat[k] - ref_c) ** 2)) for k in keys(c, 1)]
        tau = C.tau_of(cal_scores)
        sh = np.array([float(np.mean((feat[k] - ref_h) ** 2)) for k in kh])
        dec = sh > tau
        per.append(dict(protocol="D2", newcomer=h, n_correct=int((labs == 1).sum()),
                        n_error=int((labs != 1).sum()), fpr=float(dec[labs == 1].mean()),
                        tpr=float(dec[labs != 1].mean()) if (labs != 1).any() else float("nan"),
                        auc=auc(sh[labs != 1], sh[labs == 1])))
        store["D2"].append((sh, labs))
    pooled = {}
    for kind, parts in store.items():
        sc = np.concatenate([s for s, _ in parts])
        lab = np.concatenate([L for _, L in parts])
        pooled[kind] = dict(n=int(len(sc)), n_people=len(parts), auc=auc(sc[lab != 1], sc[lab == 1]))
    return per, pooled


def pack(feat, act, people, key_person, key_lab, key_rep):
    correct, error, n_ok = {}, {}, {}
    for p in people:
        ck = sorted([k for k in feat if key_person(k) == p and key_lab(k) == 1 and k[0] == act], key=key_rep)
        ek = sorted([k for k in feat if key_person(k) == p and key_lab(k) != 1 and k[0] == act], key=key_rep)
        # GAITEX keys are (sub, lab, i) and have no action in the key. Caller passes act-filtered feats.
        correct[p] = np.stack([np.asarray(feat[k], float).reshape(-1) for k in ck], 0) if ck else np.zeros((0, 1))
        error[p] = np.stack([np.asarray(feat[k], float).reshape(-1) for k in ek], 0) if ek else np.zeros((0, 1))
        n_ok[p] = len(ck)
    return correct, error, n_ok


def pack_gaitex(reps, people):
    correct, error, n_ok = {}, {}, {}
    for p in people:
        ck = sorted([k for k in reps if k[0] == p and k[1] == 1], key=lambda k: k[2])
        ek = sorted([k for k in reps if k[0] == p and k[1] != 1], key=lambda k: k[2])
        correct[p] = np.stack([np.asarray(reps[k], float).reshape(-1) for k in ck], 0)
        error[p] = np.stack([np.asarray(reps[k], float).reshape(-1) for k in ek], 0) if ek else np.zeros((0, correct[p].shape[1]))
        n_ok[p] = len(ck)
    return correct, error, n_ok


def eval_once(correct, error, chosen, people):
    mus, remain = {}, {}
    for p in people:
        mus[p] = correct[p][chosen[p]].mean(axis=0)
        mask = np.ones(len(correct[p]), dtype=bool)
        mask[chosen[p]] = False
        remain[p] = np.flatnonzero(mask)
    chunks_s, chunks_y = [], []
    person = {}
    for h in people:
        sc_c = row_mse(correct[h][remain[h]], mus[h])
        sc_e = row_mse(error[h], mus[h])
        cal = [row_mse(correct[c][remain[c]], mus[c]) for c in people if c != h]
        tau = float(np.quantile(np.concatenate(cal), 1.0 - C.ALPHA))
        fpr = float(np.mean(sc_c > tau)) if len(sc_c) else float("nan")
        tpr = float(np.mean(sc_e > tau)) if len(sc_e) else float("nan")
        person[h] = (fpr, tpr, auc(sc_e, sc_c), int(len(sc_c)), int(len(sc_e)))
        if len(sc_c):
            chunks_s.append(sc_c)
            chunks_y.append(np.ones(len(sc_c)))
        if len(sc_e):
            chunks_s.append(sc_e)
            chunks_y.append(np.zeros(len(sc_e)))
    sc = np.concatenate(chunks_s)
    lab = np.concatenate(chunks_y)
    return auc(sc[lab == 0], sc[lab == 1]), int(len(sc)), person


def rho_table():
    out = {}
    with open(os.path.join(EXP, "e1/rho_realign.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row["view"] == "positions":
                out[("EC3D", row["act"], "positions")] = float(row["rho"])
    with open(os.path.join(EXP, "round2/r2/rho.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row["view"] == "bone angles, all joints":
                out[("EC3D", row["act"], "bone27")] = float(row["ratio"])
    r4 = json.load(open(os.path.join(EXP, "round2/r4/selfcheck.json")))
    for d in r4["diffs"]:
        out[("GAITEX", d["session"], "imu")] = float(d["ratio"])
    return out


def kstar(means, q):
    for k in KS:
        if means[k] >= q:
            return str(k)
    return ">5"


def rank_k(token):
    return 6 if token == ">5" else int(token)


def update_manifest(script, outputs, t0, t1):
    path = os.path.join(ROOT, "MANIFEST.json")
    man = {"experiments": []}
    if os.path.exists(path):
        man = json.load(open(path, encoding="utf-8"))
    entry = {
        "script": script,
        "script_md5": md5_file(script),
        "outputs": [{"path": p, "md5": md5_file(p)} for p in outputs if os.path.exists(p)],
        "seed": 0,
        "start": t0,
        "end": t1,
        "python": sys.version,
        "numpy": np.__version__,
    }
    man["experiments"] = [e for e in man.get("experiments", []) if e.get("script") != script]
    man["experiments"].append(entry)
    json.dump(man, open(path, "w"), indent=2, ensure_ascii=False)


def main():
    t0 = C.now_iso()
    t_start = time.time()
    prereg = os.path.join(ROOT, "PREREG_R3.md")
    stamp = json.load(open(os.path.join(ROOT, "R0_stamp.json")))
    digest = md5_file(prereg)
    if digest != stamp["md5"]:
        json.dump({"passed": False, "reason": "prereg md5 changed", "md5": digest, "stamp": stamp["md5"]},
                  open(os.path.join(OUT, "selfcheck.json"), "w"), indent=2)
        sys.exit(2)

    aligned = load_aligned()
    subs = sorted({k[1] for k in aligned})
    acts = ["Lunges", "Plank", "SQUAT"]
    pos = {k: P.flat(v) for k, v in aligned.items()}
    pairs = C.bone_pairs_23()
    ang = {k: C.angles_27(v, pairs) for k, v in aligned.items()}

    targets = []
    saved_pos = {}
    with open(os.path.join(EXP, "e1/pooled.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row["view"] == "positions" and row["ref"] in ("O", "D2"):
                saved_pos[(row["ref"], row["act"])] = float(row["auc"])
    saved_ang = {}
    with open(os.path.join(EXP, "round2/r2/e1_pooled.csv"), newline="") as f:
        for row in csv.DictReader(f):
            if row["ref"] in ("O", "D2"):
                saved_ang[(row["ref"], row["act"])] = float(row["auc"])
    saved_g = {}
    for ses in ("rd", "rgs"):
        blob = json.load(open(os.path.join(EXP, "round2/r4/pooled_%s.json" % ses)))
        saved_g[ses] = {r["ref"]: float(r["auc"]) for r in blob}

    diffs = []
    per_rows = []
    od_auc = {}
    rulers = {"positions": pos, "bone27": ang}
    for ruler, feat in rulers.items():
        saved = saved_pos if ruler == "positions" else saved_ang
        for act in acts:
            per, pooled = ec3d_od(feat, act, subs)
            for kind in ("O", "D2"):
                got = float(pooled[kind]["auc"])
                exp = saved[(kind, act)]
                d = abs(got - exp)
                diffs.append(dict(dataset="EC3D", act=act, ruler=ruler, ref=kind, got=got, expected=exp, abs_diff=d, n=pooled[kind]["n"]))
                od_auc[( "EC3D", act, ruler, kind)] = got
            for row in per:
                row.update(dataset="EC3D", act=act, ruler=ruler, k="")
                per_rows.append(row)
    cells = G.build_cells(sessions=("rd", "rgs"))
    gaitex_pack = {}
    gaitex_people = {}
    for ses in ("rd", "rgs"):
        reps = expand(cells[ses])
        per, pooled = gaitex_od(reps)
        for kind in ("O", "D2"):
            got = float(pooled[kind]["auc"])
            exp = saved_g[ses][kind]
            d = abs(got - exp)
            diffs.append(dict(dataset="GAITEX", act=ses, ruler="imu", ref=kind, got=got, expected=exp, abs_diff=d, n=pooled[kind]["n"]))
            od_auc[("GAITEX", ses, "imu", kind)] = got
        for row in per:
            row.update(dataset="GAITEX", act=ses, ruler="imu", k="")
            per_rows.append(row)
        people = sorted({r["newcomer"] for r in per if r["protocol"] == "O"})
        gaitex_people[ses] = people
        gaitex_pack[ses] = pack_gaitex(reps, people)
        short = [p for p in people if gaitex_pack[ses][2][p] < 5]
        if short:
            json.dump({"passed": False, "reason": "a person has fewer than 5 correct reps", "session": ses, "people": short},
                      open(os.path.join(OUT, "selfcheck.json"), "w"), indent=2)
            print("STOP short correct reps", ses, short, flush=True)
            sys.exit(2)

    max_diff = max(d["abs_diff"] for d in diffs)
    passed = max_diff < 1e-9
    selfcheck = {
        "passed": passed,
        "max_abs_diff": max_diff,
        "n_compared": len(diffs),
        "diffs": diffs,
        "prereg_md5": digest,
        "sampling_order": "dataset EC3D then GAITEX; action Lunges, Plank, SQUAT or rd, rgs; k in 1,2,3,5; repeat 0..49; person alphabetical. One draw is shared by every ruler of that dataset and action.",
        "correct_index": "EC3D correct reps are indexed by increasing rep id. GAITEX correct reps are indexed by the cache enumeration order.",
        "recovery_mean": "mean of the 50 per-repeat ratios, not the ratio of the mean AUCs",
        "quantile": "numpy default linear quantile, same function as tau",
    }
    json.dump(selfcheck, open(os.path.join(OUT, "selfcheck.json"), "w"), indent=2)
    print("K1 SELFCHECK", passed, max_diff, flush=True)
    if not passed:
        update_manifest(os.path.abspath(__file__), [os.path.join(OUT, "selfcheck.json")], t0, C.now_iso())
        sys.exit(2)

    for act in acts:
        nmin = min(sum(1 for k in aligned if k[0] == act and k[1] == p and k[2] == 1) for p in subs)
        if nmin < 5:
            json.dump({"passed": False, "reason": "EC3D correct reps below 5", "act": act, "nmin": nmin},
                      open(os.path.join(OUT, "selfcheck.json"), "w"), indent=2)
            sys.exit(2)

    ec_pack = {}
    for ruler, feat in rulers.items():
        ec_pack[ruler] = {}
        for act in acts:
            correct, error, n_ok = {}, {}, {}
            for p in subs:
                ck = sorted([k for k in feat if k[0] == act and k[1] == p and k[2] == 1], key=lambda k: k[3])
                ek = sorted([k for k in feat if k[0] == act and k[1] == p and k[2] != 1], key=lambda k: k[3])
                correct[p] = np.stack([np.asarray(feat[k], float).reshape(-1) for k in ck], 0)
                error[p] = np.stack([np.asarray(feat[k], float).reshape(-1) for k in ek], 0) if ek else np.zeros((0, correct[p].shape[1]))
                n_ok[p] = len(ck)
            ec_pack[ruler][act] = (correct, error, n_ok)

    rng = np.random.default_rng(0)
    pooled_rows = []
    acc = {}  # (dataset, act, ruler, k, person) -> lists

    def consume(dataset, act, people, n_ok, packs):
        for k in KS:
            for rep in range(N_REP):
                chosen = {}
                for person in people:
                    chosen[person] = rng.choice(n_ok[person], size=k, replace=False)
                for ruler, (correct, error, _) in packs.items():
                    pooled, n, person = eval_once(correct, error, chosen, people)
                    pooled_rows.append(dict(dataset=dataset, act=act, ruler=ruler, k=k, rep=rep,
                                            pooled_auc=pooled, n=n, n_people=len(people)))
                    for h, (fpr, tpr, ua, nc, ne) in person.items():
                        key = (dataset, act, ruler, k, h)
                        slot = acc.setdefault(key, {"fpr": [], "tpr": [], "auc": [], "n_correct": nc, "n_error": ne})
                        slot["fpr"].append(fpr)
                        slot["tpr"].append(tpr)
                        slot["auc"].append(ua)

    for act in acts:
        n_ok = ec_pack["positions"][act][2]
        packs = {ruler: ec_pack[ruler][act] for ruler in ("positions", "bone27")}
        print("EC3D", act, "start", flush=True)
        consume("EC3D", act, subs, n_ok, packs)
    for ses in ("rd", "rgs"):
        print("GAITEX", ses, "start", flush=True)
        correct, error, n_ok = gaitex_pack[ses]
        consume("GAITEX", ses, gaitex_people[ses], n_ok, {"imu": (correct, error, n_ok)})

    for key, slot in acc.items():
        dataset, act, ruler, k, h = key
        per_rows.append(dict(dataset=dataset, act=act, ruler=ruler, protocol="P%d" % k, k=k, newcomer=h,
                             n_correct=slot["n_correct"], n_error=slot["n_error"], n_reps=N_REP,
                             fpr=float(np.mean(slot["fpr"])), tpr=float(np.mean(slot["tpr"])),
                             auc=float(np.nanmean(slot["auc"]))))

    rhos = rho_table()
    pooled = pd.DataFrame(pooled_rows)
    rec_rows = []
    by = {}
    for (dataset, act, ruler, k), g in pooled.groupby(["dataset", "act", "ruler", "k"], sort=False):
        aucs = g.pooled_auc.to_numpy(float)
        delta = od_auc[(dataset, act, ruler, "O")] - od_auc[(dataset, act, ruler, "D2")]
        applicable = delta >= 0.05
        if applicable:
            ratios = (aucs - od_auc[(dataset, act, ruler, "D2")]) / delta
            rec_mean, rec_lo, rec_hi = float(np.mean(ratios)), float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975))
        else:
            rec_mean = rec_lo = rec_hi = float("nan")
        by.setdefault((dataset, act, ruler), {})[int(k)] = rec_mean
        rec_rows.append(dict(
            dataset=dataset, act=act, ruler=ruler, k=int(k), n_reps=N_REP,
            auc_mean=float(np.mean(aucs)), auc_p025=float(np.quantile(aucs, 0.025)),
            auc_p975=float(np.quantile(aucs, 0.975)),
            auc_O=od_auc[(dataset, act, ruler, "O")], auc_D2=od_auc[(dataset, act, ruler, "D2")],
            delta=float(delta), applicable=bool(applicable),
            recovery_mean=rec_mean, recovery_p025=rec_lo, recovery_p975=rec_hi,
            rho=rhos[(dataset, act, ruler)], n=int(g["n"].iloc[0]), n_people=int(g["n_people"].iloc[0]),
        ))
    for row in rec_rows:
        key = (row["dataset"], row["act"], row["ruler"])
        if key in APPLICABLE and row["applicable"]:
            row["kstar_0.5"] = kstar(by[key], 0.5)
            row["kstar_0.8"] = kstar(by[key], 0.8)
        else:
            row["kstar_0.5"] = "不适用"
            row["kstar_0.8"] = "不适用"

    def cell_means(dataset, act, ruler):
        rows = [r for r in rec_rows if r["dataset"] == dataset and r["act"] == act and r["ruler"] == ruler]
        return {r["k"]: r["auc_mean"] for r in rows}, {r["k"]: r["recovery_mean"] for r in rows}, rows[0]

    detail = []
    for dataset, act, ruler in APPLICABLE:
        aucs, recs, head = cell_means(dataset, act, ruler)
        means = {r["k"]: r["recovery_mean"] for r in rec_rows if (r["dataset"], r["act"], r["ruler"]) == (dataset, act, ruler)}
        drops = [aucs[a] - aucs[b] for a, b in ((1, 2), (2, 3), (3, 5))]
        detail.append(dict(dataset=dataset, act=act, ruler=ruler,
                           auc_k=aucs, auc_D2=head["auc_D2"], auc_O=head["auc_O"], delta=head["delta"],
                           higher_than_d2=bool(aucs[1] > head["auc_D2"]),
                           adjacent_drop=drops, monotone=bool(all(d <= 0.01 for d in drops)),
                           recovery_k1=recs[1], kstar_0_5=kstar(means, 0.5), kstar_0_8=kstar(means, 0.8)))
    h11_n = sum(d["higher_than_d2"] for d in detail)
    h12_n = sum(d["monotone"] for d in detail)

    def band(n):
        if n == 7:
            return "支持"
        if 4 <= n <= 6:
            return "部分支持"
        return "不支持"

    squat = [d for d in detail if d["dataset"] == "EC3D" and d["act"] == "SQUAT" and d["ruler"] == "positions"][0]
    lunge = [d for d in detail if d["dataset"] == "EC3D" and d["act"] == "Lunges" and d["ruler"] == "positions"][0]
    h13 = "支持" if squat["recovery_k1"] >= 0.5 else "不支持"
    h14 = "支持" if rank_k(lunge["kstar_0_8"]) >= rank_k(squat["kstar_0_8"]) else "不支持"
    h15_sets = {}
    h15_ok = {}
    per_df = pd.DataFrame(per_rows)
    for ses in ("rd", "rgs"):
        sub = per_df[(per_df.dataset == "GAITEX") & (per_df.act == ses) & (per_df.protocol == "P3")]
        if sub.tpr.isna().any() or sub.fpr.isna().any() or len(sub) == 0:
            h15_ok[ses] = None
            h15_sets[ses] = {"n": int(len(sub)), "n_tpr_below_0.1": None, "n_fpr_above_0.5": None}
            continue
        nt = int((sub.tpr < 0.1).sum())
        nf = int((sub.fpr > 0.5).sum())
        h15_sets[ses] = {"n": int(len(sub)), "n_tpr_below_0.1": nt, "n_fpr_above_0.5": nf}
        h15_ok[ses] = nt <= 1 and nf <= 1
    if any(v is None for v in h15_ok.values()):
        h15 = "无法判定"
    else:
        n_ok = sum(bool(v) for v in h15_ok.values())
        h15 = "支持" if n_ok == 2 else ("部分支持" if n_ok == 1 else "不支持")
    if any(not r["applicable"] for r in rec_rows if (r["dataset"], r["act"], r["ruler"]) in APPLICABLE):
        # A preregistered cell is not applicable under the recomputed delta. Stop before a verdict.
        json.dump({"passed": False, "reason": "a preregistered cell has delta < 0.05", "detail": detail},
                  open(os.path.join(OUT, "prereg_check.json"), "w"), indent=2)
        print("STOP applicable set mismatch", flush=True)
        sys.exit(2)

    check = {
        "H11": band(h11_n), "H11_n": h11_n,
        "H12": band(h12_n), "H12_n": h12_n,
        "H13": h13, "H14": h14, "H15": h15,
        "H15_sets": h15_sets,
        "cells": detail,
        "n_reps": N_REP, "seed": 0,
    }
    pooled.to_csv(os.path.join(OUT, "pooled.csv"), index=False)
    pd.DataFrame(rec_rows).to_csv(os.path.join(OUT, "recovery.csv"), index=False)
    per_df.to_csv(os.path.join(OUT, "per_person.csv"), index=False)
    json.dump(check, open(os.path.join(OUT, "prereg_check.json"), "w"), indent=2)
    outputs = [os.path.join(OUT, f) for f in
               ["pooled.csv", "recovery.csv", "per_person.csv", "prereg_check.json", "selfcheck.json"]]
    update_manifest(os.path.abspath(__file__), outputs, t0, C.now_iso())
    print("K1 DONE", check["H11"], check["H12"], check["H13"], check["H14"], check["H15"],
          "seconds", round(time.time() - t_start, 3), flush=True)


if __name__ == "__main__":
    main()
