"""PRL supplements on EC3D, using the paper's alignment and bone-pair angles.

Scheme 1: rank correlation and equal-FPR decision flips between joint positions
and bone angles, before and after per-coordinate scale calibration.
Scheme 2: inject a known knee bend into correct reps and test whether the
error energy tracks it, while the correct-set style term stays put.
"""
import itertools, json, os, pickle
import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import spearmanr

HERE = "/data1/users/yangrui/projects/idea67b"
RAW = os.path.join(HERE, "data/ec3d/data_3D.pickle")
OUT = os.path.join(HERE, "exp/prl_supplement")
os.makedirs(OUT, exist_ok=True)

JOINTS = list(range(15)) + [19, 21, 22, 24]
BONES = [(1, 0), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8),
         (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14), (14, 15),
         (14, 16), (11, 17), (11, 18)]
INC = {}
for a, b in BONES:
    INC.setdefault(a, []).append(b)
    INC.setdefault(b, []).append(a)
PAIRS = [(j, u, v) for j in sorted(INC) for u, v in itertools.combinations(sorted(INC[j]), 2)]
LEG = {10, 11, 13, 14, 15, 16, 17, 18}


@njit
def _dtw(C):
    Ta, Tb = C.shape
    D = np.full((Ta + 1, Tb + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, Ta + 1):
        for j in range(1, Tb + 1):
            m = D[i - 1, j - 1]
            if D[i - 1, j] < m:
                m = D[i - 1, j]
            if D[i, j - 1] < m:
                m = D[i, j - 1]
            D[i, j] = C[i - 1, j - 1] + m
    i, j = Ta, Tb
    n = 0
    pi = np.empty(Ta + Tb, np.int64)
    pj = np.empty(Ta + Tb, np.int64)
    while i > 0 and j > 0:
        pi[n] = i - 1
        pj[n] = j - 1
        n += 1
        a, b, c = D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]
        if a <= b and a <= c:
            i -= 1
            j -= 1
        elif b <= c:
            i -= 1
        else:
            j -= 1
    return D[Ta, Tb], pi[:n][::-1].copy(), pj[:n][::-1].copy()


def cost(a, b):
    return np.linalg.norm(a[:, None] - b[None], axis=-1).mean(-1)


def align_to(x, template):
    tot, pi, pj = _dtw(cost(template, x))
    out = np.zeros_like(template)
    cnt = np.zeros(len(template))
    for a, b in zip(pi, pj):
        out[a] += x[b]
        cnt[a] += 1
    return out / np.maximum(cnt[:, None, None], 1), tot / max(len(pi), 1)


def angles_of(X):
    out = []
    for j, u, v in PAIRS:
        a = X[:, u] - X[:, j]
        b = X[:, v] - X[:, j]
        a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)
        b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9)
        out.append(np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1))))
    return np.stack(out, 1)


def flat(X):
    return X.reshape(len(X), -1)


def load_reps():
    d = pickle.load(open(RAW, "rb"))
    lab = pd.DataFrame(d["labels"], columns=["act", "sub", "lab", "rep", "frame"])
    lab[["lab", "rep"]] = lab[["lab", "rep"]].astype(int)
    poses = d["poses"][:, :, JOINTS]
    reps = {}
    for k, idx in lab.groupby(["act", "sub", "lab", "rep"]).groups.items():
        x = np.ascontiguousarray(poses[np.asarray(idx)].transpose(0, 2, 1)).astype(np.float64)
        reps[k] = x - x[:, 8:9, :]
    return reps


def align_all(reps):
    aligned = {}
    for a in sorted({k[0] for k in reps}):
        cor = [k for k in reps if k[0] == a and k[2] == 1]
        best, bestd = None, 1e18
        for k in cor:
            d = 0.0
            for q in cor:
                if q == k:
                    continue
                d += align_to(reps[q], reps[k])[1]
            if d < bestd:
                bestd, best = d, k
        tmpl = reps[best]
        for k, x in reps.items():
            if k[0] == a:
                aligned[k] = align_to(x, tmpl)[0]
        print("aligned", a, flush=True)
    return aligned


def scores(feats):
    """Distance of each rep to its own leave-one-out correct mean."""
    rows = []
    keys = list(feats)
    for k, x in feats.items():
        act, sub, lab, rep = k
        own = [feats[q] for q in keys if q[0] == act and q[1] == sub and q[2] == 1 and q != k]
        if len(own) < 1:
            continue
        ref = np.mean(own, 0)
        rows.append(dict(act=act, sub=sub, lab=int(lab), rep=int(rep),
                         score=float(np.mean((x - ref) ** 2))))
    return pd.DataFrame(rows)


def zscore_feats(feats):
    """Per-coordinate scale calibration from correct reps of the same action."""
    out = {}
    for a in sorted({k[0] for k in feats}):
        cor = [feats[k] for k in feats if k[0] == a and k[2] == 1]
        sd = np.std(np.concatenate(cor, 0), 0) + 1e-6
        mu = np.mean(np.concatenate(cor, 0), 0)
        for k, x in feats.items():
            if k[0] == a:
                out[k] = (x - mu) / sd
    return out


def spearman_ci(df, n_boot=1000, seed=6705):
    rng = np.random.default_rng(seed)
    wide = df.pivot_table(index=["act", "sub", "lab", "rep"], columns="view", values="score")
    wide = wide.dropna()
    rho, p = spearmanr(wide["positions"], wide["bone angles"])
    subs = wide.index.get_level_values("sub").unique()
    boots = []
    for _ in range(n_boot):
        take = rng.choice(subs, len(subs), replace=True)
        parts = [wide.xs(s, level="sub") for s in take]
        # xs drops the subject level; stack with a fake index
        cat = pd.concat(parts, ignore_index=True)
        if len(cat) < 5:
            continue
        r, _ = spearmanr(cat["positions"], cat["bone angles"])
        if np.isfinite(r):
            boots.append(r)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n=int(len(wide)), spearman=float(rho), p=float(p),
                ci_lo=float(lo), ci_hi=float(hi), n_boot=len(boots))


def flip_rate(df, fpr=0.10):
    """Equal false-positive thresholds, fit on correct reps, applied to all reps."""
    wide = df.pivot_table(index=["act", "sub", "lab", "rep"], columns="view", values="score").dropna()
    cor = wide[wide.index.get_level_values("lab") == 1]
    thr_p = np.quantile(cor["positions"], 1 - fpr)
    thr_a = np.quantile(cor["bone angles"], 1 - fpr)
    dec_p = wide["positions"] > thr_p
    dec_a = wide["bone angles"] > thr_a
    flip = dec_p != dec_a
    inc = wide.index.get_level_values("lab") != 1
    return dict(
        fpr_target=fpr,
        flip_all=float(flip.mean()),
        flip_correct=float(flip[~inc].mean()),
        flip_incorrect=float(flip[inc].mean()) if inc.any() else None,
        n_all=int(len(wide)),
        n_incorrect=int(inc.sum()),
    )


def bend_knee(X, hip, knee, ankle, children, deg, frac=(0.25, 0.75)):
    """Rotate the shank and everything distal around the knee, middle portion of the rep."""
    Y = X.copy()
    T = len(Y)
    a0, a1 = int(frac[0] * T), int(frac[1] * T)
    rad = np.radians(deg)
    for t in range(a0, a1):
        v1 = Y[t, knee] - Y[t, hip]
        v2 = Y[t, ankle] - Y[t, knee]
        axis = np.cross(v1, v2)
        n = np.linalg.norm(axis)
        if n < 1e-8:
            continue
        axis = axis / n
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(rad) * K + (1 - np.cos(rad)) * (K @ K)
        for j in children:
            Y[t, j] = Y[t, knee] + R @ (Y[t, j] - Y[t, knee])
    return Y


def joint_energy(A, B):
    """Mean squared displacement per joint. A,B (T,J,3)."""
    return np.mean(np.sum((A - B) ** 2, axis=2), 0)


def main():
    reps = load_reps()
    aligned = align_all(reps)
    pos = {k: flat(v) for k, v in aligned.items()}
    ang = {k: angles_of(v) for k, v in aligned.items()}

    # ----- scheme 1 -----
    raw_scores = pd.concat([
        scores(pos).assign(view="positions"),
        scores(ang).assign(view="bone angles"),
    ], ignore_index=True)
    z_scores = pd.concat([
        scores(zscore_feats(pos)).assign(view="positions"),
        scores(zscore_feats(ang)).assign(view="bone angles"),
    ], ignore_index=True)

    scheme1 = {"raw": {}, "zscored": {}}
    for name, df in [("raw", raw_scores), ("zscored", z_scores)]:
        scheme1[name]["overall"] = spearman_ci(df)
        scheme1[name]["flip"] = flip_rate(df)
        scheme1[name]["by_act"] = {}
        for a, g in df.groupby("act"):
            scheme1[name]["by_act"][a] = {**spearman_ci(g, n_boot=400), **{k: v for k, v in flip_rate(g).items()}}
    raw_scores.to_csv(os.path.join(OUT, "scheme1_scores_raw.csv"), index=False)
    json.dump(scheme1, open(os.path.join(OUT, "scheme1_summary.json"), "w"), indent=2)
    print("SCHEME1", json.dumps(scheme1["raw"]["overall"]), json.dumps(scheme1["raw"]["flip"]))
    print("SCHEME1z", json.dumps(scheme1["zscored"]["overall"]), json.dumps(scheme1["zscored"]["flip"]))

    # ----- scheme 2: known knee bend on correct squats, after alignment -----
    rows = []
    for deg in (5, 10, 15):
        energies = []
        for k, X in aligned.items():
            if k[0] != "SQUAT" or k[2] != 1:
                continue
            Y = bend_knee(X, 9, 10, 11, [11, 17, 18], deg)
            Y = bend_knee(Y, 12, 13, 14, [14, 15, 16], deg)
            e = joint_energy(Y, X)
            frac = float(e[list(LEG)].sum() / max(e.sum(), 1e-12))
            energies.append(float(e.sum()))
            rows.append(dict(deg=deg, sub=k[1], rep=int(k[3]), energy=float(e.sum()), leg_frac=frac))
        print("perturb", deg, "median energy", float(np.median(energies)), flush=True)
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(OUT, "scheme2_perturb.csv"), index=False)
    # correlation of per-rep energy with intended degrees
    rho, p = spearmanr(P.deg, P.energy)
    # Noise-corrected error energy, and what happens if the bent reps are
    # mislabeled as correct (they inflate within-person variance).
    decomp = []
    squat = {k: v for k, v in aligned.items() if k[0] == "SQUAT" and k[2] == 1}
    by_sub = {}
    for k, X in squat.items():
        by_sub.setdefault(k[1], []).append(X)
    for deg in (5, 10, 15):
        m2s, inflate = [], []
        for sub, xs in by_sub.items():
            if len(xs) < 2:
                continue
            bent = [bend_knee(bend_knee(X, 9, 10, 11, [11, 17, 18], deg), 12, 13, 14, [14, 15, 16], deg) for X in xs]
            mu_c = np.mean(xs, 0)
            mu_b = np.mean(bent, 0)
            s2_c = float(np.mean([(np.mean((x - mu_c) ** 2)) for x in xs]) * len(xs) / (len(xs) - 1))
            s2_b = float(np.mean([(np.mean((x - mu_b) ** 2)) for x in bent]) * len(bent) / (len(bent) - 1))
            m2 = float(np.mean((mu_b - mu_c) ** 2) - s2_b / len(bent) - s2_c / len(xs))
            pool = xs + bent
            mu_p = np.mean(pool, 0)
            s2_p = float(np.mean([(np.mean((x - mu_p) ** 2)) for x in pool]) * len(pool) / (len(pool) - 1))
            m2s.append(m2)
            inflate.append(s2_p / max(s2_c, 1e-12))
        decomp.append(dict(deg=deg, median_m2=float(np.median(m2s)), median_variance_inflation_if_mislabeled=float(np.median(inflate))))
    scheme2 = dict(
        spearman_energy_vs_deg=float(rho),
        p=float(p),
        median_leg_fraction=float(P.leg_frac.median()),
        leg_fraction_is_by_construction=True,
        decomp=decomp,
        by_deg=P.groupby("deg")[["energy", "leg_frac"]].median().to_dict(),
        note="Knee bend after alignment, both knees, middle 50 percent of frames. leg_frac is 1 by construction because only those joints move. decomp.median_m2 is the noise-corrected error energy when the bent reps are labeled as a mistake. variance_inflation is within-person variance if those reps are instead pooled with the correct reps.",
    )
    json.dump(scheme2, open(os.path.join(OUT, "scheme2_summary.json"), "w"), indent=2)
    print("SCHEME2", json.dumps(scheme2))
    print("DONE")


if __name__ == "__main__":
    main()
