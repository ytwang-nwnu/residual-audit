"""Shared EC3D data access for prl_v2 round7.

Same reading path as experiment/code/ec3d_common.py (data_3D.pickle,
19-joint Zhao'22 subset, labels grouped by (act, sub, lab, rep)).
Common input per PREREG_R7: per-frame subtract joint 8 (MidHip), then
linear resample to T=100 frames. No DTW, no rotation.
"""
import pickle
import numpy as np
import pandas as pd

JOINTS = list(range(15)) + [19, 21, 22, 24]
# 18 bone pairs over the 19-joint subset (indices into the subset)
BONES = [(1, 0), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8),
         (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14), (14, 15),
         (14, 16), (11, 17), (11, 18)]
ROOT_JOINT = 8
T_TARGET = 100


def seg_id(act, sub, lab, rep):
    return "{}|{}|{}|{}".format(act, sub, lab, rep)


def load_segments(pickle_path):
    """Return list of segment dicts with raw poses, sorted by id.

    Each: id, act, sub, lab, rep, err (lab != 1 -> error; lab 10 unknown
    counts as error), x raw (T,19,3) float64.
    """
    d = pickle.load(open(pickle_path, "rb"))
    lab = pd.DataFrame(d["labels"], columns=["act", "sub", "lab", "rep", "frame"])
    lab[["lab", "rep"]] = lab[["lab", "rep"]].astype(int)
    poses = d["poses"][:, :, JOINTS]
    segs = []
    for (act, sub, labv, rep), idx in lab.groupby(["act", "sub", "lab", "rep"]).groups.items():
        x = np.ascontiguousarray(poses[np.asarray(idx)].transpose(0, 2, 1)).astype(np.float64)
        segs.append(dict(id=seg_id(act, sub, labv, rep), act=act, sub=sub,
                         lab=int(labv), rep=int(rep), err=int(labv != 1), x=x))
    segs.sort(key=lambda s: s["id"])
    return segs


def subjects(segs):
    return sorted({s["sub"] for s in segs})


def actions(segs):
    return sorted({s["act"] for s in segs})


def preprocess(x):
    """Root-center (subtract joint 8 per frame) then linear resample to T_TARGET."""
    x = x - x[:, ROOT_JOINT:ROOT_JOINT + 1, :]
    T = len(x)
    if T == T_TARGET:
        return np.ascontiguousarray(x)
    t_old = np.linspace(0.0, 1.0, T)
    t_new = np.linspace(0.0, 1.0, T_TARGET)
    flat = x.reshape(T, -1)
    out = np.stack([np.interp(t_new, t_old, flat[:, i]) for i in range(flat.shape[1])], axis=1)
    return np.ascontiguousarray(out.reshape(T_TARGET, x.shape[1], x.shape[2]))


def split_train_test(segs, split, fold):
    """Return (train_segs, test_segs) for split in {s0, s1, s2}.

    s0: repo's own split (subjects 0,1,2 train; subject 3 test, sorted order).
    s1/s2: leave-one-person-out, fold = held-out subject index (sorted order).
    """
    subs = subjects(segs)
    if split == "s0":
        test_subs = {subs[3]}
    else:
        test_subs = {subs[fold]}
    train = [s for s in segs if s["sub"] not in test_subs]
    test = [s for s in segs if s["sub"] in test_subs]
    return train, test


def correct_mean(segs, act, sub=None, exclude_ids=()):
    """Mean preprocessed rep over correct segs of one action.

    sub None -> over all given segs; else restricted to that subject.
    exclude_ids: segment ids to leave out (LOO for S2-all).
    Returns (T,19,3) or None if empty.
    """
    xs = [preprocess(s["x"]) for s in segs
          if s["act"] == act and s["lab"] == 1 and (sub is None or s["sub"] == sub)
          and s["id"] not in exclude_ids]
    if not xs:
        return None
    return np.mean(np.stack(xs), axis=0)


def calibrate(x, minus, plus):
    """Personal calibration: subtract personal term, add back global term."""
    return x - minus + plus


def prepare_fold(segs, split, fold, draws=None):
    """Build (train_items, test_conditions) with preprocessed (+calibrated) reps.

    train_items / test items are (seg, X) with X (100,19,3) preprocessed;
    for s2 they are additionally personal-calibrated per PREREG_R7.

    test_conditions: list of dicts {name, k, draw, excluded_ids, items}.
      s0/s1 -> single condition named 's0'/'s1'.
      s2    -> 's2_all' plus 's2_k{k}' x 20 draws each.
    """
    train_segs, test_segs = split_train_test(segs, split, fold)
    test_sub = test_segs[0]["sub"]

    # preprocess each rep once
    X = {s["id"]: preprocess(s["x"]) for s in train_segs + test_segs}

    def cmean(pool, act, sub=None, exclude_ids=()):
        xs = [X[s["id"]] for s in pool
              if s["act"] == act and s["lab"] == 1
              and (sub is None or s["sub"] == sub) and s["id"] not in exclude_ids]
        return np.mean(np.stack(xs), axis=0) if xs else None

    if split in ("s0", "s1"):
        train_items = [(s, X[s["id"]]) for s in train_segs]
        cond = dict(name=split, k=None, draw=None, excluded_ids=[],
                    items=[(s, X[s["id"]]) for s in test_segs])
        return train_items, [cond]

    acts = actions(segs)

    def amean(pool, act, sub=None, exclude_ids=()):
        """Mean over ALL reps (any label): label-free counterpart of cmean (PREREG_R8 S4)."""
        xs = [X[s["id"]] for s in pool
              if s["act"] == act and (sub is None or s["sub"] == sub) and s["id"] not in exclude_ids]
        return np.mean(np.stack(xs), axis=0) if xs else None

    def calib_conds(prefix, own_fn, glob, draws):
        """Held-out conditions: <prefix>_all (own mean, LOO for the item itself where
        the pool contains it) + <prefix>_k{k} x draws (own mean from k drawn reps,
        drawn reps excluded from that draw's test set)."""
        conds = []
        items = [(s, calibrate(X[s["id"]], own_fn(s), glob[s["act"]])) for s in test_segs]
        conds.append(dict(name=prefix + "_all", k=None, draw=None, excluded_ids=[], items=items))
        for k in (1, 2, 4):
            n_draws = len(draws[(test_sub, acts[0], k)])
            for d_i in range(n_draws):
                items, excluded = [], []
                for act in acts:
                    drawn = draws[(test_sub, act, k)][d_i]
                    excluded.extend(drawn)
                    own = np.mean(np.stack([X[i] for i in drawn]), axis=0)
                    for s in test_segs:
                        if s["act"] != act or s["id"] in drawn:
                            continue
                        items.append((s, calibrate(X[s["id"]], own, glob[act])))
                conds.append(dict(name="{}_k{}".format(prefix, k), k=k, draw=d_i,
                                  excluded_ids=excluded, items=items))
        return conds

    if split == "s3":
        # PREREG_R8 S3: test-only calibration. Training identical to s1 (raw);
        # held-out person calibrated exactly as in s2 (own CORRECT mean, LOO for
        # correct items; k drawn correct reps). Also re-scores the raw s1 condition
        # from the same trained model ("s3_raw") as a within-run control.
        train_items = [(s, X[s["id"]]) for s in train_segs]
        glob_tr = {a: cmean(train_segs, a) for a in acts}
        raw = dict(name="s3_raw", k=None, draw=None, excluded_ids=[],
                   items=[(s, X[s["id"]]) for s in test_segs])
        own_fn = lambda s: (cmean(test_segs, s["act"], sub=test_sub, exclude_ids={s["id"]})
                            if s["lab"] == 1 else cmean(test_segs, s["act"], sub=test_sub))
        return train_items, [raw] + calib_conds("s3", own_fn, glob_tr, draws)

    if split == "s4":
        # PREREG_R8 S4: label-free calibration. Training persons centred on their
        # ALL-rep mean (+ training ALL-rep grand mean); held-out person centred on
        # the mean of ALL own reps (LOO for the item itself) or k drawn reps of any label.
        glob_all = {a: amean(train_segs, a) for a in acts}
        own_all = {(a, sub): amean(train_segs, a, sub=sub)
                   for a in acts for sub in {s["sub"] for s in train_segs}}
        train_items = [(s, calibrate(X[s["id"]], own_all[(s["act"], s["sub"])], glob_all[s["act"]]))
                       for s in train_segs]
        own_fn = lambda s: amean(test_segs, s["act"], sub=test_sub, exclude_ids={s["id"]})
        return train_items, calib_conds("s4", own_fn, glob_all, draws)

    # --- s2: personal calibration ---
    acts = actions(segs)
    train_items = []
    glob_tr = {a: cmean(train_segs, a) for a in acts}
    own_tr = {(a, s["sub"]): cmean(train_segs, a, sub=s["sub"])
              for a in acts for s in train_segs}
    for s in train_segs:
        x = calibrate(X[s["id"]], own_tr[(s["act"], s["sub"])], glob_tr[s["act"]])
        train_items.append((s, x))

    conds = []
    # s2_all: correct reps -> mean of own OTHER correct reps (LOO);
    #         error reps   -> mean of ALL own correct reps.
    items = []
    for s in test_segs:
        if s["lab"] == 1:
            own = cmean(test_segs, s["act"], sub=test_sub, exclude_ids={s["id"]})
        else:
            own = cmean(test_segs, s["act"], sub=test_sub)
        items.append((s, calibrate(X[s["id"]], own, glob_tr[s["act"]])))
    conds.append(dict(name="s2_all", k=None, draw=None, excluded_ids=[], items=items))

    # s2_k draws: held-out person's mean from k drawn own correct reps;
    # the drawn reps are excluded from that draw's test set.
    for k in (1, 2, 4):
        n_draws = len(draws[(test_sub, acts[0], k)])
        for d_i in range(n_draws):
            items = []
            excluded = []
            for act in acts:
                drawn = draws[(test_sub, act, k)][d_i]
                excluded.extend(drawn)
                own = np.mean(np.stack([X[i] for i in drawn]), axis=0)
                for s in test_segs:
                    if s["act"] != act or s["id"] in drawn:
                        continue
                    items.append((s, calibrate(X[s["id"]], own, glob_tr[act])))
            conds.append(dict(name="s2_k{}".format(k), k=k, draw=d_i,
                              excluded_ids=excluded, items=items))
    return train_items, conds


def make_draws(segs, seed, ks=(1, 2, 4), n_draws=20, pool="correct"):
    """S2/S3 calibration draws: draws[(sub, act, k)] = list of n_draws lists of seg ids.

    np.random.default_rng(seed), fixed iteration order so the draw sequence
    is identical across methods. pool="correct" (S2/S3, PREREG_R7) draws from the
    person's correct reps; pool="all" (S4, PREREG_R8) draws from all reps regardless
    of label, i.e. label-free calibration.
    """
    rng = np.random.default_rng(seed)
    subs = subjects(segs)
    acts = actions(segs)
    draws = {}
    for sub in subs:
        for act in acts:
            correct = [s["id"] for s in segs
                       if s["sub"] == sub and s["act"] == act and (pool == "all" or s["lab"] == 1)]
            for k in ks:
                lst = []
                for _ in range(n_draws):
                    take = rng.choice(len(correct), size=k, replace=False)
                    lst.append([correct[int(i)] for i in take])
                draws[(sub, act, k)] = lst
    return draws
