"""GAITEX repetitions, grouped the way the paper's estimator expects.

Two exercise sessions carry a correct variant and three named error variants each, with
about ten repetitions per variant per person across nineteen and eighteen people. That is
the first dataset where both terms of the decomposition are well supported: EC3D estimates
between-performer style from three.

Representation is joint angles rather than positions. The inverse kinematics here is
IMU-driven and carries no global translation -- the pelvis translation columns are
constant -- so positions are not recoverable without forward kinematics through the
OpenSim models. Angles are also scale-free, which removes the bone-length normalisation
that IRDS needed. The ratio is dimensionless either way, but it is not the same estimand
as the position-based one, because angles weight joints differently; that is stated rather
than hidden.
"""
import os, re, glob, pickle
import numpy as np, pandas as pd

ROOT = __import__('os').environ.get('IDEA67B_GAITEX_ROOT', str(__import__('pathlib').Path(__file__).resolve().parents[1] / 'data' / 'gaitex' / 'sel' / 'data'))
CACHE = str((__import__('pathlib').Path(__file__).resolve().parents[1] / 'exp' / 'cache' / ('gaitex_cells_%s.pkl' % __import__('os').environ.get('COORDS', 'observed'))))
T_RS = 60
LABELS = {'rd': ['rd_correct', 'rd_toes', 'rd_pronation', 'rd_supination'],
          'rgs': ['rgs_correct', 'rgs_flexion', 'rgs_abduction', 'rgs_stork']}
# Only nine segments carry sensors -- pelvis, torso, both thighs, both shanks, both feet
# and the right toes -- so the model's upper-limb coordinates are unobserved and inverse
# kinematics leaves them wherever the solver happens to land. A reliability check on the
# static calibration poses, which each subject repeats once per session with fresh sensor
# placement, puts those coordinates at ICC 0.06 to 0.27 with a within-subject spread of 36
# to 41 degrees, and they carry 91% of the within-subject variance. Including them would
# charge solver drift to between-performer style. Both exercises are lower-limb anyway.
#
# Pelvis translation is constant under IMU-driven inverse kinematics and the knee beta
# coordinate is a dependent constraint rather than a measured angle.
DROP_ALL = re.compile(r'^(time|pelvis_t[xyz]|knee_angle_[rl]_beta)$')
DROP_UNOBSERVED = re.compile(r'^(time|pelvis_t[xyz]|knee_angle_[rl]_beta'
                             r'|arm_.*|elbow_.*|wrist_.*|pro_sup_.*|neck_.*)$')
DROP = DROP_ALL if os.environ.get('COORDS') == 'all' else DROP_UNOBSERVED
sq = lambda d: (np.linalg.norm(d, axis=-1) ** 2).mean()


def read_mot(p):
    with open(p) as f:
        while f.readline().strip() != 'endheader': pass
        cols = f.readline().split()
        A = np.loadtxt(f)
    keep = [i for i, c in enumerate(cols) if not DROP.match(c)]
    return A[:, 0], A[:, keep], [cols[i] for i in keep]


def resample(X, T):
    s, t = np.linspace(0, 1, len(X)), np.linspace(0, 1, T)
    return np.stack([np.interp(t, s, X[:, j]) for j in range(X.shape[1])], 1)


def dtw_onto(X, ref):
    C = ((X[:, None] - ref[None]) ** 2).sum(-1)
    n, m = C.shape
    D = np.full((n + 1, m + 1), np.inf); D[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = C[i - 1, j - 1] + min(D[i - 1, j - 1], D[i - 1, j], D[i, j - 1])
    i, j = n, m; out = np.zeros_like(ref); w = np.zeros(len(ref))
    while i > 0 and j > 0:
        out[j - 1] += X[i - 1]; w[j - 1] += 1
        k = int(np.argmin([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]]))
        i, j = [(i - 1, j - 1), (i - 1, j), (i, j - 1)][k]
    return out / np.maximum(w, 1)[:, None]


def build_cells(force=False, sessions=('rd', 'rgs')):
    cache = CACHE.replace('.pkl', '_' + '-'.join(sessions) + '.pkl')
    if os.path.exists(cache) and not force:
        g = pickle.load(open(cache, 'rb'))
        return {k: v for k, v in g.items() if k in sessions}
    groups = {}
    for ses in sessions:
        cells, names = {}, None
        for sub_dir in sorted(glob.glob(os.path.join(ROOT, '*', ses))):
            sub = sub_dir.split('/')[-2]
            mot = glob.glob(os.path.join(sub_dir, 'ik_imus/results_imu_ik/ik_*.mot'))
            ts = glob.glob(os.path.join(sub_dir, 'timestamps_*.csv'))
            mot = [m for m in mot if 'orientationErrors' not in m]
            if not mot or not ts: continue
            t, A, names = read_mot(mot[0])
            S = pd.read_csv(ts[0]); S.columns = [c.strip() for c in S.columns]
            for _, r in S.iterrows():
                if r.label not in LABELS[ses]: continue
                lab = LABELS[ses].index(r.label) + 1
                i0, i1 = np.searchsorted(t, [r['temporal_segment_start_[s]'],
                                             r['temporal_segment_end_[s]']])
                if i1 - i0 < 20: continue
                cells.setdefault((sub, lab), []).append(resample(A[i0:i1], T_RS))
        subs = sorted({s for (s, l) in cells if l == 1 and len(cells[(s, 1)]) >= 2})
        if len(subs) < 3: continue
        pool = [x for s in subs for x in cells[(s, 1)]]
        ref = pool[int(np.argmin([np.mean([sq(a - b) for b in pool]) for a in pool]))]
        groups[ses] = {k: [dtw_onto(x, ref)[:, :, None] for x in v] for k, v in cells.items()}
        groups[ses + '__names'] = names
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    pickle.dump(groups, open(cache, 'wb'))
    return {k: v for k, v in groups.items() if k in sessions}
