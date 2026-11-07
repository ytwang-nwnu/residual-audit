"""Shared EC3D helpers: loading repetitions, fast DTW (numba), alignment to a template, bones."""
import os, pickle
import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
EC3D_RAW = os.path.join(HERE, 'zhao22', 'data', 'EC3D', 'data_3D.pickle')
JOINTS = list(range(15)) + [19, 21, 22, 24]  # Zhao'22 subset of OpenPose BODY_25 (19 joints)
# indices inside the 19-joint subset: 0 Nose 1 Neck 2 RSho 3 RElb 4 RWri 5 LSho 6 LElb 7 LWri 8 MidHip
# 9 RHip 10 RKnee 11 RAnk 12 LHip 13 LKnee 14 LAnk 15 LBigToe 16 LHeel 17 RBigToe 18 RHeel
BONES = [(1, 0), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (10, 11),
         (8, 12), (12, 13), (13, 14), (14, 15), (14, 16), (11, 17), (11, 18)]
LABEL_NAME = {1: 'Correct', 2: 'Feet too wide', 3: 'Knees inward', 4: 'Not low enough', 5: 'Front bended',
              6: 'Knees pass toes', 7: 'Banana back', 8: 'Rolled back', 9: 'Asymmetric', 10: 'Unknown'}


def load_reps():
    d = pickle.load(open(EC3D_RAW, 'rb'))
    lab = pd.DataFrame(d['labels'], columns=['act', 'sub', 'lab', 'rep', 'frame'])
    lab[['lab', 'rep']] = lab[['lab', 'rep']].astype(int)
    poses = d['poses'][:, :, JOINTS]  # (N,3,19)
    reps = {}
    for k, idx in lab.groupby(['act', 'sub', 'lab', 'rep']).groups.items():
        reps[k] = np.ascontiguousarray(poses[np.asarray(idx)].transpose(0, 2, 1)).astype(np.float64)  # (T,19,3)
    return reps


def zhao_to_tj3(x57):
    """(57,T) Zhao layout -> (T,19,3)"""
    return np.ascontiguousarray(x57.reshape(3, 19, -1).transpose(2, 1, 0))


def root_center(x):
    return x - x[:, 8:9, :]


@njit(cache=True)
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
    # backtrack
    i, j = Ta, Tb
    pi = np.empty(Ta + Tb, np.int64); pj = np.empty(Ta + Tb, np.int64); n = 0
    while i > 0 and j > 0:
        pi[n] = i - 1; pj[n] = j - 1; n += 1
        a, b, c = D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]
        if a <= b and a <= c:
            i -= 1; j -= 1
        elif b <= c:
            i -= 1
        else:
            j -= 1
    return D[Ta, Tb], pi[:n][::-1].copy(), pj[:n][::-1].copy()


def cost(a, b):
    """mean per-joint Euclidean distance between all frame pairs; a:(Ta,J,3) b:(Tb,J,3)"""
    return np.linalg.norm(a[:, None] - b[None], axis=-1).mean(-1)


def dtw(a, b):
    """returns (mean aligned per-joint distance, path_i, path_j)"""
    tot, pi, pj = _dtw(cost(a, b))
    return tot / len(pi), pi, pj


def align_to(x, template):
    """DTW-align x onto template's time axis: for each template frame, mean of x frames mapped to it."""
    _, pi, pj = dtw(template, x)
    out = np.zeros_like(template); cnt = np.zeros(len(template))
    for a, b in zip(pi, pj):
        out[a] += x[b]; cnt[a] += 1
    return out / cnt[:, None, None]


def bone_lengths(x):
    """x:(T,19,3) -> (T, n_bones)"""
    return np.stack([np.linalg.norm(x[:, i] - x[:, j], axis=-1) for i, j in BONES], -1)
