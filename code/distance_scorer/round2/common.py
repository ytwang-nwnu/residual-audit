"""Round-2 helpers. Does not import scripts that write results on import."""
import ast
import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

CODE = "/data1/users/yangrui/projects/idea67b/experiment/code"
sys.path.insert(0, CODE)
import prl_scheme12 as P  # noqa: E402
from ec3d_common import BONES  # noqa: E402

# DISTAL lives in ec3d_representation.py, which writes results on import.
DISTAL = [10, 11, 13, 14, 15, 16, 17, 18]
EXP = "/data1/users/yangrui/projects/idea67b/experiment/runs/prl_v2/round2"
ALPHA = 0.10
SEED = 0


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def auc(err, cor):
    err = np.asarray(err, float)
    cor = np.asarray(cor, float)
    if len(err) == 0 or len(cor) == 0:
        return float("nan")
    allv = np.concatenate([err, cor])
    ranks = pd.Series(allv).rank(method="average").to_numpy()
    rpos = ranks[: len(err)].sum()
    npos, nneg = len(err), len(cor)
    return float((rpos - npos * (npos + 1) / 2.0) / (npos * nneg))


def tau_of(correct_scores):
    correct_scores = np.asarray(correct_scores, float)
    if len(correct_scores) == 0:
        return float("nan")
    return float(np.quantile(correct_scores, 1.0 - ALPHA))


def score_mse(x, ref):
    return float(np.mean((np.asarray(x) - np.asarray(ref)) ** 2))


def load_function(path, name, env):
    tree = ast.parse(open(path, encoding="utf-8").read())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(keep) != 1:
        raise RuntimeError("%s not in %s" % (name, path))
    mod = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, path, "exec"), env)
    return env[name]


def distances():
    env = {"np": np}
    resample = load_function(os.path.join(CODE, "e1_refcorr.py"), "resample", env)
    env["resample"] = resample
    d_lin = load_function(os.path.join(CODE, "e4_loso_helpers.py"), "d_lin", env)

    def d_sq(a, b):
        return float(np.mean((resample(a, 100) - resample(b, 100)) ** 2))

    return d_lin, d_sq


def bone_pairs_23():
    inc = {}
    for a, b in BONES:
        inc.setdefault(a, []).append(b)
        inc.setdefault(b, []).append(a)
    return [(j, u, v) for j in sorted(inc) for u, v in __import__("itertools").combinations(sorted(inc[j]), 2)]


def angle_between(X, i0, i1, j0, j1):
    a = X[:, i1] - X[:, i0]
    b = X[:, j1] - X[:, j0]
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1.0, 1.0)))


def angles_23(X, pairs):
    cols = []
    for j, u, v in pairs:
        cols.append(angle_between(X, j, u, j, v))
    return np.stack(cols, 1)


# trunk midhip->neck with left/right thigh and left/right upper arm
EXTRA = [(8, 1, 12, 13), (8, 1, 9, 10), (8, 1, 5, 6), (8, 1, 2, 3)]


def angles_27(X, pairs):
    base = angles_23(X, pairs)
    extra = [angle_between(X, *e) for e in EXTRA]
    return np.concatenate([base, np.stack(extra, 1)], 1)


def update_manifest(script, outputs, t0, t1, seed=SEED):
    man_path = os.path.join(EXP, "MANIFEST.json")
    man = {"experiments": []}
    if os.path.exists(man_path):
        man = json.load(open(man_path, encoding="utf-8"))
    entry = {
        "script": script,
        "script_md5": md5_file(script),
        "outputs": [{"path": p, "md5": md5_file(p)} for p in outputs if os.path.exists(p)],
        "seed": seed,
        "start": t0,
        "end": t1,
        "python": sys.version,
        "numpy": np.__version__,
    }
    man["experiments"] = [e for e in man.get("experiments", []) if e.get("script") != script]
    man["experiments"].append(entry)
    json.dump(man, open(man_path, "w"), indent=2, ensure_ascii=False)
