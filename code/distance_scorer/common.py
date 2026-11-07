"""Shared helpers for prl_v2. Imports existing functions; does not modify them."""
import ast
import hashlib
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

CODE = "/data1/users/yangrui/projects/idea67b/experiment/code"
sys.path.insert(0, CODE)
import prl_scheme12 as P  # noqa: E402
import prl_scheme2b as S2  # noqa: E402
from ec3d_common import (  # noqa: E402
    BONES, LABEL_NAME, align_to, dtw, load_reps, root_center,
)

EXP = "/data1/users/yangrui/projects/idea67b/experiment/runs/prl_v2"
ALPHA = 0.10
SEED = 0
VIEWS = ("positions", "bone angles")
ACTS_CN = {"Lunges": "弓步", "Plank": "平板", "SQUAT": "深蹲"}


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def auc(err, cor):
    """Mann-Whitney. err = positive class. Ties count 0.5."""
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


def decide(scores, tau):
    return np.asarray(scores, float) > tau


def score_mse(x, ref):
    return float(np.mean((x - ref) ** 2))


def load_function(path, name, env):
    """Exec one function definition. Avoids importing scripts that write results."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(keep) != 1:
        raise RuntimeError("function %s not found in %s" % (name, path))
    mod = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(mod)
    exec(compile(mod, path, "exec"), env)
    return env[name]


def a2_estimate():
    env = {"np": np, "sq": lambda d: (np.linalg.norm(d, axis=-1) ** 2).mean(), "LABEL_NAME": LABEL_NAME}
    return load_function(os.path.join(CODE, "a2_ec3d_decomp_v2.py"), "estimate", env)


def bone_pairs():
    inc = {}
    for a, b in BONES:
        inc.setdefault(a, []).append(b)
        inc.setdefault(b, []).append(a)
    return [(j, u, v) for j in sorted(inc) for u, v in __import__("itertools").combinations(sorted(inc[j]), 2)]


def angles_of_ec3d(X, pairs):
    """Same construction as ec3d_representation.angles, without importing that script."""
    out = []
    for j, u, v in pairs:
        a = X[:, u] - X[:, j]
        b = X[:, v] - X[:, j]
        a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)
        b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-9)
        out.append(np.degrees(np.arccos(np.clip((a * b).sum(-1), -1.0, 1.0))))
    return np.stack(out, 1)[..., None]


DISTAL = [10, 11, 13, 14, 15, 16, 17, 18]


def update_manifest(script, outputs, t0, t1, seed=SEED):
    man_path = os.path.join(EXP, "MANIFEST.json")
    man = {"experiments": []}
    if os.path.exists(man_path):
        man = json.load(open(man_path, encoding="utf-8"))
    import numpy
    entry = {
        "script": script,
        "script_md5": md5_file(script),
        "outputs": [{"path": p, "md5": md5_file(p)} for p in outputs if os.path.exists(p)],
        "seed": seed,
        "start": t0,
        "end": t1,
        "python": sys.version,
        "numpy": numpy.__version__,
    }
    man["experiments"] = [e for e in man.get("experiments", []) if e.get("script") != script]
    man["experiments"].append(entry)
    json.dump(man, open(man_path, "w"), indent=2, ensure_ascii=False)
    return man_path
