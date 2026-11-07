"""PREREG_R8 self-check: s2 unchanged; s3/s4 well-formed."""
import sys, hashlib, numpy as np, yaml
sys.path.insert(0, "src")
import ec3d

cfg = yaml.safe_load(open("configs/m1.yaml"))
segs = ec3d.load_segments(cfg["data_pickle"])

def digest(items):
    h = hashlib.md5()
    for s, x in items:
        h.update(s["id"].encode()); h.update(np.ascontiguousarray(x, dtype=np.float64).tobytes())
    return h.hexdigest()

# 1) s2 digest (compare against pre-patch value recorded in git stash? -> recompute with HEAD copy)
import subprocess, importlib.util, os, tempfile
head_src = subprocess.run(["git", "show", "HEAD:src/ec3d.py"], capture_output=True, text=True).stdout
tmp = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False); tmp.write(head_src); tmp.close()
spec = importlib.util.spec_from_file_location("ec3d_head", tmp.name); old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
for fold in range(4):
    d_new = ec3d.make_draws(segs, 0); d_old = old.make_draws(segs, 0)
    assert d_new == d_old, "draws changed"
    tr_n, c_n = ec3d.prepare_fold(segs, "s2", fold, d_new)
    tr_o, c_o = old.prepare_fold(segs, "s2", fold, d_old)
    assert digest(tr_n) == digest(tr_o) and len(c_n) == len(c_o)
    for a, b in zip(c_n, c_o):
        assert a["name"] == b["name"] and digest(a["items"]) == digest(b["items"])
    tr1, c1 = ec3d.prepare_fold(segs, "s1", fold)
    tr1o, c1o = old.prepare_fold(segs, "s1", fold)
    assert digest(tr1) == digest(tr1o) and digest(c1[0]["items"]) == digest(c1o[0]["items"])
print("s1/s2 byte-identical to HEAD: OK")

# 2) s3: training == s1 training; s3_raw == s1 test; s3_all == s2_all items; s3_k == s2_k items
for fold in range(4):
    d = ec3d.make_draws(segs, 0)
    tr1, c1 = ec3d.prepare_fold(segs, "s1", fold)
    tr2, c2 = ec3d.prepare_fold(segs, "s2", fold, d)
    tr3, c3 = ec3d.prepare_fold(segs, "s3", fold, d)
    assert digest(tr3) == digest(tr1), "s3 training must equal s1 training"
    assert c3[0]["name"] == "s3_raw" and digest(c3[0]["items"]) == digest(c1[0]["items"])
    m2 = {c["name"].replace("s2", "X") + str(c["draw"]): digest(c["items"]) for c in c2}
    m3 = {c["name"].replace("s3", "X") + str(c["draw"]): digest(c["items"]) for c in c3[1:]}
    assert m2 == m3, "s3 test conditions must equal s2 test conditions"
print("s3: train=s1, test conds=s2: OK  (conds per fold = %d)" % len(c3))

# 3) s4: label-free. training centred on all-rep means; draws may contain error reps
for fold in range(4):
    d4 = ecd = ec3d.make_draws(segs, 0, pool="all")
    tr4, c4 = ec3d.prepare_fold(segs, "s4", fold, d4)
    assert digest(tr4) != digest(tr1)
    labs = {s["id"]: s["lab"] for s in segs}
    n_err_drawn = sum(labs[i] != 1 for v in d4.values() for lst in v for i in lst)
    assert n_err_drawn > 0, "label-free draws should include error reps"
    assert c4[0]["name"] == "s4_all" and len(c4) == 1 + 3 * 20
    # per training subject & action, centred training reps have ~zero mean (up to the added grand mean)
    tr_segs = [s for s, _ in tr4]
    for act in ec3d.actions(segs):
        for sub in {s["sub"] for s in tr_segs}:
            xs = np.stack([x for s, x in tr4 if s["act"] == act and s["sub"] == sub])
            glob = np.mean(np.stack([x for s, x in tr4 if s["act"] == act]), axis=0)
            assert np.allclose(xs.mean(0), glob, atol=1e-6)
print("s4: training all-rep centred, label-free draws contain error reps: OK  (conds per fold = %d)" % len(c4))
print("SELFCHECK_R8 PASS")
