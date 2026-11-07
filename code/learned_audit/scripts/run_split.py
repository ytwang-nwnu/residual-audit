"""Run one (method, split, fold, seed) unit of PREREG_R7.

Usage (cwd = code dir):
  python scripts/run_split.py --config configs/m1.yaml --split s1 --fold 0 --seed 0 --run-dir <dir>

Writes into run dir: scores.csv, draws.json (s2), config.yaml, commit.txt,
log.txt (via shell tee from the launcher).
"""
import argparse, csv, json, os, subprocess, sys, threading, time
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import ec3d  # noqa: E402

METHODS = ["m1", "m2", "m3p", "m4", "m5"]  # m4/m5: PREREG_R7_amend1
SPLITS = ["s0", "s1", "s2", "s3", "s4"]  # s3/s4: PREREG_R8
# adapters whose score_items takes no torch device (TensorFlow methods)
NO_DEVICE_METHODS = ("m2", "m4", "m5")


def log(msg):
    print("[{}] {}".format(time.strftime("%H:%M:%S"), msg), flush=True)


class MemPoller(threading.Thread):
    """Sample this process's GPU memory via nvidia-smi; keep the max (MiB)."""

    def __init__(self, interval=5.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_flag = False
        self.max_mib = 0

    def run(self):
        pid = str(os.getpid())
        while not self.stop_flag:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=10).stdout
                for line in out.strip().splitlines():
                    p, mem = [x.strip() for x in line.split(",")[:2]]
                    if p == pid:
                        self.max_mib = max(self.max_mib, int(float(mem)))
            except Exception:
                pass
            time.sleep(self.interval)


BASE_COLS = ["split", "k", "draw", "id", "person", "exercise", "label", "score", "pred"]


def write_scores(path, rows):
    """rows: list of dicts with keys split,k,draw,id,person,exercise,label,score,pred
    plus optional method-specific extra keys (e.g. m1 pred_class), appended as
    extra columns after the base columns."""
    extra_cols = sorted({k for r in rows for k in r if k not in BASE_COLS})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(BASE_COLS + extra_cols)
        for r in rows:
            w.writerow([r["split"], r["k"] if r["k"] is not None else "",
                        r["draw"] if r["draw"] is not None else "",
                        r["id"], r["person"], r["exercise"], r["label"],
                        "{:.10g}".format(r["score"]), r["pred"]]
                       + [r.get(k, "") for k in extra_cols])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--fold", required=True, type=int)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--run-dir", required=True)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    method = cfg["method"]
    os.makedirs(a.run_dir, exist_ok=True)

    commit = subprocess.run(["git", "-C", cfg["code_root"], "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    open(os.path.join(a.run_dir, "commit.txt"), "w").write(commit + "\n")
    yaml.safe_dump({**cfg, "split": a.split, "fold": a.fold, "seed": a.seed,
                    "run_dir": a.run_dir, "commit": commit},
                   open(os.path.join(a.run_dir, "config.yaml"), "w"))

    t0 = time.time()
    poller = MemPoller()
    poller.start()
    segs = ec3d.load_segments(cfg["data_pickle"])
    draws = None
    if a.split in ("s2", "s3"):
        draws = ec3d.make_draws(segs, a.seed)                 # correct-rep draws (identical to s2)
    elif a.split == "s4":
        draws = ec3d.make_draws(segs, a.seed, pool="all")     # label-free draws
    if draws:
        json.dump({"|".join(map(str, k)): v for k, v in draws.items()},
                  open(os.path.join(a.run_dir, "draws.json"), "w"), indent=1)
    train_items, conds = ec3d.prepare_fold(segs, a.split, a.fold, draws)
    log("method={} split={} fold={} seed={} train={} conds={}".format(
        method, a.split, a.fold, a.seed, len(train_items), len(conds)))

    device = "cuda:0"
    if method == "m1":
        import m1_zhao as ad
        ad.init(cfg["third_party"]["zhao22"])
        model = ad.train_model(train_items, cfg["third_party"]["zhao22"], a.seed, device, log=log)
    elif method == "m2":
        import m2_stgcn as ad
        model = ad.train_model(train_items, ec3d.BONES, a.seed, a.run_dir, log=log)
    elif method == "m3p":
        import m3p_rehab as ad
        model = ad.train_model(train_items, cfg["third_party"]["rehabexer"],
                               ec3d.BONES, a.seed, device, log=log)
    elif method == "m4":
        import m4_litemv as ad
        model = ad.train_model(train_items, cfg["third_party"]["deeprehabpile"],
                               a.seed, a.run_dir, log=log)
    elif method == "m5":
        import m5_convtran as ad
        model = ad.train_model(train_items, cfg["third_party"]["deeprehabpile"],
                               a.seed, a.run_dir, log=log)

    rows = []
    for cond in conds:
        outs = (ad.score_items(model, cond["items"], device) if method not in NO_DEVICE_METHODS
                else ad.score_items(model, cond["items"]))
        for out in outs:
            s, score, pred = out[:3]
            extra = out[3] if len(out) > 3 else {}
            rows.append(dict(split=cond["name"], k=cond["k"], draw=cond["draw"],
                             id=s["id"], person=s["sub"], exercise=s["act"],
                             label=s["err"], score=score, pred=pred, **extra))
    write_scores(os.path.join(a.run_dir, "scores.csv"), rows)
    poller.stop_flag = True
    poller.join(timeout=3)
    elapsed = time.time() - t0
    json.dump(dict(elapsed_sec=elapsed, gpu_mem_max_mib=poller.max_mib,
                   n_scores=len(rows), method=method, split=a.split,
                   fold=a.fold, seed=a.seed),
              open(os.path.join(a.run_dir, "resource.json"), "w"), indent=2)
    log("wrote {} rows; elapsed {:.1f}s; gpu_mem_max {} MiB".format(
        len(rows), elapsed, poller.max_mib))


if __name__ == "__main__":
    main()
