"""Smoke test for the amend1 adapters (m4 LITEMV, m5 ConvTran).

Monkeypatches drp_bench.hydra_defaults -> epochs=3, runs=1, then drives each
method through the same run_split path (prepare_fold -> train_model ->
score_items) on S1 fold 0 seed 0, output to
exp/prl_v2/round7/smoke_amend1/<method>/. Checks:
  * training runs without error for all 3 actions
  * best_model.keras reloads via _load_best_models (ConvTran custom_objects)
  * scores.csv row counts match the S1 condition items
  * p(error) in [0,1], pred == (p > 0.5), err_col maps to label 1
"""
import csv, json, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import ec3d  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = "/data1/users/yangrui/results/idea67b/exp/prl_v2/round7/smoke_amend1"
CFG = dict(
    data_pickle="/data1/users/yangrui/projects/idea67b/experiment/code/zhao22/data/EC3D/data_3D.pickle",
    repo="/data1/users/yangrui/projects/idea67b/third_party/deeprehabpile")


def log(m):
    print("[{}] {}".format(time.strftime("%H:%M:%S"), m), flush=True)


def main(method):
    if method == "m4":
        import m4_litemv as ad
    else:
        import m5_convtran as ad
    import drp_bench

    real = drp_bench.hydra_defaults

    def patched(repo_dir, estimator):
        d = real(repo_dir, estimator)
        d["epochs"] = 3
        d["runs"] = 1
        return d

    drp_bench.hydra_defaults = patched

    run_dir = os.path.join(OUT_ROOT, method)
    os.makedirs(run_dir, exist_ok=True)
    segs = ec3d.load_segments(CFG["data_pickle"])
    train_items, conds = ec3d.prepare_fold(segs, "s1", 0, None)
    log("{} train={} conds={}".format(method, len(train_items), len(conds)))

    t0 = time.time()
    bundle = ad.train_model(train_items, CFG["repo"], 0, run_dir, log=log)
    fit_secs = {a: [i["fit_sec"] for i in pa["fit_infos"]] for a, pa in bundle["per_act"].items()}

    rows = []
    for cond in conds:
        for s, score, pred in ad.score_items(bundle, cond["items"]):
            assert 0.0 <= score <= 1.0, "score out of range: {}".format(score)
            assert pred == int(score > 0.5)
            rows.append(dict(split=cond["name"], id=s["id"], person=s["sub"],
                             exercise=s["act"], label=s["err"],
                             score=score, pred=pred))

    # err_col really points at label 1 (error)
    for a, pa in bundle["per_act"].items():
        assert pa["classes"][pa["err_col"]] == 1, (a, pa["classes"], pa["err_col"])

    # checkpoint reload path actually exercised inside ensemble_proba already;
    # verify files exist
    ckpts = []
    for a, pa in bundle["per_act"].items():
        for clf in pa["clfs"]:
            p = clf.output_dir + clf.best_file_name + ".keras"
            ckpts.append(p)
            assert os.path.exists(p), p

    with open(os.path.join(run_dir, "scores.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "id", "person", "exercise", "label", "score", "pred"])
        for r in rows:
            w.writerow([r["split"], r["id"], r["person"], r["exercise"],
                        r["label"], "{:.10g}".format(r["score"]), r["pred"]])

    report = dict(method=method, n_train=len(train_items), n_scores=len(rows),
                  expected_scores=len(conds[0]["items"]),
                  fit_secs=fit_secs, elapsed_sec=time.time() - t0,
                  checkpoints=ckpts,
                  label_counts={str(k): sum(1 for r in rows if r["label"] == k) for k in (0, 1)},
                  score_range=[min(r["score"] for r in rows), max(r["score"] for r in rows)])
    assert report["n_scores"] == report["expected_scores"]
    json.dump(report, open(os.path.join(run_dir, "smoke.json"), "w"), indent=2)
    log("{} smoke OK: {}".format(method, json.dumps({k: report[k] for k in
        ("n_train", "n_scores", "fit_secs", "score_range")})))


if __name__ == "__main__":
    main(sys.argv[1])
