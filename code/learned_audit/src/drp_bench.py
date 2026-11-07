"""Shared adapter for the DeepRehabPile benchmark estimators (PREREG_R7_amend1).

M4 = LITEMV (Ismail-Fawaz et al. 2023), M5 = ConvTran (Foumani et al. 2024),
implementations taken unmodified from third_party/deeprehabpile
(github.com/MSD-IRIMAS/DeepRehabPile, commit in MANIFEST; package installed
with `pip install -e .` into env `deeprehabpile`). The estimator classes
LITE_MV_CLASSIFIER / ConvTran_CLASSIFIER are used as the benchmark driver
`_main_classification.py` uses them; only the data split and the input view
are ours.

Recipe = benchmark defaults, read from their `config/config_hydra.yaml`
(epochs 1500, batch_size 64, runs 5, estimator_params[<estimator>]) and baked
into their classes: Adam, categorical cross-entropy,
ReduceLROnPlateau(monitor="loss", factor 0.5, patience 50, min_lr 1e-4),
ModelCheckpoint(monitor="loss", save_best_only) -> the checkpoint with the
lowest TRAINING loss is what `predict_proba` loads (no test data involved).
Ensemble: 5 initializations per run unit, prediction-level probability
averaging (their `preds += predict_proba; preds / runs`). Our run seed
controls the 5 initializations: tf.keras.utils.set_random_seed(5*seed + i),
i = 0..4 (the benchmark itself does not seed).

Task granularity: benchmark convention "one exercise = one dataset" -> one
binary model (error = 1, correct = 0, from seg["err"]) per action; the three
actions' scores of a run unit go into the same scores.csv.

Input view: our preprocessed (100, 19, 3) rep (root-centred, resampled by
ec3d.prepare_fold, personal calibration for S2 already applied there)
-> (19, 3, 100) = their (n_joints, dim, length_TS) -> their
`_normalize_skeletons` (per-dimension min-max; statistics from the training
persons' reps of that action only, applied to the held-out person, exactly
as `load_classification_data` fits on x_train and applies to x_test)
-> reshape (57, 100) channels-first; their fit / predict_proba transpose to
(100, 57). Labels go through LabelEncoder as in their loader (0 -> 0,
1 -> 1), so the error class is softmax column 1 (looked up, not assumed).

Score = ensembled p(error). Decision: p(error) > 0.5.
Models are loaded once per (action, initialization) for scoring with the
same tf.keras.models.load_model(..., compile=False) call (and ConvTran
custom_objects) as their predict_proba, then applied to every test
condition; a Keras model in inference mode gives the same output whichever
batch it sits in, so this equals calling their predict_proba per condition.
"""
import contextlib
import os
import time

import numpy as np

ESTIMATOR = {"m4": "LITEMV", "m5": "ConvTran"}
N_JOINTS = 19
N_DIM = 3
LENGTH_TS = 100
ERROR_LABEL = 1  # seg["err"] value of the error class
DECISION_THR = 0.5

_REPO_DIR = None


def init(repo_dir):
    """Remember the repo dir (hydra config lives there; package is pip -e).

    GPU memory: their _main_classification.py sets memory growth. Several
    run units share one V100 here, and XLA's cuDNN autotuning let a LITEMV
    process grow its pool to 12-17 GB and starve its neighbours (OOM at
    start-up), so when DRP_GPU_MEM_LIMIT_MIB is set the process is capped to
    that many MiB instead (TF logical device limit; purely a resource
    setting, the recipe is unchanged). Without the variable: memory growth
    as in the benchmark driver.
    """
    global _REPO_DIR
    _REPO_DIR = repo_dir
    import tensorflow as tf
    limit = os.environ.get("DRP_GPU_MEM_LIMIT_MIB")
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            if limit:
                tf.config.set_logical_device_configuration(
                    gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=int(limit))])
            else:
                tf.config.experimental.set_memory_growth(gpu, True)
        except (RuntimeError, ValueError):
            pass


def hydra_defaults(repo_dir, estimator):
    """Benchmark defaults from config/config_hydra.yaml via OmegaConf (as hydra
    loads them; YAML floats such as 1e-5 are parsed as floats)."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(os.path.join(repo_dir, "config", "config_hydra.yaml"))
    params = OmegaConf.to_container(cfg.estimator_params[estimator], resolve=True)
    return dict(epochs=int(cfg.epochs), batch_size=int(cfg.batch_size),
                runs=int(cfg.runs), estimator_params=params)


def estimator_class(estimator):
    from deep_rehab_pile.classifiers import LITE_MV_CLASSIFIER, ConvTran_CLASSIFIER
    return {"LITEMV": LITE_MV_CLASSIFIER, "ConvTran": ConvTran_CLASSIFIER}[estimator]


def to_benchmark_view(items):
    """(seg, X(100,19,3)) items -> (n, n_joints, dim, length_TS) float64."""
    X = np.stack([x for _, x in items]).astype(np.float64)  # (n,100,19,3)
    return np.ascontiguousarray(X.transpose(0, 2, 3, 1))     # (n,19,3,100)


def normalize_like_loader(items, min_max_list=None):
    """Their _normalize_skeletons then their reshape to (n, n_joints*dim, length_TS)."""
    from deep_rehab_pile.utils._utils import _normalize_skeletons
    X4 = to_benchmark_view(items)
    Xn, mm = _normalize_skeletons(X=X4, min_max_list=min_max_list)
    return np.reshape(Xn, (len(items), N_JOINTS * N_DIM, LENGTH_TS)), mm


def _fit_summary(clf):
    """Best training loss / epoch / final lr from the Keras History if present."""
    out = {}
    try:
        h = clf.model.history.history
        loss = [float(v) for v in h["loss"]]
        out["epochs_run"] = len(loss)
        out["best_train_loss"] = float(np.min(loss))
        out["best_epoch"] = int(np.argmin(loss)) + 1
        out["final_train_loss"] = loss[-1]
        if "learning_rate" in h:
            out["final_lr"] = float(h["learning_rate"][-1])
    except Exception as e:  # pragma: no cover - diagnostics only
        out["history_error"] = repr(e)
    try:
        out["params_total"] = int(clf.model.count_params())
        out["params_non_trainable"] = int(sum(int(np.prod(w.shape)) for w in clf.model.non_trainable_weights))
    except Exception as e:  # pragma: no cover
        out["params_error"] = repr(e)
    return out


def train_model(train_items, repo_dir, method, seed, run_dir, log=print):
    """Train the per-action ensembles. Returns a bundle used by score_items."""
    import tensorflow as tf
    from sklearn.preprocessing import LabelEncoder
    init(repo_dir)
    estimator = ESTIMATOR[method]
    hd = hydra_defaults(repo_dir, estimator)
    cls = estimator_class(estimator)
    log("{} estimator={} hydra defaults: epochs={} batch_size={} runs={} params={}".format(
        method, estimator, hd["epochs"], hd["batch_size"], hd["runs"], hd["estimator_params"]))

    acts = sorted({s["act"] for s, _ in train_items})
    bundle = dict(method=method, estimator=estimator, hydra=hd, seed=seed, per_act={})
    for act in acts:
        items = [(s, x) for s, x in train_items if s["act"] == act]
        Xc, min_max = normalize_like_loader(items, None)
        y_raw = np.array([s["err"] for s, _ in items], dtype=int)
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        classes = [int(c) for c in le.classes_]
        assert len(classes) == 2, "action {} has {} classes in training".format(act, classes)
        err_col = classes.index(ERROR_LABEL)
        log("{} action={} n_train={} (error={} correct={}) min_max_per_dim={}".format(
            method, act, len(items), int((y_raw == 1).sum()), int((y_raw == 0).sum()),
            [(float(a), float(b)) for a, b in min_max]))
        clfs, fit_infos = [], []
        for i in range(hd["runs"]):
            sub_seed = hd["runs"] * seed + i
            out_dir = os.path.join(run_dir, "drp", act, "run{}".format(i)) + "/"
            os.makedirs(out_dir, exist_ok=True)
            tf.keras.utils.set_random_seed(sub_seed)
            clf = cls(output_dir=out_dir, best_file_name="best_model",
                      init_file_name="init_model", length_TS=LENGTH_TS,
                      n_joints=N_JOINTS, n_dim=N_DIM, epochs=hd["epochs"],
                      batch_size=hd["batch_size"], **hd["estimator_params"])
            t0 = time.time()
            # their fit prints model.summary() and a verbose=1 progress bar per
            # epoch; keep that in the model's own folder, not in log.txt
            with open(os.path.join(out_dir, "fit_stdout.txt"), "w") as fh, \
                    contextlib.redirect_stdout(fh):
                clf.fit(X=Xc, y=y)
            info = dict(init=i, sub_seed=sub_seed, fit_sec=time.time() - t0, **_fit_summary(clf))
            fit_infos.append(info)
            log("{} {} init {}/{} seed {}: {}".format(method, act, i + 1, hd["runs"], sub_seed,
                                                     {k: (round(v, 6) if isinstance(v, float) else v)
                                                      for k, v in info.items() if k not in ("init", "sub_seed")}))
            clfs.append(clf)
        bundle["per_act"][act] = dict(min_max=min_max, clfs=clfs, classes=classes,
                                      err_col=err_col, fit_infos=fit_infos, models=None)
    return bundle


def _load_best_models(pa, estimator):
    """Load each initialization's lowest-training-loss checkpoint once, with
    the same load_model call as their predict_proba."""
    import tensorflow as tf
    if pa["models"] is not None:
        return pa["models"]
    custom_objects = None
    if estimator == "ConvTran":
        from deep_rehab_pile.classifiers._convtran import tAPE, Attention_Rel_Scl, MySqueezeLayer
        custom_objects = {"tAPE": tAPE, "Attention_Rel_Scl": Attention_Rel_Scl,
                          "MySqueezeLayer": MySqueezeLayer}
    models = []
    for clf in pa["clfs"]:
        path = clf.output_dir + clf.best_file_name + ".keras"
        if custom_objects is None:
            models.append(tf.keras.models.load_model(path, compile=False))
        else:
            models.append(tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False))
    pa["models"] = models
    return models


def ensemble_proba(bundle, act, Xc):
    """Their ensemble: sum of predict_proba over the initializations / runs.
    predict_proba = load best_model.keras, swap to (n, T, C), model.predict."""
    models = _load_best_models(bundle["per_act"][act], bundle["estimator"])
    X = np.swapaxes(Xc, 1, 2)
    preds = np.zeros((len(Xc), len(bundle["per_act"][act]["classes"])), dtype=np.float64)
    for m in models:
        preds = preds + np.asarray(m.predict(X, verbose=0), dtype=np.float64)
    return preds / (1.0 * len(models))


def score_items(bundle, items):
    """-> list of (seg, score, pred) in the order of items."""
    if not items:
        return []
    out = [None] * len(items)
    for act in sorted({s["act"] for s, _ in items}):
        idx = [j for j, (s, _) in enumerate(items) if s["act"] == act]
        sub = [items[j] for j in idx]
        pa = bundle["per_act"][act]
        Xc, _ = normalize_like_loader(sub, pa["min_max"])
        proba = ensemble_proba(bundle, act, Xc)
        p_err = proba[:, pa["err_col"]]
        for j, (s, _), p in zip(idx, sub, p_err):
            out[j] = (s, float(p), int(p > DECISION_THR))
    return out
