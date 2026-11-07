"""M4: LITEMV (Ismail-Fawaz et al. 2023) as shipped in DeepRehabPile.

Thin wrapper over drp_bench (shared DeepRehabPile recipe, see its docstring):
estimator "LITEMV", hydra defaults n_filters 32, kernel_size 40, strides 1,
activation relu; hand-crafted hybrid filters are non-trainable (requires
keras >= 3.6, checked by the env). 1500 epochs, batch 64, lowest-training-loss
checkpoint, 5-initialization probability-averaging ensemble, one binary model
per action. Score = ensembled p(error); pred = p(error) > 0.5.
"""
import drp_bench

METHOD = "m4"
ESTIMATOR = drp_bench.ESTIMATOR[METHOD]


def train_model(train_items, repo_dir, seed, run_dir, log=print):
    return drp_bench.train_model(train_items, repo_dir, METHOD, seed, run_dir, log=log)


def score_items(bundle, items):
    return drp_bench.score_items(bundle, items)
