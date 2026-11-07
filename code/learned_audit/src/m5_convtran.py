"""M5: ConvTran (Foumani et al. 2024) as shipped in DeepRehabPile.

Thin wrapper over drp_bench (shared DeepRehabPile recipe, see its docstring):
estimator "ConvTran", hydra defaults kernel_size 8, factor_filters 4,
activation gelu, dropout_rate 0.01, emb_size 16, num_heads 8, epsilon 1e-5,
dimm_ff 256, activation_ff relu. 1500 epochs, batch 64, lowest-training-loss
checkpoint, 5-initialization probability-averaging ensemble, one binary model
per action. Score = ensembled p(error); pred = p(error) > 0.5.
"""
import drp_bench

METHOD = "m5"
ESTIMATOR = drp_bench.ESTIMATOR[METHOD]


def train_model(train_items, repo_dir, seed, run_dir, log=print):
    return drp_bench.train_model(train_items, repo_dir, METHOD, seed, run_dir, log=log)


def score_items(bundle, items):
    return drp_bench.score_items(bundle, items)
