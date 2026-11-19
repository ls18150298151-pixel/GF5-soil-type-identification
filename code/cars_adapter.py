import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from feature_algorithms import *

def check_cars_validation_targets(y,n):
    """STEP07-A01: only the internal validation positive-class check is relaxed."""
    y=np.asarray(y,dtype=np.float64)
    require(n>0 and y.shape==(n,6),'CARS internal validation must be nonempty with six responses')
    require(np.isfinite(y).all() and np.isin(y,[0.,1.]).all(),'Invalid CARS validation one-hot values')
    require((y.sum(axis=1)==1).all(),'CARS validation rows must each contain exactly one 1')
    return y

def step07_cars_internal_splits(context, Y, seed):
    Y = check_targets(Y, len(context.training_ids))
    real = np.asarray(context.is_real, dtype=bool)
    groups = np.asarray(context.groups)
    leakage = np.asarray(context.leakage_groups)
    real_idx = np.flatnonzero(real)
    labels = np.asarray(CLASSES)[Y[real].argmax(axis=1)]
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=int(seed))
    result = []
    for _, v_local in splitter.split(np.zeros((len(real_idx), 1)), labels, groups[real]):
        val = real_idx[v_local]
        held_groups = np.unique(groups[val])
        train = np.flatnonzero(~np.isin(groups, held_groups))
        excluded = np.flatnonzero(~real & np.isin(groups, held_groups))
        check_cars_validation_targets(Y[val], len(val))
        check_targets(Y[train], len(train))
        require(real[val].all(), "Virtual CARS internal validation")
        require(not set(groups[train]) & set(groups[val]), "CARS internal group leakage")
        require(not set(leakage[train]) & set(leakage[val]), "CARS duplicate-coordinate leakage")
        require(len(train) + len(val) + len(excluded) == len(real), "CARS internal role conservation failed")
        result.append((train, val, excluded))
    require(len(result) == 3, "CARS requires three internal folds")
    return result

class Step07CARSSelector(CARSSelector):
    """Explicit stage07 adapter, not a global monkeypatch.

    The _fit body is identical to the protected stage06 implementation except
    the internal-split function name. An AST-equivalence test enforces this.
    """
    def _fit(self, X, Y):
        rng = np.random.default_rng(self.seed)
        self.partitions_ = step07_cars_internal_splits(self.context_, Y, self.seed)
        groups = np.asarray(self.context_.groups)
        unique_groups = np.unique(groups)
        sample_groups_n = int(np.ceil(self.params["calibration_fraction"] * len(unique_groups)))
        require(1 < sample_groups_n <= len(unique_groups), "Invalid calibration group count")
        counts = edf_counts(X.shape[1], self.params["n_iterations"])
        pool = np.arange(X.shape[1])
        self.history_ = []
        all_importance = []
        for iteration, count in enumerate(counts, 1):
            chosen_groups = np.sort(rng.choice(unique_groups, size=sample_groups_n, replace=False))
            calibration = np.flatnonzero(np.isin(groups, chosen_groups))
            check_targets(Y[calibration], len(calibration))
            scaler, pls = fit_pls(X[np.ix_(calibration, pool)], Y[calibration], self.params["max_components"])
            importance = np.linalg.norm(pls.coef_, axis=0)
            selected = ars_without_replacement(pool, importance, int(count), rng)
            score, fold_scores = subset_rmsecv(X, Y, selected, self.partitions_, self.params["max_components"])
            full_importance = np.zeros(X.shape[1])
            full_importance[pool] = importance
            all_importance.append(full_importance)
            self.history_.append({"iteration": iteration, "edf_target": int(count), "pool_indices": pool.tolist(),
                "selected_indices": selected.tolist(), "calibration_ids": [int(self.context_.training_ids[i]) for i in calibration],
                "calibration_groups": chosen_groups.tolist(), "calibration_fraction_groups": sample_groups_n / len(unique_groups),
                "calibration_n_real": int(np.asarray(self.context_.is_real)[calibration].sum()),
                "calibration_n_virtual": int((~np.asarray(self.context_.is_real)[calibration]).sum()),
                "calibration_scaler_hash": object_hash({"mean": scaler.mean_, "scale": scaler.scale_}),
                "importance_L2": importance.tolist(), "coefficient_shape": list(pls.coef_.shape),
                "pls_components": int(pls.n_components), "solver": "SVD_PLS2_regression_deflation", "rmsecv_mean": score, "rmsecv_folds": fold_scores,
                "ars_replace": False})
            pool = selected
        winner = min(range(len(self.history_)), key=lambda i: (self.history_[i]["rmsecv_mean"],
                     len(self.history_[i]["selected_indices"]), tuple(self.history_[i]["selected_indices"])))
        self.selected_indices_ = np.asarray(self.history_[winner]["selected_indices"], dtype=int)
        self.importance_ = all_importance[winner]
        self.winner_iteration_ = winner + 1
        self.best_rmsecv_ = self.history_[winner]["rmsecv_mean"]
        self.feature_names_out_ = tuple(self.feature_names_in_[i] for i in self.selected_indices_)
