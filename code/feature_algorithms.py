"""Leakage-guarded, fold-local feature engineering; no classifiers.

Every public fit accepts only manifest-authorized training rows. Transform is
label-free and rejects sealed/excluded IDs. Selection transforms return raw
selected spectral columns; PCA returns scores from its training-fitted scaler.
PLS fits below are multivariate one-hot regressions for variable selection only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

CLASSES = ("红壤", "紫色土", "燥红土", "黄棕壤", "石灰岩土", "黄壤")
TRAIN_ROLES = ("train_real", "train_virtual_only")
BASE_SEED = 20260915
ENV_NAMES = (
    "DEM_Elevation_m", "Terrain_Available", "Terrain_Slope_deg", "Terrain_Aspect_sin",
    "Terrain_Aspect_cos", "Terrain_Shaded_Relief", "Terrain_Profile_Convexity",
    "Terrain_Plan_Convexity", "Terrain_Longitudinal_Convexity", "Terrain_Cross_Sectional_Convexity",
    "Terrain_Minimum_Curvature", "Terrain_Maximum_Curvature", "Terrain_RMS_Error",
    "Eco_IOI", "Eco_BSI", "Eco_NDWI", "Eco_NDVI")


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value


def canonical(value):
    return (json.dumps(native(value), ensure_ascii=False, sort_keys=True,
                       indent=2, allow_nan=False) + "\n").encode("utf-8")


def object_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(canonical({"shape": a.shape, "dtype": a.dtype.str}))
    h.update(a.tobytes())
    return h.hexdigest()


def one_hot(labels):
    labels = np.asarray(labels)
    require(set(labels) <= set(CLASSES), "Unrecognized soil class")
    return np.column_stack([labels == c for c in CLASSES]).astype(np.float64)


def check_targets(y, n):
    y = np.asarray(y, dtype=np.float64)
    require(y.shape == (n, 6), "Supervised selection requires exactly six one-hot responses")
    require(np.isfinite(y).all() and np.isin(y, [0., 1.]).all(), "Invalid one-hot target values")
    require((y.sum(axis=1) == 1).all(), "Targets must be mutually exclusive one-hot rows")
    require((y.sum(axis=0) > 0).all(), "Training or internal validation fold lacks a class")
    return y


@dataclass(frozen=True)
class FoldContext:
    """Only IDs/roles/groups/source flags, never validation X or labels."""
    name: str
    training_ids: tuple
    validation_ids: tuple
    groups: tuple
    leakage_groups: tuple
    is_real: tuple
    group_column: str

    @classmethod
    def from_manifest(cls, manifest, name=None):
        m = manifest.copy()
        require(m.sample_id.is_unique, "Choose exactly one manifest fold")
        for key in ("protocol", "outer_fold", "inner_fold"):
            if key in m:
                require(m[key].nunique() == 1, "Mixed fold boundaries")
        require(np.array_equal(m.is_training, m.role.isin(TRAIN_ROLES).astype(int)), "Training role flag mismatch")
        require(np.array_equal(m.is_validation, (m.role == "validation_real").astype(int)), "Validation role flag mismatch")
        train = m[m.is_training == 1]
        val = m[m.is_validation == 1]
        require(len(train) > 0 and len(val) > 0, "Empty fold side")
        require((val.is_real == 1).all(), "Virtual validation is prohibited")
        require((train.is_real + train.is_virtual == 1).all(), "Invalid sample source flags")
        require((train.loc[train.role == "train_real", "is_real"] == 1).all(), "Training real role mismatch")
        require((train.loc[train.role == "train_virtual_only", "is_virtual"] == 1).all(), "Training virtual role mismatch")
        protocol = str(m.protocol.iloc[0])
        group_column = "spatial_block_id" if "spatial" in protocol else "leakage_group_id"
        for group in (group_column, "leakage_group_id"):
            require(not set(train[group]) & set(val[group]), "Group leakage across train/validation")
        if "inner_data_eligible" in m:
            require(np.array_equal(m.inner_data_eligible, m.is_training + m.is_validation), "Sealed inner eligibility mismatch")
        parts = [protocol, f"outer{int(m.outer_fold.iloc[0])}"]
        if "inner_fold" in m:
            parts.append(f"inner{int(m.inner_fold.iloc[0])}")
        return cls(name or "/".join(parts), tuple(train.sample_id.astype(int)), tuple(val.sample_id.astype(int)),
                   tuple(train[group_column].astype(str)), tuple(train.leakage_group_id.astype(str)),
                   tuple(train.is_real.astype(bool)), group_column)

    @property
    def training_id_hash(self):
        return object_hash(self.training_ids)


class TrainingOnlyLabels:
    """Runtime label-access guard: a validation-label request raises immediately."""
    def __init__(self, labels, allowed_ids):
        self._labels = labels
        self.allowed_ids = set(allowed_ids)
        self.read_batches = []

    @property
    def loc(self):
        return self

    def __getitem__(self, ids):
        ids = list(ids)
        require(set(ids) <= self.allowed_ids, "Forbidden validation/sealed/excluded label access")
        self.read_batches.append(tuple(ids))
        return self._labels.loc[ids]


class FoldTransformer:
    supervised = False
    allow_nan = False

    def __init__(self, seed=BASE_SEED, **params):
        self.seed = int(seed)
        self.params = params
        self.fit_calls_ = 0
        self.fitted_ = False

    def fit(self, X, y=None, *, context):
        require(isinstance(context, FoldContext), "A manifest-derived FoldContext is mandatory")
        require(isinstance(X, pd.DataFrame), "X must be a DataFrame indexed by sample_id")
        require(tuple(X.index) == context.training_ids, "fit input must equal the authorized ordered training IDs")
        require(X.columns.is_unique and X.index.is_unique, "Duplicate feature or sample IDs")
        # Identical values must follow identical reduction/BLAS paths even if
        # pandas changed its memory layout while unrelated validation rows changed.
        a = np.array(X.to_numpy(), dtype=np.float64, order="C", copy=True)
        require(a.ndim == 2 and min(a.shape) > 0 and not np.isinf(a).any(), "Invalid fitting array")
        if not self.allow_nan:
            require(np.isfinite(a).all(), "Spectral fitting values must be finite")
        if self.supervised:
            y = check_targets(y, len(X))
        else:
            require(y is None, "Unsupervised transformer must not receive labels")
        self.context_ = context
        self.feature_names_in_ = tuple(map(str, X.columns))
        self.n_features_in_ = a.shape[1]
        self.fit_input_hash_ = array_hash(a)
        self.fit_target_hash_ = array_hash(y) if y is not None else "not_used"
        self.fitted_ = False
        with threadpool_limits(limits=1):
            self._fit(a, y)
        self.fit_calls_ += 1
        self.fitted_ = True
        require(len(self.feature_names_out_) > 0, "Empty output feature set")
        return self

    def transform(self, X):
        require(self.fitted_, "Transformer is not fitted")
        require(isinstance(X, pd.DataFrame), "Transform X must be indexed by sample_id")
        require(tuple(map(str, X.columns)) == self.feature_names_in_, "Feature names/order changed")
        require(X.index.is_unique, "Duplicate transform IDs")
        allowed = set(self.context_.training_ids) | set(self.context_.validation_ids)
        require(set(X.index) <= allowed, "Sealed/excluded/unknown IDs are forbidden in transform")
        a = np.array(X.to_numpy(), dtype=np.float64, order="C", copy=True)
        require(not np.isinf(a).any(), "Infinite transform input")
        if not self.allow_nan:
            require(np.isfinite(a).all(), "Nonfinite spectral transform input")
        before, count = self.state_hash(), self.fit_calls_
        with threadpool_limits(limits=1):
            result = self._transform(a)
        require(self.fit_calls_ == count and self.state_hash() == before, "Transform mutated/refitted fitting state")
        require(result.shape == (len(X), len(self.feature_names_out_)), "Unexpected transform shape")
        require(np.isfinite(result).all(), "Nonfinite transformed values")
        return pd.DataFrame(result, index=X.index.copy(), columns=self.feature_names_out_)

    def get_feature_names_out(self):
        require(self.fitted_, "Transformer is not fitted")
        return np.asarray(self.feature_names_out_, dtype=str)

    def state_summary(self):
        require(self.fitted_, "Transformer is not fitted")
        return native({"transformer": type(self).__name__, "seed": self.seed, "params": self.params,
                       "boundary": self.context_.name, "training_ids": self.context_.training_ids,
                       "fit_id_hash": self.context_.training_id_hash, "fit_input_hash": self.fit_input_hash_,
                       "fit_target_hash": self.fit_target_hash_, "feature_names_in": self.feature_names_in_,
                       "feature_names_out": self.feature_names_out_, "state": self._state()})

    def state_hash(self):
        return object_hash(self.state_summary())


class EnvironmentPreprocessor(FoldTransformer):
    allow_nan = True

    def _fit(self, X, y):
        require(self.feature_names_in_ == ENV_NAMES, "Environment feature order differs from stage04")
        require(np.isin(X[:, 1], [0., 1.]).all(), "Terrain_Available must remain binary")
        require(not np.isnan(X).all(axis=0).any(), "An environment column is entirely missing in training")
        self.medians_ = np.nanmedian(X, axis=0)
        self.missing_counts_ = np.isnan(X).sum(axis=0)
        filled = np.where(np.isnan(X), self.medians_, X)
        self.numeric_ = np.array([i for i in range(17) if i != 1])
        self.scaler_ = StandardScaler().fit(filled[:, self.numeric_])
        self.feature_names_out_ = self.feature_names_in_

    def _transform(self, X):
        require(np.isin(X[:, 1], [0., 1.]).all(), "Terrain_Available must remain binary")
        return np.where(np.isnan(X), self.medians_, X)

    def transform_standardized(self, X):
        before = self.state_hash()
        result = self.transform(X)
        result.iloc[:, self.numeric_] = self.scaler_.transform(result.iloc[:, self.numeric_].to_numpy())
        require(np.isfinite(result.to_numpy()).all() and self.state_hash() == before, "Environment scaler state mutation")
        return result

    def _state(self):
        return {"medians": self.medians_, "missing_counts": self.missing_counts_, "numeric_indices": self.numeric_,
                "scaler_mean": self.scaler_.mean_, "scaler_scale": self.scaler_.scale_,
                "terrain_available": "unchanged binary; not standardized; no duplicate indicator"}


class FoldPCA(FoldTransformer):
    def __init__(self, variance_threshold=.99, seed=BASE_SEED):
        require(0 < variance_threshold < 1, "PCA threshold must be between zero and one")
        super().__init__(seed, variance_threshold=float(variance_threshold), svd_solver="full")

    def _fit(self, X, y):
        self.scaler_ = StandardScaler().fit(X)
        Z = self.scaler_.transform(X)
        # Full deterministic spectrum provides the 95/99% diagnostic curve;
        # only the predeclared reference threshold determines transformed PCs.
        self.pca_ = PCA(svd_solver="full", random_state=self.seed).fit(Z)
        self.cumulative_ = np.cumsum(self.pca_.explained_variance_ratio_)
        require(np.isfinite(self.cumulative_).all(), "Invalid PCA variance spectrum")
        self.n_components_ = int(np.searchsorted(self.cumulative_, self.params["variance_threshold"], side="left") + 1)
        self.components_ = self.pca_.components_[:self.n_components_].copy()
        block = self.feature_names_in_[0].split("_")[0]
        self.feature_names_out_ = tuple(f"{block}_PC{i+1:03d}" for i in range(self.n_components_))

    def _transform(self, X):
        return (self.scaler_.transform(X) - self.pca_.mean_) @ self.components_.T

    def _state(self):
        return {"scaler_mean": self.scaler_.mean_, "scaler_scale": self.scaler_.scale_,
                "pca_center": self.pca_.mean_, "components": self.components_,
                "explained_variance": self.pca_.explained_variance_,
                "explained_variance_ratio": self.pca_.explained_variance_ratio_,
                "n_components": self.n_components_, "type": "projection_extraction_not_band_selection"}


class SVDPLSRegression(PLSRegression):
    """PLS2 regression deflation with exact SVD of each X.T @ Y matrix.

    scikit-learn 1.7.2's shared PLS fit supports both SVD and NIPALS. Its
    PLSRegression constructor fixes NIPALS; this explicit adapter selects the
    existing SVD path, keeping regression deflation and all scaling unchanged.
    This avoids power-iteration nonconvergence; it is not a warning fallback.
    """
    def __init__(self, n_components=2, *, scale=False, copy=True):
        super().__init__(n_components=n_components, scale=scale, copy=copy)
        self.algorithm = "svd"


def fit_pls(X, Y, max_components=10):
    check_targets(Y, len(X))
    require(np.isfinite(X).all(), "Nonfinite PLS input")
    scaler = StandardScaler().fit(X)
    Z = scaler.transform(X)
    variable_columns = int(np.sum(scaler.var_ > 0))
    n = min(int(max_components), len(X) - 1, X.shape[1], variable_columns)
    require(n >= 1, "No usable PLS components")
    model = SVDPLSRegression(n_components=n, scale=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(Z, Y)
    require(not caught, "PLS fit warning: " + "; ".join(str(w.message) for w in caught))
    require(model.algorithm == "svd" and model.deflation_mode == "regression", "PLS solver configuration changed")
    require(model.coef_.shape == (6, X.shape[1]) and np.isfinite(model.coef_).all(), "Invalid multiresponse PLS coefficients")
    return scaler, model


def multiresponse_vip(scores, weights, y_loadings):
    """VIP_j^2 = p sum_a[(w_ja^2/||w_a||^2) ||t_a||^2 ||q_a||^2] / sum_a SSY_a."""
    T, W, Q = map(lambda x: np.asarray(x, dtype=float), (scores, weights, y_loadings))
    require(T.shape[1] == W.shape[1] == Q.shape[1] and Q.shape[0] == 6, "VIP requires six response loadings")
    require(all(np.isfinite(a).all() for a in (T, W, Q)), "Nonfinite VIP arrays")
    ssy = np.sum(T ** 2, axis=0) * np.sum(Q ** 2, axis=0)
    norm2 = np.sum(W ** 2, axis=0)
    require((norm2 > 0).all() and ssy.sum() > 0, "Degenerate VIP denominator")
    return np.sqrt(W.shape[0] * ((W ** 2 / norm2) @ ssy) / ssy.sum())


def vip_indices(vip, threshold=1., minimum=5):
    selected = np.flatnonzero(np.asarray(vip) > threshold)
    fallback = len(selected) < minimum
    if fallback:
        require(len(vip) >= minimum, "Fewer than five available spectral variables")
        selected = np.sort(np.lexsort((np.arange(len(vip)), -np.asarray(vip)))[:minimum])
    return selected, fallback


class PLSVIPSelector(FoldTransformer):
    supervised = True

    def __init__(self, vip_threshold=1., max_components=10, seed=BASE_SEED):
        super().__init__(seed, vip_threshold=float(vip_threshold), max_components=int(max_components), minimum_features=5)

    def _fit(self, X, Y):
        self.scaler_, self.pls_ = fit_pls(X, Y, self.params["max_components"])
        self.vip_ = multiresponse_vip(self.pls_.x_scores_, self.pls_.x_weights_, self.pls_.y_loadings_)
        self.selected_indices_, self.fallback_ = vip_indices(self.vip_, self.params["vip_threshold"], 5)
        self.feature_names_out_ = tuple(self.feature_names_in_[i] for i in self.selected_indices_)

    def _transform(self, X):
        return X[:, self.selected_indices_].copy()

    def _state(self):
        return {"scaler_mean": self.scaler_.mean_, "scaler_scale": self.scaler_.scale_,
                "vip": self.vip_, "selected_indices": self.selected_indices_, "fallback_top5": self.fallback_,
                "n_components": self.pls_.n_components, "solver": "SVD_PLS2_regression_deflation", "coef_six_responses": self.pls_.coef_,
                "x_weights": self.pls_.x_weights_, "y_loadings": self.pls_.y_loadings_,
                "x_scores_hash": array_hash(self.pls_.x_scores_), "output_scale": "raw_selected_columns"}


def spa_order(X, length):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    centered = X - mean
    norms = np.linalg.norm(centered, axis=0)
    require(length <= X.shape[1] and length >= 1, "Invalid SPA requested dimension")
    require(np.isfinite(X).all(), "Nonfinite SPA input")
    residual = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)
    selected, residual_sizes = [], []
    available = np.ones(X.shape[1], dtype=bool)
    for _ in range(length):
        size2 = np.sum(residual ** 2, axis=0)
        size2[~available] = -np.inf
        best = np.max(size2)
        require(np.isfinite(best) and best > 1e-24, "SPA exhausted independent columns")
        # Unit-norm columns are tied up to roundoff. Index-order breaks ties.
        candidates = np.flatnonzero(available & np.isclose(size2, best, rtol=1e-12, atol=1e-14))
        j = int(candidates[0])
        selected.append(j)
        residual_sizes.append(float(size2[j]))
        q = residual[:, j] / np.linalg.norm(residual[:, j])
        for _pass in range(2):
            residual -= q[:, None] * (q @ residual)[None, :]
        available[j] = False
    return np.asarray(selected, dtype=int), mean, norms, np.asarray(residual_sizes)


class SPASelector(FoldTransformer):
    def __init__(self, n_features=30, seed=BASE_SEED):
        super().__init__(seed, n_features=int(n_features), tie_rtol=1e-12, tie_atol=1e-14)

    def _fit(self, X, y):
        self.selected_indices_, self.mean_, self.column_norms_, self.residual_sizes_ = spa_order(X, self.params["n_features"])
        self.feature_names_out_ = tuple(self.feature_names_in_[i] for i in self.selected_indices_)

    def _transform(self, X):
        return X[:, self.selected_indices_].copy()

    def _state(self):
        return {"center": self.mean_, "column_norms": self.column_norms_, "selected_indices": self.selected_indices_,
                "selected_residual_norm2": self.residual_sizes_, "labels": "never_received",
                "start_rule": "maximum unit-norm residual; ties by original index", "output_scale": "raw_selected_columns"}


def edf_counts(p, n_iterations=50):
    require(p >= 2 and n_iterations >= 2, "CARS requires at least two variables and iterations")
    ratio = np.exp(np.log(2. / p) * np.arange(n_iterations) / (n_iterations - 1))
    counts = np.clip(np.rint(p * ratio).astype(int), 2, p)
    counts[0], counts[-1] = p, 2
    return counts


def ars_without_replacement(pool, importance, count, rng):
    pool = np.asarray(pool, dtype=int)
    importance = np.asarray(importance, dtype=float)
    require(len(pool) == len(set(pool)) and 1 <= count <= len(pool), "Invalid ARS pool/count")
    require(importance.shape == pool.shape and np.isfinite(importance).all() and (importance >= 0).all(), "Invalid ARS importance")
    require(importance.sum() > 0, "Zero CARS importance")
    if count == len(pool):
        return np.sort(pool.copy())
    # A predeclared epsilon floor ensures zero-coefficient columns remain
    # sampleable if the EDF target exceeds the number of positive weights.
    weights = np.maximum(importance, np.finfo(float).eps * importance.max())
    selected = rng.choice(pool, size=count, replace=False, p=weights / weights.sum())
    require(len(set(selected)) == count, "ARS produced repeated indices")
    return np.sort(selected)


def cars_internal_splits(context, Y, seed):
    Y = check_targets(Y, len(context.training_ids))
    real = np.asarray(context.is_real, dtype=bool)
    groups = np.asarray(context.groups)
    leakage = np.asarray(context.leakage_groups)
    real_idx = np.flatnonzero(real)
    labels = np.asarray(CLASSES)[Y[real].argmax(axis=1)]  # for stratification only
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=int(seed))
    result = []
    for _, v_local in splitter.split(np.zeros((len(real_idx), 1)), labels, groups[real]):
        val = real_idx[v_local]
        held_groups = np.unique(groups[val])
        train = np.flatnonzero(~np.isin(groups, held_groups))
        excluded = np.flatnonzero(~real & np.isin(groups, held_groups))
        check_targets(Y[val], len(val))
        check_targets(Y[train], len(train))
        require(real[val].all(), "Virtual CARS internal validation")
        require(not set(groups[train]) & set(groups[val]), "CARS internal group leakage")
        require(not set(leakage[train]) & set(leakage[val]), "CARS duplicate-coordinate leakage")
        require(len(train) + len(val) + len(excluded) == len(real), "CARS internal role conservation failed")
        result.append((train, val, excluded))
    require(len(result) == 3, "CARS requires three internal folds")
    return result


def subset_rmsecv(X, Y, selected, partitions, max_components=10):
    """Only temporary continuous six-response estimates; no class predictions."""
    values = []
    for train, val, _ in partitions:
        scaler, pls = fit_pls(X[np.ix_(train, selected)], Y[train], max_components)
        estimate = pls.predict(scaler.transform(X[np.ix_(val, selected)]))
        require(estimate.shape == Y[val].shape and np.isfinite(estimate).all(), "Invalid one-hot regression CV estimates")
        values.append(float(np.sqrt(np.mean((estimate - Y[val]) ** 2))))
    require(np.isfinite(values).all(), "Nonfinite CARS RMSECV")
    return float(np.mean(values)), values


class CARSSelector(FoldTransformer):
    supervised = True

    def __init__(self, seed=BASE_SEED, n_iterations=50, calibration_fraction=.8, max_components=10):
        super().__init__(seed, n_iterations=int(n_iterations), calibration_fraction=float(calibration_fraction),
                         max_components=int(max_components), internal_folds=3, response_importance="coefficient_L2_across_6_responses")

    def _fit(self, X, Y):
        rng = np.random.default_rng(self.seed)
        self.partitions_ = cars_internal_splits(self.context_, Y, self.seed)
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

    def _transform(self, X):
        return X[:, self.selected_indices_].copy()

    def _state(self):
        return {"selected_indices": self.selected_indices_, "winner_iteration": self.winner_iteration_,
                "best_onehot_rmsecv": self.best_rmsecv_, "importance_at_winner": self.importance_,
                "history_hash": object_hash(self.history_),
                "internal_partition_hash": object_hash([[t.tolist(), v.tolist(), e.tolist()] for t, v, e in self.partitions_]),
                "output_scale": "raw_selected_columns", "rmsecv_usage": "training_only_selection_criterion_not_classification_accuracy"}


def make_transformer(method, seed=BASE_SEED):
    constructors = {"PCA": FoldPCA, "PLS-VIP": PLSVIPSelector, "SPA": SPASelector, "CARS": CARSSelector}
    require(method in constructors, "Unknown feature method")
    return constructors[method](seed=seed)


def fit_from_manifest(method, features, labels, manifest, seed=BASE_SEED):
    """Safe integration entry point; slicing precedes fit and one-hot encoding.

    features/labels may be the original indexed tables. Only authorized training
    rows reach fit; neither validation labels nor sealed/excluded rows are read.
    Returned transformer is local to this boundary and must be refit for another.
    """
    context = FoldContext.from_manifest(manifest)
    model = EnvironmentPreprocessor(seed=seed) if method == "Environment" else make_transformer(method, seed)
    X = features.loc[list(context.training_ids)]
    Y = one_hot(labels.loc[list(context.training_ids)].to_numpy()) if model.supervised else None
    model.fit(X, Y, context=context)
    return model
