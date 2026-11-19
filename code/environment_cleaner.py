import numpy as np
import pandas as pd
from feature_algorithms import require,native,object_hash
LC_CLASS="LC_class"
ALL_ENV_NUMERIC=[]
MISSING_DROP_RATIO=0.5
NVZ_MIN_UNIQUE=1
NVZ_MODE_RATIO=0.95
REDUNDANCY_THRESHOLD=0.95
def ids_hash(ids):return object_hash([str(x) for x in ids])

def _cat_key(v):
    """Canonical LC_class category key (NaN -> None)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    f = float(v)
    return str(int(f)) if f.is_integer() else repr(f)

def _nan_identical(a, b):
    am, bm = np.isnan(a), np.isnan(b)
    if not np.array_equal(am, bm):
        return False
    both = ~am
    return np.array_equal(a[both], b[both])

class Layer1Cleaner:
    """Strict inner-train Layer1, fit on a MODULE-LOCAL column subset.

    ``numeric_cols`` / ``categorical_cols`` scope Layer1 to the environment
    factors actually present in a given module combination (e.g. a T-only combo
    sees the 12 Terrain/DEM columns and never sees H/C/E/L columns, so an absent
    module cannot influence the cleanup). Every statistic (missingness, near-zero
    variance, duplicate, redundancy, median imputation, LC one-hot schema) is
    fitted ONLY on env.loc[ids] (the authorized training rows) and then applied
    to any requested rows. No validation row or label can influence the state.

    Duplicate/redundancy keep-ties use a deterministic, label-independent rule:
      1. lower missing rate, 2. no-NaN before has-NaN (data quality),
      3. higher variance (effective information), 4. fixed column-name order.
    There is NO "new over old" source priority (GLO30 vs ASTER is FORBIDDEN).
    """

    def __init__(self, env: pd.DataFrame, numeric_cols=None, categorical_cols=None):
        self.env = env
        self.numeric_cols = list(numeric_cols) if numeric_cols is not None else list(ALL_ENV_NUMERIC)
        self.categorical_cols = (list(categorical_cols) if categorical_cols is not None
                                 else ([LC_CLASS] if LC_CLASS in env.columns else []))
        self.fitted_ = False
        self._variances_ = {}

    def fit(self, ids):
        ids = list(ids)
        require(len(ids) > 0 and len(ids) == len(set(ids)), "Layer1 fit needs unique nonempty training IDs")
        cols = self.numeric_cols
        if not cols:
            # S-only: no environment factors -> Layer1 is a no-op (empty state).
            self._variances_ = {}
            self.kept_numeric_ = []
            self.medians_ = {}
            self.missing_rate_ = []
            self.dropped_ = dict(missing=[], near_zero_var=[], duplicate=[], redundant=[])
            self.lc_categories_ = []
            self.lc_onehot_cols_ = ()
            self.lc_category_counts_ = {}
            self.training_ids_ = tuple(ids)
            self.training_id_hash_ = ids_hash(ids)
            self.fit_calls_ = 1
            self.fitted_ = True
            return self
        X = self.env.loc[ids, cols].to_numpy(dtype=float)
        alive = [True] * len(cols)
        dropped = dict(missing=[], near_zero_var=[], duplicate=[], redundant=[])

        missing_rate = np.isnan(X).mean(axis=0)

        # per-column variance (for the effective-information tie-break), NaN-safe
        self._variances_ = {}
        for j in range(len(cols)):
            nn = X[:, j][~np.isnan(X[:, j])]
            self._variances_[cols[j]] = float(np.var(nn)) if nn.size > 1 else 0.0

        # 1. missingness
        for j in range(len(cols)):
            if missing_rate[j] > MISSING_DROP_RATIO:
                alive[j] = False
                dropped["missing"].append(cols[j])

        # 2. near-zero variance (on non-NaN values)
        for j in range(len(cols)):
            if not alive[j]:
                continue
            col = X[:, j]
            nn = col[~np.isnan(col)]
            if nn.size == 0:
                alive[j] = False
                dropped["near_zero_var"].append(cols[j])
                continue
            n_unique = len(np.unique(nn))
            mode_ratio = float(np.mean(nn == np.nanmedian(nn)))
            if n_unique <= NVZ_MIN_UNIQUE or mode_ratio > NVZ_MODE_RATIO:
                alive[j] = False
                dropped["near_zero_var"].append(cols[j])

        # 3. exact duplicates (NaN-aware), keep lower keep-rank
        surv = [j for j in range(len(cols)) if alive[j]]
        for a in range(len(surv)):
            i = surv[a]
            if not alive[i]:
                continue
            for b in range(a + 1, len(surv)):
                k = surv[b]
                if not alive[k]:
                    continue
                if _nan_identical(X[:, i], X[:, k]):
                    ri = self._keep_rank(cols[i], missing_rate[i], i)
                    rk = self._keep_rank(cols[k], missing_rate[k], k)
                    if ri <= rk:
                        alive[k] = False
                        dropped["duplicate"].append(cols[k])
                    else:
                        alive[i] = False
                        dropped["duplicate"].append(cols[i])

        # 4. redundancy |Spearman r| > threshold (median-imputed training)
        surv = [j for j in range(len(cols)) if alive[j]]
        if len(surv) >= 2:
            Xc = np.where(np.isnan(X[:, surv]), np.nanmedian(X[:, surv], axis=0), X[:, surv])
            dfc = pd.DataFrame(Xc, columns=[cols[j] for j in surv])
            corr = dfc.corr(method="spearman").to_numpy()
            for a in range(len(surv)):
                i = surv[a]
                if not alive[i]:
                    continue
                for b in range(a + 1, len(surv)):
                    k = surv[b]
                    if not alive[k]:
                        continue
                    if abs(corr[a, b]) > REDUNDANCY_THRESHOLD:
                        ri = self._keep_rank(cols[i], missing_rate[i], i)
                        rk = self._keep_rank(cols[k], missing_rate[k], k)
                        if ri <= rk:
                            alive[k] = False
                            dropped["redundant"].append(cols[k])
                        else:
                            alive[i] = False
                            dropped["redundant"].append(cols[i])

        kept = [cols[j] for j in range(len(cols)) if alive[j]]
        kept_idx = [j for j in range(len(cols)) if alive[j]]
        medians = {}
        for j, c in zip(kept_idx, kept):
            m = float(np.nanmedian(X[:, j]))
            require(np.isfinite(m), f"Layer1 kept fully-missing column {c}")
            medians[c] = m

        # 5. LC_class one-hot category schema (training rows only)
        self.lc_categories_ = []
        self.lc_onehot_cols_ = ()
        self.lc_category_counts_ = {}
        if LC_CLASS in self.categorical_cols:
            from collections import Counter
            cat = self.env.loc[ids, LC_CLASS].to_numpy()
            keys = [_cat_key(v) for v in cat]
            counts = Counter(k for k in keys if k is not None)
            present = sorted(counts.keys())
            self.lc_category_counts_ = {c: int(counts[c]) for c in present}
            self.lc_categories_ = present
            self.lc_onehot_cols_ = tuple(f"{LC_CLASS}_{c}" for c in present)

        self.kept_numeric_ = kept
        self.medians_ = medians
        self.missing_rate_ = missing_rate.tolist()
        self.dropped_ = dropped
        self.training_ids_ = tuple(ids)
        self.training_id_hash_ = ids_hash(ids)
        self.fit_calls_ = 1
        self.fitted_ = True
        return self

    def _keep_rank(self, feat, missing_rate, col_index):
        # deterministic, label-independent total order (lower tuple => KEEP)
        miss = missing_rate if not np.isnan(missing_rate) else 1.0
        has_nan = 1.0 if miss > 0 else 0.0
        var = self._variances_.get(feat, 0.0)
        return (miss, has_nan, -var, feat)

    def impute_numeric(self, ids, cols):
        require(self.fitted_, "Layer1 not fitted")
        require(all(c in self.kept_numeric_ for c in cols), "Requested a dropped Layer1 column")
        X = self.env.loc[list(ids), cols].to_numpy(dtype=float)
        med = np.array([self.medians_[c] for c in cols])
        out = np.where(np.isnan(X), med, X)
        require(np.isfinite(out).all(), "Non-finite after Layer1 imputation")
        return np.ascontiguousarray(out, dtype=float)

    def onehot(self, ids):
        require(self.fitted_, "Layer1 not fitted")
        if not self.lc_categories_:
            return None
        cat = self.env.loc[list(ids), LC_CLASS].to_numpy()
        keys = np.array([_cat_key(v) for v in cat], dtype=object)
        out = np.zeros((len(ids), len(self.lc_categories_)), dtype=float)
        lookup = {c: k for k, c in enumerate(self.lc_categories_)}
        for r, key in enumerate(keys):
            if key in lookup:
                out[r, lookup[key]] = 1.0
        return np.ascontiguousarray(out, dtype=float)

    def lc_unknown_categories(self, ids):
        """Sorted category keys present in ids but absent from the training schema."""
        require(self.fitted_, "Layer1 not fitted")
        if LC_CLASS not in self.categorical_cols:
            return []
        cat = self.env.loc[list(ids), LC_CLASS].to_numpy()
        keys = [_cat_key(v) for v in cat]
        return sorted({k for k in keys if k is not None and k not in self.lc_categories_})

    def state_summary(self):
        require(self.fitted_, "Layer1 not fitted")
        return native(dict(
            training_ids=self.training_ids_, training_id_hash=self.training_id_hash_,
            kept_numeric=self.kept_numeric_, medians=self.medians_,
            missing_rate=self.missing_rate_, dropped=self.dropped_,
            lc_categories=self.lc_categories_, lc_onehot_cols=self.lc_onehot_cols_,
            lc_category_counts=self.lc_category_counts_,
            fit_calls=self.fit_calls_,
        ))

    def state_hash(self):
        return object_hash(self.state_summary())
