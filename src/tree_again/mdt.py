from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from imodels import FIGSClassifier
from interpret.glassbox import ExplainableBoostingClassifier
try:
    from tabpfn import TabPFNClassifier
except ImportError:
    TabPFNClassifier = None

from scipy.optimize import brentq
from sklearn.model_selection import StratifiedKFold, KFold

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)

from xgboost import XGBClassifier, XGBRegressor

from .wrappers import (
    LicketySPLITClassifierStage,
    LicketySPLITRegressorStage,
    CLARITreeRegressorStage,
)

from .weightedseparable import WeightedSeparable
from sklearn.tree import DecisionTreeClassifier
from .sklearndefertree import SklearnDeferTree

from licketysplit import ThresholdGuessBinarizer




PLACEHOLDER_DEFAULT = 99
CATEGORICAL_EPS_DEFAULT = 1e-8

def _to_dataframe(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    return pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])])


def _to_binary_y(y, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    if y.ndim != 1 or y.shape[0] != n:
        raise ValueError("y must be shape (n_samples,)")
    vals = set(np.unique(y).tolist())
    if not vals.issubset({0, 1}):
        raise ValueError(f"y must be binary 0/1; got {sorted(vals)}")
    return y

def _to_binary_pred_vector(pred, n: int, name: str = "bb_pred") -> np.ndarray:
    pred = np.asarray(pred, dtype=np.int64)
    if pred.ndim != 1 or pred.shape[0] != n:
        raise ValueError(f"{name} must be shape (n_samples,).")
    vals = set(np.unique(pred).tolist())
    if not vals.issubset({0, 1}):
        raise ValueError(f"{name} must be binary 0/1; got {sorted(vals)}")
    return pred.astype(np.uint8, copy=False) & np.uint8(1)

def _is_binary_01(y) -> bool:
    vals = set(np.unique(np.asarray(y)).tolist())
    return vals.issubset({0, 1})


def _to_multiclass_y(y, n: int, max_classes: int = 50) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    if y.ndim != 1 or y.shape[0] != n:
        raise ValueError("y must be shape (n_samples,)")
    if np.any(y < 0):
        raise ValueError("classification labels must be nonnegative integers 0,1,...,C-1")

    max_y = int(y.max())
    n_classes = max_y + 1
    if n_classes > max_classes:
        raise ValueError(f"Expected at most {max_classes} classes; got {n_classes}")

    present = np.zeros(n_classes, dtype=bool)
    present[y] = True
    if not np.all(present):
        raise ValueError("classification labels must be contiguous 0,1,...,C-1")

    return y.astype(np.int64, copy=False)


def _to_multiclass_pred_vector(pred, n: int, n_classes: int, name: str = "bb_pred") -> np.ndarray:
    pred = np.asarray(pred, dtype=np.int64)
    if pred.ndim != 1 or pred.shape[0] != n:
        raise ValueError(f"{name} must be shape (n_samples,).")
    if np.any(pred < 0) or np.any(pred >= n_classes):
        raise ValueError(f"{name} values must be in 0,1,...,{n_classes - 1}")
    return pred.astype(np.int64, copy=False)


def _to_regression_y(y, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or y.shape[0] != n:
        raise ValueError("y must be shape (n_samples,)")
    if not np.all(np.isfinite(y)):
        raise ValueError("regression y must be finite")
    return y


def _to_regression_pred_vector(pred, n: int, name: str = "bb_pred") -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    if pred.ndim != 1 or pred.shape[0] != n:
        raise ValueError(f"{name} must be shape (n_samples,).")
    if not np.all(np.isfinite(pred)):
        raise ValueError(f"{name} must be finite")
    return pred

def _validate_grouped_categories(
    X_df: pd.DataFrame,
    grouped_categories: Optional[List[List[int]]],
) -> Tuple[
    Optional[List[List[int]]],
    Dict[str, int],          # feature_name -> group_id
    Dict[str, int],          # feature_name -> position within group
    Dict[int, List[str]],    # group_id -> feature names in order
]:
    if grouped_categories is None:
        return None, {}, {}, {}

    if not isinstance(grouped_categories, list):
        raise ValueError("grouped_categories must be None or a list of lists of original column indices.")

    p = X_df.shape[1]
    used = set()
    feature_to_group: Dict[str, int] = {}
    feature_to_pos: Dict[str, int] = {}
    group_feature_names: Dict[int, List[str]] = {}

    for g_idx, group in enumerate(grouped_categories):
        if not isinstance(group, list) or len(group) < 2:
            raise ValueError("Each grouped_categories entry must be a list of at least 2 original column indices.")

        cur_names: List[str] = []
        for pos, col_idx in enumerate(group):
            if not isinstance(col_idx, int):
                raise ValueError("Grouped categorical indices must be ints.")
            if col_idx < 0 or col_idx >= p:
                raise ValueError(f"Grouped categorical column index {col_idx} out of range for {p} columns.")
            if col_idx in used:
                raise ValueError(f"Column index {col_idx} appears in more than one categorical group.")
            used.add(col_idx)

            col_name = str(X_df.columns[col_idx])
            vals = set(pd.unique(X_df.iloc[:, col_idx]).tolist())
            if not vals.issubset({0, 1}):
                raise ValueError(
                    f"Grouped categorical column '{col_name}' must contain only numeric 0/1 values; "
                    f"got values {sorted(vals)}."
                )
            
            col = X_df.iloc[:, col_idx]
            vals = set(pd.unique(col).tolist())
            if pd.api.types.is_bool_dtype(col):
                raise ValueError(
                    f"Grouped categorical column '{col_name}' must be numeric 0/1, not bool."
                )

            feature_to_group[col_name] = g_idx
            feature_to_pos[col_name] = pos
            cur_names.append(col_name)

        group_feature_names[g_idx] = cur_names

    return grouped_categories, feature_to_group, feature_to_pos, group_feature_names


def _threshold_is_binary_split(threshold: float, eps: float) -> bool:
    thr = float(threshold)
    return (eps < thr) and (thr < 1.0 - eps)

def _compute_stage_weights(n: int, deferred_mask: np.ndarray, c: float) -> Tuple[np.ndarray, float]:
    deferred_mask = np.asarray(deferred_mask, dtype=bool)
    d = int(deferred_mask.sum())
    if d <= 0:
        raise ValueError("Deferred set is empty when computing weights.")
    mu = float(c) * float(n) / float(d + float(c) * n)
    w = np.where(deferred_mask, 1.0, 1.0 - mu).astype(np.float64, copy=False)
    return w, mu

# for each feature, return feature_name -> (sorted_unique_values, quantiles_in_[0,1])
def _compute_empirical_quantile_maps(
    X_df: pd.DataFrame,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    # mid-rank empirical quantiles:
    # q(v) = (#{X < v} + 0.5 * #{X = v}) / n
    maps: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for col in X_df.columns:
        vals = pd.to_numeric(X_df[col], errors="coerce").to_numpy(dtype=np.float64, copy=False)

        uniq, counts = np.unique(vals, return_counts=True)
        cum_before = np.cumsum(counts) - counts
        qs = (cum_before + 0.5 * counts).astype(np.float64) / float(vals.size)

        maps[str(col)] = (uniq.astype(np.float64, copy=False), qs)

    return maps


# given the values we want to map to quantiles, and the unique values to unique quantiles
def _apply_empirical_cdf(values: np.ndarray, grid_x: np.ndarray, grid_q: np.ndarray) -> np.ndarray:
    # map values to mid-rank quantiles by exact lookup only.
    # for values not exactly in grid_x, we take the quantile of the largest observed value <= that value.
    # (no interpolation)
    values = values.astype(np.float64, copy=False)
    out = np.empty(values.shape, dtype=np.float64)

    if grid_x.size == 1:
        out.fill(float(grid_q[0]))  # constant feature -> mid-rank 0.5
        return out

    # idx = largest j such that grid_x[j] <= values[i]
    idx = np.searchsorted(grid_x, values, side="right") - 1

    # below the minimum observed value -> 0
    below = idx < 0
    out[below] = 0.0

    # otherwise use the mid-rank quantile of the largest observed value <= value
    valid = ~below
    out[valid] = grid_q[idx[valid]]

    return out


# map a threshold t into quantile space
def _threshold_to_quantile_lower_open(threshold, grid_x, grid_q):
    # for lower bound in (lo, hi]
    thr = float(threshold)
    if np.isneginf(thr):
        return 0.0
    if np.isposinf(thr):
        return 1.0

    idx = np.searchsorted(grid_x, thr, side="right") # first observed value > thr
    if idx >= grid_x.size:
        return 1.0
    return float(grid_q[idx])

def _threshold_to_quantile_upper_closed(threshold, grid_x, grid_q):
    # for upper bound in (lo, hi]
    thr = float(threshold)
    if np.isneginf(thr):
        return 0.0
    if np.isposinf(thr):
        return 1.0

    idx = np.searchsorted(grid_x, thr, side="right") - 1 # largest observed value <= thr
    if idx < 0:
        return 0.0
    return float(grid_q[idx])


# takes a state (conjunction of path constraints) and converts its stored raw bounds into quantile-space bounds.
def _state_bounds_in_quantile_space(
    state: IntervalState,
    quantile_maps: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Tuple[float, float]]:

    out: Dict[str, Tuple[float, float]] = {}

    for feat, (lo_raw, hi_raw) in state.bounds.items():
        grid_x, grid_q = quantile_maps[str(feat)]

        lo_q = 0.0 if np.isneginf(lo_raw) else _threshold_to_quantile_lower_open(lo_raw, grid_x, grid_q)
        hi_q = 1.0 if np.isposinf(hi_raw) else _threshold_to_quantile_upper_closed(hi_raw, grid_x, grid_q)

        # safety clamp
        lo_q = min(max(lo_q, 0.0), 1.0)
        hi_q = min(max(hi_q, 0.0), 1.0)

        out[str(feat)] = (lo_q, hi_q)

    return out

def _is_history_feature_name(feature_name: str) -> bool:
    return str(feature_name).startswith("hist_")


def _strip_history_bounds_from_state(state: IntervalState) -> IntervalState:
    out = state.clone()
    out.bounds = {
        feat: bounds
        for feat, bounds in out.bounds.items()
        if not _is_history_feature_name(feat)
    }
    return out


def _strip_history_bounds_from_states(
    states: List[IntervalState],
) -> List[IntervalState]:
    return [_strip_history_bounds_from_state(st) for st in states]


def _distances_to_deferred_region_quantile_l1(
    X_df: pd.DataFrame,
    feasible_states: List[IntervalState],
    quantile_maps: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    n = len(X_df)

    # precompute quantile coordinates for all columns once
    Z: Dict[str, np.ndarray] = {}
    for feat in X_df.columns:
        grid_x, grid_q = quantile_maps[str(feat)]
        vals = pd.to_numeric(X_df[feat], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        Z[str(feat)] = _apply_empirical_cdf(vals, grid_x, grid_q)

    best = np.full(n, np.inf, dtype=np.float64)

    for state in feasible_states:
        cur = np.zeros(n, dtype=np.float64)

        # numeric / ordinal features
        q_bounds = _state_bounds_in_quantile_space(state, quantile_maps)

        for feat, (a, b) in q_bounds.items():
            # skip one-hot categorical columns here; they are handled below
            if feat in state.feature_to_group:
                continue

            z = Z[feat]
            cur += np.maximum(a - z, 0.0) + np.maximum(z - b, 0.0)

        # categorical feature groups
        for group_id, feat_names in state.group_feature_names.items():
            k = len(feat_names)

            # if this group has never been constrained, then every category is allowed
            vec = state.categorical_vectors.get(group_id)
            if vec is None:
                continue # our category isn't banned, so it is allowed, so 0 distance added

            # shape (n, k)
            group_vals = np.column_stack([X_df[f] for f in feat_names])

            # for a threshold guessed dataset, boolean matrix where it is 1 if the value was 0.
            # the category of a data point is the unique column with value 0 (because x<=0.5 being 0 means x>0.5, or True for binary)
            # however, we are on the original dataset, so we need the opposite convention
            zero_mask = group_vals > 0.5 

            cat_idx = np.argmax(zero_mask, axis=1) # row i's category position, unique according to category definition
            status = vec[cat_idx] # status of that category for each row, this is in -1, 0, or 1. banned, maybe, or certainly true. maybe and certainly true are in the allowed region.

            # allowed if status is 0 or 1, disallowed if -1
            cur += 0.5 * (status == -1).astype(np.float64)

        best = np.minimum(best, cur)

    return best

def _distances_and_projected_dataframe_to_deferred_region_quantile_l1(
    X_df: pd.DataFrame,
    feasible_states: List[IntervalState],
    quantile_maps: Dict[str, Tuple[np.ndarray, np.ndarray]],
    deferred_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    n = len(X_df)
    X_proj = X_df.copy()

    if deferred_mask is None:
        deferred_mask = np.zeros(n, dtype=bool)
    else:
        deferred_mask = np.asarray(deferred_mask, dtype=bool)

    # quantile coords for all columns
    Z: Dict[str, np.ndarray] = {}
    for feat in X_df.columns:
        grid_x, grid_q = quantile_maps[str(feat)]
        vals = pd.to_numeric(X_df[feat], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        Z[str(feat)] = _apply_empirical_cdf(vals, grid_x, grid_q)

    best = np.full(n, np.inf, dtype=np.float64)

    # store best projected quantile coordinates for numeric columns
    best_proj_q: Dict[str, np.ndarray] = {
        str(feat): Z[str(feat)].copy() for feat in X_df.columns
    }

    # for each numeric feature, store whether the winning projection landed on:
    # 0 = unchanged / interior, 1 = lower bound, 2 = upper bound
    best_proj_side: Dict[str, np.ndarray] = {
        str(feat): np.zeros(n, dtype=np.uint8) for feat in X_df.columns
    }

    # for each numeric feature, store the winning raw thresholds from the state
    # these are only used when the projection changed on that coordinate
    best_proj_lower_raw: Dict[str, np.ndarray] = {
        str(feat): np.full(n, np.nan, dtype=np.float64) for feat in X_df.columns
    }
    best_proj_upper_raw: Dict[str, np.ndarray] = {
        str(feat): np.full(n, np.nan, dtype=np.float64) for feat in X_df.columns
    }

    # store best categorical repair per group
    best_cat_choice: Dict[int, np.ndarray] = {}

    for state in feasible_states:
        cur = np.zeros(n, dtype=np.float64)

        cand_proj_q = {
            str(feat): Z[str(feat)].copy() for feat in X_df.columns
        }
        cand_proj_side: Dict[str, np.ndarray] = {
            str(feat): np.zeros(n, dtype=np.uint8) for feat in X_df.columns
        }
        cand_proj_lower_raw: Dict[str, np.ndarray] = {
            str(feat): np.full(n, np.nan, dtype=np.float64) for feat in X_df.columns
        }
        cand_proj_upper_raw: Dict[str, np.ndarray] = {
            str(feat): np.full(n, np.nan, dtype=np.float64) for feat in X_df.columns
        }
        cand_cat_choice: Dict[int, np.ndarray] = {}

        q_bounds = _state_bounds_in_quantile_space(state, quantile_maps)

        # numeric / ordinal projection by clamping quantiles
        # note this is vectorized across samples
        for feat, (a, b) in q_bounds.items():
            if feat in state.feature_to_group:
                continue

            z = Z[feat]
            z_proj = np.minimum(np.maximum(z, a), b) # projecting onto an interval. if z is in (a,b), nothing changes.
            cand_proj_q[feat] = z_proj # did we project onto a or b, this will tell us

            # remember whether each row projected to the lower bound, upper bound, or stayed inside
            go_lo = z < a
            go_hi = z > b
            cand_proj_side[feat][go_lo] = 1
            cand_proj_side[feat][go_hi] = 2

            # remember the raw state thresholds corresponding to those bounds
            lo_raw, hi_raw = state.bounds.get(feat, (-np.inf, np.inf))
            if np.isfinite(lo_raw):
                cand_proj_lower_raw[feat][go_lo] = float(lo_raw)
            if np.isfinite(hi_raw):
                cand_proj_upper_raw[feat][go_hi] = float(hi_raw)

            cur += np.maximum(a - z, 0.0) + np.maximum(z - b, 0.0) # the contribution to the L1 distance

        # grouped one-hot categorical projection
        for group_id, feat_names in state.group_feature_names.items():
            group_vals = np.column_stack([X_df[f] for f in feat_names])
            zero_mask = group_vals > 0.5 # True iff feature is 1.
            cat_idx = np.argmax(zero_mask, axis=1) # argmax returns the True. since exactly one column is 1, argmax gives its index of the 1 feature.

            vec = state.categorical_vectors.get(group_id) # -1 banned, 0 unknown, 1 forced true
            if vec is None:
                # all categories allowed; keep original category
                cand_cat_choice[group_id] = cat_idx.copy()
                continue

            allowed = (vec != -1) # True if category j is allowed, False if banned
            if np.any(~allowed): # if any samples are using a banned category
                bad = (vec[cat_idx] == -1) # which samples are using a banned category (not allowed by the state)
                cur += 0.5 * bad.astype(np.float64) # this categorical violation costs 0.5

                proj_idx = cat_idx.copy()
                # for rows that are using a banned category, swap them to a fallback
                if np.any(allowed):
                    allowed_idx = np.flatnonzero(allowed) # choose the most frequent category that is allowed
                    counts = np.bincount(cat_idx, minlength=len(vec))
                    allowed_counts = counts[allowed_idx]

                    fallback = int(allowed_idx[np.argmax(allowed_counts)])
                    proj_idx[bad] = fallback
                cand_cat_choice[group_id] = proj_idx # if we project onto this state, row i should use proj_idx[i]
            else:
                cand_cat_choice[group_id] = cat_idx.copy()

        # for rows where this state is better, update the best distance, and the projected nuemric quantiles and projected categorical choices
        improved = cur < best
        if np.any(improved):
            best[improved] = cur[improved]
            for feat in best_proj_q:
                best_proj_q[feat][improved] = cand_proj_q[feat][improved]
                best_proj_side[feat][improved] = cand_proj_side[feat][improved]
                best_proj_lower_raw[feat][improved] = cand_proj_lower_raw[feat][improved]
                best_proj_upper_raw[feat][improved] = cand_proj_upper_raw[feat][improved]
            for group_id, proj_idx in cand_cat_choice.items():
                if group_id not in best_cat_choice:
                    best_cat_choice[group_id] = proj_idx.copy()
                else:
                    best_cat_choice[group_id][improved] = proj_idx[improved]

    # do not move deferred points themselves
    nondef = ~deferred_mask

    # map projected numeric quantiles back to observed raw values
    # if a quantile changed to the lower state bound, map back to the smallest observed value > lower threshold
    # if a quantile changed to the upper state bound, map back to the largest observed value <= upper threshold
    # if unchanged / interior, keep the original raw value
    some_state = feasible_states[0] if len(feasible_states) > 0 else None
    for feat in X_df.columns:
        feat = str(feat)

        # skip grouped one-hot categorical features here; repaired below
        if some_state is not None and feat in some_state.feature_to_group:
            continue

        # original raw feature value of sample i
        raw_vals = pd.to_numeric(X_df[feat], errors="coerce").to_numpy(dtype=np.float64, copy=False).copy()
        grid_x, grid_q = quantile_maps[feat] # the sorted unique observed values and their quantiles

        side = best_proj_side[feat] # 0 means feature didn't move, 1 means moved to lower, 2 means moved to upper
        lower_thr = best_proj_lower_raw[feat]
        upper_thr = best_proj_upper_raw[feat]

        # rows projected to the lower bound
        lo_mask = nondef & (side == 1)
        if np.any(lo_mask):
            lo_thr = lower_thr[lo_mask]
            # smallest observed value strictly greater than the lower threshold
            idx_lo = np.searchsorted(grid_x, lo_thr, side="right")
            idx_lo = np.clip(idx_lo, 0, len(grid_x) - 1)
            raw_vals[lo_mask] = grid_x[idx_lo]

        # rows projected to the upper bound
        hi_mask = nondef & (side == 2)
        if np.any(hi_mask):
            hi_thr = upper_thr[hi_mask]
            # largest observed value less than or equal to the upper threshold
            idx_hi = np.searchsorted(grid_x, hi_thr, side="right") - 1
            idx_hi = np.clip(idx_hi, 0, len(grid_x) - 1)
            raw_vals[hi_mask] = grid_x[idx_hi]

        # rows with side == 0 keep their original raw value
        X_proj[feat] = raw_vals

    # categorical repair
    if len(feasible_states) > 0:
        template_state = feasible_states[0]
        for group_id, feat_names in template_state.group_feature_names.items():
            if group_id not in best_cat_choice:
                continue
            proj_idx = best_cat_choice[group_id] # what category index was the best choice per sample

            group_arr = np.column_stack([X_proj[f] for f in feat_names]).astype(float, copy=True)
            for i in np.flatnonzero(nondef):
                group_arr[i, :] = 0.0 # to make sure of, here we want 1 for True, 0 for false
                group_arr[i, proj_idx[i]] = 1.0

            for j, feat_name in enumerate(feat_names):
                X_proj[feat_name] = group_arr[:, j]

    return best, X_proj

# deferred points keep weight 1.
# non-deferred points receive normalized exponential weights based on the quantile L1 distance to nearest defer leaf bounding box
# gamma = 0 recovers the current constant (1-mu) weighting.
# gamma = -1 will binary search for the one with the dessired properties
def _compute_stage_weights_soft_deferred_region(
    X_df: pd.DataFrame,
    deferred_mask: np.ndarray,
    feasible_states: List[IntervalState],
    c: float,
    gamma: float,
    quantile_maps: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Tuple[np.ndarray, float]:
    deferred_mask = np.asarray(deferred_mask, dtype=bool)
    d = int(deferred_mask.sum())
    n = int(len(deferred_mask))

    if d <= 0:
        raise ValueError("Deferred set is empty when computing weights.")
    if n <= 0:
        raise ValueError("Need n > 0.")
    if gamma < 0:
        raise ValueError("gamma must be >= 0.")

    mu = float(c) * float(n) / float(d + float(c) * n)
    outside = ~deferred_mask

    w = np.ones(n, dtype=np.float64)

    if not np.any(outside):
        return w, mu

    # gamma = 0 exactly reproduces the uniform scheme
    if float(gamma) == 0.0:
        w[outside] = 1.0 - mu
        return w, mu

    dists = _distances_to_deferred_region_quantile_l1(
        X_df=X_df,
        feasible_states=feasible_states,
        quantile_maps=quantile_maps,
    )
    d_out = np.asarray(dists[outside], dtype=np.float64)
    if np.any(d_out < 0):
        raise ValueError("Distances must be nonnegative.")

    # gamma > 0 mode
    # normalized power weights with target average 1 - mu outside
    q = np.exp(-float(gamma) * np.log1p(d_out))  # equivalent to (1 + d_out)^(-gamma), but stabler
    q_sum = float(q.sum())

    # fallback for numerical underflow
    if q_sum <= 0.0 or not np.isfinite(q_sum):
        w[outside] = 1.0 - mu
        return w, mu

    target_mass = float((n - d) * (1.0 - mu))
    w[outside] = target_mass * (q / q_sum)

    if np.any(w[outside] > 1.0):
        print("Some non-deferred weights are greater than 1")

    return w, mu

def _compute_stage_weights_soft_deferred_region_scaled_nondeferred(
    X_df: pd.DataFrame,
    deferred_mask: np.ndarray,
    feasible_states: List[IntervalState],
    c: float,
    gamma: float,
    quantile_maps: Dict[str, Tuple[np.ndarray, np.ndarray]],
    return_projected_df: bool = False,
) -> Tuple[np.ndarray, float, Optional[pd.DataFrame]]:
    deferred_mask = np.asarray(deferred_mask, dtype=bool)
    d = int(deferred_mask.sum())
    n = int(len(deferred_mask))

    if d <= 0:
        raise ValueError("Deferred set is empty when computing weights.")
    if n <= 0:
        raise ValueError("Need n > 0.")
    if gamma < 0:
        raise ValueError("gamma must be >= 0.")

    mu = float(c)
    outside = ~deferred_mask

    w = np.ones(n, dtype=np.float64)
    X_proj = None

    if return_projected_df:
        dists, X_proj = _distances_and_projected_dataframe_to_deferred_region_quantile_l1(
            X_df=X_df,
            feasible_states=feasible_states,
            quantile_maps=quantile_maps,
            deferred_mask=deferred_mask,
        )
    else:
        dists = _distances_to_deferred_region_quantile_l1(
            X_df=X_df,
            feasible_states=feasible_states,
            quantile_maps=quantile_maps,
        )

    if not np.any(outside):
        return w, mu, X_proj

    if float(gamma) == 0.0:
        w[outside] = 1.0 - mu
        return w, mu, X_proj

    d_out = np.asarray(dists[outside], dtype=np.float64)
    if np.any(d_out < 0):
        raise ValueError("Distances must be nonnegative.")

    w[outside] = (1.0 - mu) * np.exp(-float(gamma) * np.log1p(d_out))
    return w, mu, X_proj

def _leaf_index_matrix_for_ensemble(model, X_df: pd.DataFrame) -> np.ndarray:
    # return an (n_samples, n_trees) integer matrix of leaf ids.

    if isinstance(model, RandomForestRegressor):
        return model.apply(X_df).astype(np.int64, copy=False)

    if isinstance(model, XGBRegressor):
        return np.asarray(model.apply(X_df), dtype=np.int64)

    if isinstance(model, RandomForestClassifier):
        return model.apply(X_df).astype(np.int64, copy=False)

    if isinstance(model, XGBClassifier):
        # may cause trouble with multiclass, but that isn't supported
        return np.asarray(model.apply(X_df), dtype=np.int64)

    raise ValueError(
        "set_weights_using_bb_leaves=True currently supports only "
        "black_box_type='xgb'/'xgboost' or 'rf'/'random_forest'. "
        f"Got model type {type(model).__name__}."
    )


def _compute_stage_weights_from_black_box_leaf_similarity(
    X_df: pd.DataFrame,
    deferred_mask: np.ndarray,
    bb_model,
    c: float,
    gamma: float,
    chunk_size: int = 256,
) -> Tuple[np.ndarray, float, Optional[pd.DataFrame]]:
    # deferred points get weight 1.
    # for each non-deferred point x, compute its similarity to every deferred point z:
    # sim(x, z) = proportion of trees where x and z fall in the same leaf.
    # s(x) = max_{z deferred} sim(x, z)
    # w(x) = (1 - mu) * s(x)^gamma
    deferred_mask = np.asarray(deferred_mask, dtype=bool)
    d = int(deferred_mask.sum())
    n = int(len(deferred_mask))

    if d <= 0:
        raise ValueError("Deferred set is empty when computing weights.")
    if n <= 0:
        raise ValueError("Need n > 0.")
    if gamma < 0:
        raise ValueError("gamma must be >= 0.")

    mu = float(c)
    if mu < 0.0 or mu > 1.0:
        raise ValueError("For this weighting scheme, c is used directly as mu and must be in [0, 1].")

    outside = ~deferred_mask
    w = np.ones(n, dtype=np.float64)

    if not np.any(outside):
        return w, mu, None

    if float(gamma) == 0.0:
        w[outside] = 1.0 - mu
        return w, mu, None

    leaves = _leaf_index_matrix_for_ensemble(bb_model, X_df)

    leaves_out = leaves[outside]
    leaves_def = leaves[deferred_mask]

    if leaves_out.ndim != 2 or leaves_def.ndim != 2:
        raise ValueError("Leaf matrix must be 2D: (n_samples, n_trees).")

    n_out, n_trees = leaves_out.shape
    n_def = leaves_def.shape[0]

    max_sim = np.zeros(n_out, dtype=np.float64)

    for start in range(0, n_out, int(chunk_size)):
        end = min(start + int(chunk_size), n_out)

        # counts[j, k] = number of trees where nondef point j
        # landed in the same leaf as deferred point k.
        counts = np.zeros((end - start, n_def), dtype=np.uint16)

        for t in range(n_trees):
            counts += (leaves_out[start:end, t, None] == leaves_def[None, :, t])

        max_sim[start:end] = counts.max(axis=1).astype(np.float64) / float(n_trees)

    max_sim = np.clip(max_sim, 0.0, 1.0)
    w[outside] = (1.0 - mu) * np.power(max_sim, float(gamma))

    return w, mu, None

def _compute_top_down_weights(
    self,
    X_df: pd.DataFrame,
    deferred_mask: np.ndarray,
    feasible_states: List[IntervalState],
) -> Tuple[np.ndarray, float]:
    return _compute_stage_weights_soft_deferred_region_scaled_nondeferred(
        X_df=X_df,
        deferred_mask=deferred_mask,
        feasible_states=feasible_states,
        c=self.c,
        gamma=self.gamma,
        quantile_maps=self.quantile_maps_,
    )

# guarantee to never defer
def _huge_eta_from_n(n: int) -> float:
    return float(1000.0 * (float(n) ** 3))

@dataclass
class _ConstantBlackBox:
    value: int

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), int(self.value), dtype=np.uint8)

@dataclass
class _FixedPredictionBlackBox:
    predictions: np.ndarray

    def predict(self, X) -> np.ndarray:
        n = len(X)
        if n != len(self.predictions):
            raise ValueError(
                "This MDTChain was fit with a fixed prediction vector. "
                "For prediction on data with a different number of rows, pass bb_pred=..."
            )
        return np.asarray(self.predictions).copy()

# immutable, RuleLiteral("age", 30, True) means age <= 30, False means >30
@dataclass(frozen=True)
class RuleLiteral:
    feature_name: str
    threshold: float
    truth_value: bool

    def pretty(self) -> str:
        op = "<=" if self.truth_value else ">"
        return f"{self.feature_name} {op} {self.threshold}"

# decision tree node
@dataclass
class RuleTreeNode:
    feature_name: Optional[str] = None
    threshold: Optional[float] = None
    left: Optional["RuleTreeNode"] = None
    right: Optional["RuleTreeNode"] = None
    action: Optional[Any] = None # 0 / 1 / "defer" / "black_box"
    missing_go_to_left: Optional[bool] = None # for xgboost missing value stuff, doesn't really matter

    def is_leaf(self) -> bool:
        return self.action is not None

    def clone(self) -> "RuleTreeNode": # recursive, completely new tree
        return copy.deepcopy(self)

class LinearLeafAction:
    __slots__ = ("label", "stage_index", "leaf_index")

    def __init__(self, label: str, stage_index: int, leaf_index: int):
        self.label = str(label)
        self.stage_index = int(stage_index)
        self.leaf_index = int(leaf_index)

    def __repr__(self):
        return self.label

    def __str__(self):
        return self.label


# @dataclass
# class IntervalState:
#     bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict) # name -> low,high (low, high]

#     def clone(self) -> "IntervalState":
#         return IntervalState(bounds=dict(self.bounds))

#     # tightens the interval
#     def add_literal(self, lit: RuleLiteral) -> "IntervalState":
#         out = self.clone()
#         lo, hi = out.bounds.get(lit.feature_name, (-np.inf, np.inf))
#         if lit.truth_value:
#             hi = min(hi, float(lit.threshold))
#         else:
#             lo = max(lo, float(lit.threshold))
#         out.bounds[lit.feature_name] = (lo, hi)
#         return out

#     # list of x<5, x<7, y<3, etc, makes into name -> min,max map
#     def add_path(self, path: List[RuleLiteral]) -> "IntervalState":
#         out = self
#         for lit in path:
#             out = out.add_literal(lit)
#         return out

#     # returning true means the threshold you pass is redundant given what bounds we already have and can be removed
#     # always go left
#     def implies_true(self, feature_name: str, threshold: float) -> bool:
#         _, hi = self.bounds.get(feature_name, (-np.inf, np.inf))
#         return hi <= threshold

#     # if this returns true, the <=threshold is always false, go right.
#     def implies_false(self, feature_name: str, threshold: float) -> bool:
#         lo, _ = self.bounds.get(feature_name, (-np.inf, np.inf))
#         return lo > threshold

@dataclass
class IntervalState:
    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict) # name -> (low, high]
    categorical_vectors: Dict[int, np.ndarray] = field(default_factory=dict) # group_id -> array in {0,1,-1}
    feature_to_group: Dict[str, int] = field(default_factory=dict)
    feature_to_pos: Dict[str, int] = field(default_factory=dict)
    group_feature_names: Dict[int, List[str]] = field(default_factory=dict)
    categorical_eps: float = CATEGORICAL_EPS_DEFAULT

    known_binary: Dict[str, int] = field(default_factory=dict) # feature -> 0/1
    defer_tuple: Tuple[str, ...] = field(default_factory=tuple)
    impossible: bool = False
    history_feature_columns_by_stage: Dict[int, List[str]] = field(default_factory=dict)

    def clone(self) -> "IntervalState":
        return IntervalState(
            bounds=dict(self.bounds),
            categorical_vectors={g: v.copy() for g, v in self.categorical_vectors.items()},
            feature_to_group=self.feature_to_group,
            feature_to_pos=self.feature_to_pos,
            group_feature_names=self.group_feature_names,
            categorical_eps=self.categorical_eps,
            known_binary=dict(self.known_binary),
            defer_tuple=tuple(self.defer_tuple),
            impossible=bool(self.impossible),
            history_feature_columns_by_stage=self.history_feature_columns_by_stage,
        )

    def _is_known_binary_feature(self, feature_name: str, threshold: float) -> bool:
        return (
            str(feature_name).startswith("hist_")
            and self._is_binary_group_threshold(float(threshold))
        )

    def _set_known_binary_no_propagate(self, feature_name: str, value: int) -> None:
        feature_name = str(feature_name)
        value = int(value)

        if value not in {0, 1}:
            raise ValueError("known binary value must be 0 or 1")

        old = self.known_binary.get(feature_name)
        if old is not None and int(old) != value:
            self.impossible = True
            return

        self.known_binary[feature_name] = value


    def _set_known_binary(self, feature_name: str, value: int) -> None:
        feature_name = str(feature_name)
        value = int(value)

        self._set_known_binary_no_propagate(feature_name, value)
        if self.impossible:
            return

        self._propagate_known_history_binary(feature_name, value)

    def _propagate_known_history_binary(self, feature_name: str, value: int) -> None:
        feature_name = str(feature_name)
        value = int(value)

        # actual_j = 0 => all crosses and tuples for stage j are 0
        if value == 0 and feature_name.startswith("hist_actual_reached_stage_"):
            try:
                stage_idx = int(feature_name.removeprefix("hist_actual_reached_stage_"))
            except ValueError:
                return

            hist_cols = self.history_feature_columns_by_stage.get(stage_idx, [])
            cross_prefix = f"hist_actual_x_hist_prev_s{stage_idx - 1}_leaf_"
            tuple_prefix = f"hist_defer_tuple_to_s{stage_idx}__"

            for col in hist_cols:
                col = str(col)
                if col.startswith(cross_prefix) or col.startswith(tuple_prefix):
                    self._set_known_binary_no_propagate(col, 0)

            return

        # actual_j = 1 plus known marginal => known cross
        if value == 1 and feature_name.startswith("hist_actual_reached_stage_"):
            try:
                stage_idx = int(feature_name.removeprefix("hist_actual_reached_stage_"))
            except ValueError:
                return

            marginal_prefix = f"hist_prev_s{stage_idx - 1}_leaf_"
            cross_prefix = "hist_actual_x_"

            for col, val in list(self.known_binary.items()):
                if col.startswith(marginal_prefix):
                    self._set_known_binary_no_propagate(
                        cross_prefix + col,
                        int(val),
                    )

            return

        # marginal known, and actual known => cross known
        if feature_name.startswith("hist_prev_s"):
            # feature name looks like:
            # hist_prev_s{k}_leaf_<label>
            try:
                rest = feature_name.removeprefix("hist_prev_s")
                prev_stage_str = rest.split("_leaf_", 1)[0]
                prev_stage_idx = int(prev_stage_str)
            except Exception:
                return

            actual_col = f"hist_actual_reached_stage_{prev_stage_idx + 1}"
            actual_val = self.known_binary.get(actual_col)

            if actual_val is not None:
                self._set_known_binary_no_propagate(
                    f"hist_actual_x_{feature_name}",
                    int(actual_val) & int(value),
                )

            return

    def _is_categorical_feature(self, feature_name: str) -> bool:
        return feature_name in self.feature_to_group

    def _is_binary_group_threshold(self, threshold: float) -> bool: # are we in (0, 1) but not the endpoints, hopefully so 
        return _threshold_is_binary_split(threshold, self.categorical_eps)

    def _ensure_group_vector(self, out: "IntervalState", group_id: int) -> np.ndarray:
        if group_id not in out.categorical_vectors:
            out.categorical_vectors[group_id] = np.zeros(
                len(out.group_feature_names[group_id]),
                dtype=np.int8,
            )
        return out.categorical_vectors[group_id]

    def _propagate_group_vector(self, vec: np.ndarray) -> None:
        # if any entry is 1, force all others to -1
        ones = np.flatnonzero(vec == 1)
        if ones.size > 0:
            keep = int(ones[0])
            vec.fill(-1)
            vec[keep] = 1
            return

        # if exactly one entry is still unknown, it must be the true one
        # the rule is really if all but 1 are -1, then the remaining one must be 1
        # but we already know nothing can be 1, because if we had a 1 it was handled in the earlier case
        zeros = np.flatnonzero(vec == 0)
        if zeros.size == 1:
            vec[int(zeros[0])] = 1

    # returns 1 if the feature is definitely true, -1 if this feature is definitely false, and 0 if you don't know.
    def _categorical_status(self, feature_name: str) -> int:
        group_id = self.feature_to_group[feature_name]
        pos = self.feature_to_pos[feature_name]
        vec = self.categorical_vectors.get(group_id)
        if vec is None:
            return 0
        return int(vec[pos])

    def add_literal(self, lit: RuleLiteral, propagate: bool = True) -> "IntervalState":
        out = self.clone()

        # the ordinary interval logic too
        lo, hi = out.bounds.get(lit.feature_name, (-np.inf, np.inf))
        # lit.truth_value means (feature <= threshold) is True
        if lit.truth_value:
            hi = min(hi, float(lit.threshold))
        else:
            lo = max(lo, float(lit.threshold))
        out.bounds[lit.feature_name] = (lo, hi)
        # note that we don't guarantee these bounds don't cross over, we aren't against that

        if out._is_known_binary_feature(lit.feature_name, lit.threshold):
            # lit.truth_value means feature <= 0.5, i.e. binary feature == 0.
            # not lit.truth_value means feature > 0.5, i.e. binary feature == 1.
            out._set_known_binary(
                lit.feature_name,
                0 if lit.truth_value else 1,
            )
        
        # for now, let categorical do both the numerical and the categorical, eventually we can do an if else
    
        # categorical logic for grouped 0/1 columns
        if out._is_categorical_feature(lit.feature_name) and out._is_binary_group_threshold(lit.threshold):
            group_id = out.feature_to_group[lit.feature_name]
            pos = out.feature_to_pos[lit.feature_name]
            vec = out._ensure_group_vector(out, group_id)

            # lit.truth_value means (feature <= threshold) is True
            # for a 0/1 feature with threshold near 0.5, that means feature == 0
            if lit.truth_value:
                vec[pos] = -1
            else:
                vec[pos] = 1

            if propagate:
                out._propagate_group_vector(vec)

        return out

    def add_path(self, path: List[RuleLiteral]) -> "IntervalState":
        out = self
        for lit in path:
            out = out.add_literal(lit)
        return out

    def implies_true(self, feature_name: str, threshold: float) -> bool:
        # ordinary interval logic
        if self.impossible:
            return True

        feature_name = str(feature_name)

        if self._is_known_binary_feature(feature_name, threshold):
            val = self.known_binary.get(feature_name)
            if val == 0:
                return True # feature <= 0.5
            if val == 1:
                return False

        _, hi = self.bounds.get(feature_name, (-np.inf, np.inf))
        if hi <= threshold:
            return True


        # categorical logic:
        # for grouped binary column and threshold near 0.5,
        # "feature <= threshold" means feature == 0
        if self._is_categorical_feature(feature_name) and self._is_binary_group_threshold(threshold):
            status = self._categorical_status(feature_name)
            if status == -1:
                return True
            if status == 1:
                return False

        return False

    def implies_false(self, feature_name: str, threshold: float) -> bool:
        # ordinary interval logic
        if self.impossible:
            return True

        feature_name = str(feature_name)

        if self._is_known_binary_feature(feature_name, threshold):
            val = self.known_binary.get(feature_name)
            if val == 1:
                return True # feature > 0.5
            if val == 0:
                return False

        lo, _ = self.bounds.get(feature_name, (-np.inf, np.inf))
        if lo >= threshold:
            return True

        # categorical logic:
        # for grouped binary column and threshold near 0.5,
        # "feature <= threshold" is false iff feature == 1
        if self._is_categorical_feature(feature_name) and self._is_binary_group_threshold(threshold):
            status = self._categorical_status(feature_name)
            if status == 1:
                return True
            if status == -1:
                return False

        return False

    def is_feasible(self) -> bool:
        if self.impossible:
            return False

        for lo, hi in self.bounds.values():
            if float(lo) >= float(hi):
                return False

        for vec in self.categorical_vectors.values():
            if np.all(vec == -1):
                return False
            if np.sum(vec == 1) > 1:
                return False

        return True


@dataclass
class _PseudoWeightedSEPBlackBox:
    tg: ThresholdGuessBinarizer
    sep: WeightedSeparable
    placeholder: int

    def predict(self, X) -> np.ndarray:
        X_df = _to_dataframe(X)
        Xb = self.tg.transform(X_df).astype(np.uint8, copy=False)
        raw = self.sep.predict(Xb, bb_pred=None, placeholder=self.placeholder).astype(np.uint8, copy=False)
        if np.any(raw == np.uint8(self.placeholder)):
            raise RuntimeError(
                "Pseudo WeightedSEP black box still deferred on some samples. Huge eta was intended to prevent deferral."
            )
        return raw & np.uint8(1)


class _TreeOps:
    # to do deduplication etc. unique bounds thing and unique categorical thing.
    @staticmethod
    def state_signature(state: IntervalState):
        bounds_sig = tuple(
            sorted((str(f), float(lo), float(hi)) for f, (lo, hi) in state.bounds.items())
        )
        cat_sig = tuple(
            sorted((int(g), tuple(int(x) for x in v.tolist())) for g, v in state.categorical_vectors.items())
        )
        binary_sig = tuple(
            sorted((str(f), int(v)) for f, v in state.known_binary.items())
        )
        tuple_sig = tuple(str(x) for x in state.defer_tuple)
        return (bounds_sig, cat_sig, binary_sig, tuple_sig, bool(state.impossible))

    # apply all literals to tighten the bounds
    @staticmethod
    def state_from_path(
        path: List[RuleLiteral],
        template_state: IntervalState,
    ) -> IntervalState:
        st = IntervalState(
            feature_to_group=template_state.feature_to_group,
            feature_to_pos=template_state.feature_to_pos,
            group_feature_names=template_state.group_feature_names,
            categorical_eps=template_state.categorical_eps,
        )
        for lit in path:  # this handles categorical literals too
            st = st.add_literal(lit)
        return st

    # batch version
    @staticmethod
    def states_from_paths(
        paths: List[List[RuleLiteral]],
        template_state: IntervalState,
    ) -> List[IntervalState]:
        return [_TreeOps.state_from_path(p, template_state=template_state) for p in paths]

    # otal number of leaves at/below RuleTreeNode
    @staticmethod
    def count_leaves_rule_tree(node: Optional[RuleTreeNode]) -> int:
        if node is None:
            return 0
        if node.is_leaf() or (node.left is None and node.right is None):
            return 1
        return _TreeOps.count_leaves_rule_tree(node.left) + _TreeOps.count_leaves_rule_tree(node.right)

    # number of splits taken for one sample? 
    @staticmethod
    def path_length_one_rule_tree(node: RuleTreeNode, row: pd.Series) -> int:
        cur = node
        depth = 0
        while cur is not None and not (cur.is_leaf() or (cur.left is None and cur.right is None)):
            feat = str(cur.feature_name)
            thr = float(cur.threshold)
            val = row[feat]
            if pd.isna(val):
                go_left = True if cur.missing_go_to_left is None else bool(cur.missing_go_to_left)
            else:
                go_left = bool(val <= thr)
            depth += 1
            cur = cur.left if go_left else cur.right
        return depth

    # apply above to every row
    @staticmethod
    def predict_path_lengths_rule_tree(root: RuleTreeNode, X_df: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(X_df), dtype=np.int64)
        for i in range(len(X_df)):
            out[i] = _TreeOps.path_length_one_rule_tree(root, X_df.iloc[i])
        return out

    # traverse a row and record the path
    @staticmethod
    def traverse_rule_tree(root: RuleTreeNode, row: pd.Series) -> Tuple[Any, int, List[RuleLiteral]]:
        cur = root
        depth = 0
        path: List[RuleLiteral] = []
        while cur is not None and not (cur.is_leaf() or (cur.left is None and cur.right is None)):
            feat = str(cur.feature_name)
            thr = float(cur.threshold)
            val = row[feat]
            if pd.isna(val):
                go_left = True if cur.missing_go_to_left is None else bool(cur.missing_go_to_left)
            else:
                go_left = bool(val <= thr)
            path.append(RuleLiteral(feat, thr, go_left))
            depth += 1
            cur = cur.left if go_left else cur.right
        action = None if cur is None else cur.action
        return action, depth, path

    # annoying conversion
    @staticmethod
    def build_rule_tree_from_sklearn_tree(estimator, feature_names: List[str]) -> RuleTreeNode:
        tree_ = estimator.tree_
        children_left = tree_.children_left
        children_right = tree_.children_right
        features = tree_.feature
        thresholds = tree_.threshold
        values = tree_.value

        def rec(node_id: int) -> RuleTreeNode:
            left_id = int(children_left[node_id])
            right_id = int(children_right[node_id])
            if left_id == -1 and right_id == -1:
                val = values[node_id]
                try:
                    if val.ndim >= 2 and val.shape[-1] >= 1:
                        action = int(np.argmax(val[0]))
                    else:
                        action = float(np.ravel(val)[0])
                except Exception:
                    action = float(np.ravel(val)[0])
                return RuleTreeNode(action=action)

            feat_idx = int(features[node_id])
            thr = float(thresholds[node_id])
            return RuleTreeNode(
                feature_name=str(feature_names[feat_idx]),
                threshold=thr,
                left=rec(left_id),
                right=rec(right_id),
                action=None,
                missing_go_to_left=True,
            )

        return rec(0)

    # real valued actions here becuaase of how boosting works
    @staticmethod
    def build_rule_tree_from_xgb_dataframe(df_tree: pd.DataFrame) -> RuleTreeNode:
        rows = {int(r["Node"]): r for _, r in df_tree.iterrows()}

        def child_node_id(v) -> int:
            if pd.isna(v):
                raise ValueError("Unexpected missing child id in XGBoost tree dump.")
            s = str(v)
            if "-" in s:
                return int(s.split("-")[1])
            return int(s)

        def rec(node_id: int) -> RuleTreeNode:
            r = rows[int(node_id)]
            feat = str(r["Feature"])
            if feat == "Leaf":
                return RuleTreeNode(action=float(r["Gain"]))

            yes_id = child_node_id(r["Yes"])
            no_id = child_node_id(r["No"])
            missing_id = child_node_id(r["Missing"])
            return RuleTreeNode(
                feature_name=str(feat),
                threshold=float(r["Split"]),
                left=rec(yes_id),
                right=rec(no_id),
                action=None,
                missing_go_to_left=bool(missing_id == yes_id),
            )

        return rec(0)

    @staticmethod
    def build_rule_tree_from_figs_tree(tree, feature_names: List[str]) -> RuleTreeNode:
        def rec(node) -> RuleTreeNode:
            if node is None:
                raise ValueError("Unexpected None FIGS node.")

            left = node.left
            right = node.right

            # leaf
            if left is None and right is None:
                value = node.value
                if value is None:
                    raise RuntimeError("FIGS leaf has no value attribute.")

                value_arr = np.asarray(value).reshape(-1)

                # preserve the numeric value exactly, like xGBoost leaf gain
                if value_arr.size == 1:
                    action = float(value_arr[0])
                else:
                    action = value_arr.astype(float).copy()

                return RuleTreeNode(action=action)

            feat_idx = int(node.feature)
            thr = float(node.threshold)

            return RuleTreeNode(
                feature_name=str(feature_names[feat_idx]),
                threshold=thr,
                left=rec(left),
                right=rec(right),
                action=None,
                missing_go_to_left=True,
            )

        return rec(tree)

    # build a tree from root to leaf paths
    @staticmethod
    def tree_from_decoded_paths(decoded_paths: List[List[Tuple[str, float, bool]]], actions: List[Any]) -> RuleTreeNode:
        root = RuleTreeNode()
        for path, action in zip(decoded_paths, actions):
            cur = root
            for feat_name, thr, truth_val in path:
                if cur.is_leaf():
                    raise RuntimeError("Invalid path set: attempted to split below a leaf.")
                if cur.feature_name is None:
                    cur.feature_name = str(feat_name)
                    cur.threshold = float(thr)
                    cur.left = RuleTreeNode()
                    cur.right = RuleTreeNode()
                else:
                    if cur.feature_name != str(feat_name) or float(cur.threshold) != float(thr):
                        raise RuntimeError(
                            "Inconsistent path set while rebuilding tree. "
                            f"Expected split ({cur.feature_name}, {cur.threshold}) but got ({feat_name}, {thr})."
                        )
                cur = cur.left if truth_val else cur.right
            cur.action = action
        return root

    # replace with subtee if provided or black box leaf
    @staticmethod
    def replace_defer_leaves_with_subtree(root: RuleTreeNode, subtree: Optional[RuleTreeNode]) -> RuleTreeNode:
        node = root.clone()

        def rec(cur: RuleTreeNode) -> RuleTreeNode:
            if cur.is_leaf():
                if cur.action == "defer":
                    if subtree is None:
                        return RuleTreeNode(action="black_box")
                    return subtree.clone()
                return cur
            cur.left = rec(cur.left)
            cur.right = rec(cur.right)
            return cur

        return rec(node)

    @staticmethod
    def simplify_single_tree(node: RuleTreeNode, state: IntervalState) -> RuleTreeNode:
        if node is None or node.is_leaf():
            return node
        feat = str(node.feature_name)
        thr = float(node.threshold)
        if state.implies_true(feat, thr): # we don't need the right subtree, cant happen
            return _TreeOps.simplify_single_tree(node.left, state)
        if state.implies_false(feat, thr): # we don't need the left
            return _TreeOps.simplify_single_tree(node.right, state)
        true_state = state.add_literal(RuleLiteral(feat, thr, True))
        false_state = state.add_literal(RuleLiteral(feat, thr, False))
        # recursively simplify both sides with the bounds tightened
        new_node = RuleTreeNode(
            feature_name=feat,
            threshold=thr,
            left=_TreeOps.simplify_single_tree(node.left, true_state),
            right=_TreeOps.simplify_single_tree(node.right, false_state),
            action=None,
            missing_go_to_left=node.missing_go_to_left,
        )
        return _TreeOps.prune_trivial_extensions(new_node)

    # for the dag, we don't just have one path (which we use an interval state to do the tightest bounds), but a list of them that you could come from
    @staticmethod
    def all_states_imply_true(states: List[IntervalState], feature_name: str, threshold: float) -> bool:
        return len(states) > 0 and all(s.implies_true(feature_name, threshold) for s in states)

    @staticmethod
    def all_states_imply_false(states: List[IntervalState], feature_name: str, threshold: float) -> bool:
        return len(states) > 0 and all(s.implies_false(feature_name, threshold) for s in states)

    @staticmethod
    def prune_trivial_extensions(node: Optional[RuleTreeNode]) -> Optional[RuleTreeNode]:
        if node is None:
            return None

        if node.is_leaf():
            return RuleTreeNode(action=node.action)

        # doing this recursively in post-order is crucial because collapsing lower nodes can create new trivial extensions higher up
        left_pruned = _TreeOps.prune_trivial_extensions(node.left)
        right_pruned = _TreeOps.prune_trivial_extensions(node.right)

        # if both sides collapse to the same leaf action, remove this split
        if (
            left_pruned is not None
            and right_pruned is not None
            and left_pruned.is_leaf()
            and right_pruned.is_leaf()
            and not isinstance(left_pruned.action, LinearLeafAction)
            and not isinstance(right_pruned.action, LinearLeafAction)
            and left_pruned.action == right_pruned.action # no combining linear leaves
        ):
            return RuleTreeNode(action=left_pruned.action)

        # at least one side is not a leaf, or the leaf actions differ, so we change nothing about our internal node
        return RuleTreeNode(
            feature_name=node.feature_name,
            threshold=node.threshold,
            left=left_pruned,
            right=right_pruned,
            action=None,
            missing_go_to_left=node.missing_go_to_left,
        )

    @staticmethod
    def simplify_dag_tree(node: RuleTreeNode, states: List[IntervalState]) -> RuleTreeNode:
        if node is None or node.is_leaf():
            return node
        feat = str(node.feature_name)
        thr = float(node.threshold)
        if _TreeOps.all_states_imply_true(states, feat, thr): # only if all of them find the right redundant can we remove it
            return _TreeOps.simplify_dag_tree(node.left, states)
        if _TreeOps.all_states_imply_false(states, feat, thr):
            return _TreeOps.simplify_dag_tree(node.right, states)
        # for each possible interval state, you must grow the paths
        true_states = [
            st
            for st in (s.add_literal(RuleLiteral(feat, thr, True)) for s in states)
            if st.is_feasible()
        ]

        false_states = [
            st
            for st in (s.add_literal(RuleLiteral(feat, thr, False)) for s in states)
            if st.is_feasible()
        ]

        if len(true_states) == 0:
            return _TreeOps.simplify_dag_tree(node.right, false_states)

        if len(false_states) == 0:
            return _TreeOps.simplify_dag_tree(node.left, true_states)

        # recursively simplify
        new_node = RuleTreeNode(
            feature_name=feat,
            threshold=thr,
            left=_TreeOps.simplify_dag_tree(node.left, true_states),
            right=_TreeOps.simplify_dag_tree(node.right, false_states),
            action=None,
            missing_go_to_left=node.missing_go_to_left,
        )
        return _TreeOps.prune_trivial_extensions(new_node)

    # collect all root to leaf paths that end in a certain action. standard dfs
    @staticmethod
    def collect_paths_to_action(
        node: RuleTreeNode,
        target_action: Any,
        cur: Optional[List[RuleLiteral]] = None,
        out: Optional[List[List[RuleLiteral]]] = None,
    ) -> List[List[RuleLiteral]]:
        if cur is None:
            cur = []
        if out is None:
            out = []
        if node is None:
            return out
        if node.is_leaf():
            if node.action == target_action:
                out.append(list(cur)) # list makes a copy
            return out
        lit_true = RuleLiteral(str(node.feature_name), float(node.threshold), True)
        cur.append(lit_true)
        _TreeOps.collect_paths_to_action(node.left, target_action, cur, out)
        cur.pop()
        lit_false = RuleLiteral(str(node.feature_name), float(node.threshold), False)
        cur.append(lit_false)
        _TreeOps.collect_paths_to_action(node.right, target_action, cur, out)
        cur.pop()
        return out


class _MDTTrainingMixin:
    @staticmethod
    def _maybe_scale_xgboost_weight_sensitive_params(
        params: Dict[str, Any],
        sample_weight,
        *,
        correct_xgboost_scale: bool,
    ) -> Dict[str, Any]:
        params = dict(params)

        if not correct_xgboost_scale or sample_weight is None:
            return params

        w = np.asarray(sample_weight, dtype=np.float64)
        if w.ndim != 1:
            raise ValueError("sample_weight must be 1D.")
        if w.size == 0:
            raise ValueError("sample_weight cannot be empty.")
        if np.any(w < 0):
            raise ValueError("sample_weight cannot contain negative values.")

        scale = float(w.sum()) / float(w.size)

        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"Invalid XGBoost weight scale: {scale}")

        params.setdefault("min_child_weight", 1.0)
        params.setdefault("reg_lambda", 1.0)
        params.setdefault("gamma", 0.0)

        for key in ("min_child_weight", "reg_lambda", "lambda", "gamma", "min_split_loss"):
            if key in params and params[key] is not None:
                params[key] = float(params[key]) * scale

        return params

    def _weighted_tabpfn_training_data(
        self,
        X_df: pd.DataFrame,
        y: np.ndarray,
        sample_weight,
        seed: int,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        # convert sample weights into an unweighted TabPFN training set.

        # sample_weight is None: use all rows.
        # weights are exactly 0/1: keep only rows with weight 1.
        # general nonnegative weights: sample n rows with replacement proportional to weights.
        
        n = len(y)

        if sample_weight is None:
            return X_df, y

        w = np.asarray(sample_weight, dtype=np.float64)

        if w.ndim != 1 or w.shape[0] != n:
            raise ValueError("sample_weight must be shape (n_samples,).")
        if np.any(w < 0):
            raise ValueError("sample_weight cannot contain negative values.")
        if not np.any(w > 0):
            raise ValueError("TabPFN cannot fit with all sample weights equal to zero.")

        is_binary_weight = np.all((w == 0.0) | (w == 1.0))
        if is_binary_weight:
            keep = w == 1.0
            # return X_df.iloc[keep].copy(), y[keep].copy()
            return X_df.iloc[keep].reset_index(drop=True).copy(), y[keep].copy()

        params = getattr(self, "black_box_params", {})
        raw_resample_n = params.get("tabpfn_resample_n", n)
        resample_n = n if raw_resample_n is None else int(raw_resample_n)

        if resample_n <= 0:
            raise ValueError("tabpfn_resample_n must be positive.")

        p = w / float(w.sum())
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(n, size=resample_n, replace=True, p=p)

        #return X_df.iloc[idx].copy(), y[idx].copy()
        return X_df.iloc[idx].reset_index(drop=True).copy(), y[idx].copy()

    def _fit_black_box(self, X_df: pd.DataFrame, y: np.ndarray, sample_weight, seed: int):
        if self.task_ == "classification":
            return self._fit_classification_black_box(X_df, y, sample_weight, seed)
        return self._fit_regression_black_box(X_df, y, sample_weight, seed)
    
    def _fit_classification_black_box(self, X_df: pd.DataFrame, y: np.ndarray, sample_weight, seed: int):
        kind = self.black_box_type

        if kind in {"rf", "random_forest", "randomforest"}:
            params = dict(self.black_box_params)
            params.setdefault("n_estimators", 200)
            params.setdefault("random_state", seed)
            params.setdefault("n_jobs", -1)
            model = RandomForestClassifier(**params)
            model.fit(X_df, y, sample_weight=sample_weight)
            return model

        if kind in {"gbdt", "gradient_boosting", "boosted_decision_tree", "boosted_tree"}:
            params = dict(self.black_box_params)
            params.setdefault("n_estimators", 200)
            params.setdefault("learning_rate", 0.05)
            params.setdefault("max_depth", 3)
            params.setdefault("random_state", seed)
            model = GradientBoostingClassifier(**params)
            model.fit(X_df, y, sample_weight=sample_weight)
            return model

        if kind in {"xgb", "xgboost"}:
            params = dict(self.black_box_params)

            params.setdefault("n_estimators", 300)
            params.setdefault("max_depth", 6)
            params.setdefault("learning_rate", 0.05)
            params.setdefault("subsample", 1.0)
            params.setdefault("colsample_bytree", 1.0)
            if getattr(self, "n_classes_", 2) > 2:
                params.setdefault("objective", "multi:softmax")
                params.setdefault("num_class", int(self.n_classes_))
                params.setdefault("eval_metric", "mlogloss")
            else:
                params.setdefault("objective", "binary:logistic")
                params.setdefault("eval_metric", "logloss")
            params.setdefault("random_state", seed)

            params = self._maybe_scale_xgboost_weight_sensitive_params(
                params,
                sample_weight,
                correct_xgboost_scale=getattr(self, "correct_xgboost_scale", False),
            )

            model = XGBClassifier(**params)
            model.fit(X_df, y, sample_weight=sample_weight)
            return model
                
        if kind in {"figs", "figs_classifier"}:
            params = dict(self.black_box_params)
            params.setdefault("max_rules", 12)
            params.setdefault("max_trees", None)
            params.setdefault("min_impurity_decrease", 0.0)
            params.setdefault("max_features", None)
            params.setdefault("max_depth", None)
            params.setdefault("random_state", seed)

            model = FIGSClassifier(**params)
            model.fit(X_df.to_numpy(), y, sample_weight=sample_weight)
            return model

        if kind in {"weightedsep", "weighted_separable", "sep"}:
            return self._fit_pseudo_weightedsep_black_box(X_df, y, sample_weight=sample_weight, seed=seed)

        if kind in {"ebm", "explainable_boosting", "explainable_boosting_machine"}:
            if ExplainableBoostingClassifier is None:
                raise ImportError(
                    "black_box_type='ebm' requires interpret. Install with: pip install interpret"
                )

            params = dict(self.black_box_params)

            params.setdefault("random_state", seed)
            params.setdefault("interactions", 10)
            params.setdefault("max_rounds", 500)
            params.setdefault("learning_rate", 0.01)

            X_fit = X_df
            y_fit = np.asarray(y)
            w_fit = sample_weight

            if w_fit is not None:
                w_fit = np.asarray(w_fit, dtype=np.float64)

                if w_fit.ndim != 1 or w_fit.shape[0] != len(y_fit):
                    raise ValueError("sample_weight must be shape (n_samples,).")
                if np.any(w_fit < 0):
                    raise ValueError("sample_weight cannot contain negative values.")
                if not np.any(w_fit > 0):
                    raise ValueError("EBM cannot fit with all sample weights equal to zero.")

                keep = w_fit > 0.0
                X_fit = X_df.iloc[keep].reset_index(drop=True).copy()
                y_fit = y_fit[keep]
                w_fit = w_fit[keep]

            vals = np.unique(y_fit)
            if vals.size == 1:
                return _ConstantBlackBox(value=int(vals[0]))

            model = ExplainableBoostingClassifier(**params)
            model.fit(X_fit, y_fit, sample_weight=w_fit)
            return model

        if kind in {"tabpfn", "tab_pfn", "tabpfn_classifier"}:
            if TabPFNClassifier is None:
                raise ImportError(
                    "black_box_type='tabpfn' requires tabpfn. Install with: pip install tabpfn"
                )

            params = dict(self.black_box_params)

            # internal use
            params.pop("tabpfn_resample_n", None)

            # params.setdefault("random_state", seed)

            X_fit, y_fit = self._weighted_tabpfn_training_data(
                X_df=X_df,
                y=y,
                sample_weight=sample_weight,
                seed=seed,
            )

            # if the weighted subset/resample collapses to one class, TabPFN may fail.
            # in that case the correct weighted classifier is just constant anyway.
            vals = np.unique(y_fit)
            if vals.size == 1:
                return _ConstantBlackBox(value=int(vals[0]))

            model = TabPFNClassifier(**params)
            model.fit(X_fit, y_fit)
            return model

        raise ValueError(f"Unknown black_box_type='{self.black_box_type}'")

    def _fit_regression_black_box(self, X_df, y, sample_weight, seed):
        kind = self.black_box_type

        if kind in {"rf", "random_forest", "randomforest"}:
            params = dict(self.black_box_params)
            params.setdefault("n_estimators", 200)
            params.setdefault("random_state", seed)
            params.setdefault("n_jobs", -1)
            model = RandomForestRegressor(**params)
            model.fit(X_df, y, sample_weight=sample_weight)
            return model

        if kind in {"gbdt", "gradient_boosting", "boosted_decision_tree", "boosted_tree"}:
            params = dict(self.black_box_params)
            params.setdefault("n_estimators", 200)
            params.setdefault("learning_rate", 0.05)
            params.setdefault("max_depth", 3)
            params.setdefault("random_state", seed)
            model = GradientBoostingRegressor(**params)
            model.fit(X_df, y, sample_weight=sample_weight)
            return model

        if kind in {"xgb", "xgboost"}:
            params = dict(self.black_box_params)
            params.setdefault("n_estimators", 300)
            params.setdefault("max_depth", 6)
            params.setdefault("learning_rate", 0.05)
            params.setdefault("subsample", 1.0)
            params.setdefault("colsample_bytree", 1.0)
            params.setdefault("objective", "reg:squarederror")
            params.setdefault("eval_metric", "rmse")
            params.setdefault("random_state", seed)

            params = self._maybe_scale_xgboost_weight_sensitive_params(
                params,
                sample_weight,
                correct_xgboost_scale=getattr(self, "correct_xgboost_scale", False),
            )

            model = XGBRegressor(**params)
            model.fit(X_df, y, sample_weight=sample_weight)
            return model

        raise ValueError(f"Unknown regression black_box_type={self.black_box_type!r}")

    # def _predict_black_box(self, model, X_df: pd.DataFrame) -> np.ndarray:
    #     if isinstance(model, _PseudoWeightedSEPBlackBox):
    #         return model.predict(X_df).astype(np.uint8, copy=False) & np.uint8(1)
    #     return np.asarray(model.predict(X_df), dtype=np.uint8) & np.uint8(1)

    def _predict_black_box(self, model, X_df: pd.DataFrame) -> np.ndarray:
        pred = model.predict(X_df)

        if self.task_ == "classification":
            pred = np.asarray(pred, dtype=np.int64)
            if self.is_binary_classification_:
                return pred.astype(np.uint8, copy=False) & np.uint8(1)

            if np.any(pred < 0) or np.any(pred >= self.n_classes_):
                raise ValueError("black-box predicted class outside 0,...,C-1")
            return pred

        pred = np.asarray(pred, dtype=np.float64)
        if not np.all(np.isfinite(pred)):
            raise ValueError("black-box regression predictions must be finite")
        return pred

    # when we want the black box to be an interpretable tree
    def _fit_pseudo_weightedsep_black_box(self, X_df: pd.DataFrame, y: np.ndarray, sample_weight, seed: int) -> _PseudoWeightedSEPBlackBox:
        n = len(y)
        huge_eta = _huge_eta_from_n(n)
        params = dict(self.black_box_params)

        sep_params = dict(params.get("sep_params", {}))
        sep_params.setdefault("lambda_leaf", self.lambda_leaf)
        sep_params.setdefault("depth_budget", self.depth_budget)
        sep_params.setdefault("lookahead_k", self.lookahead_k)

        # tg = ThresholdGuessBinarizer(**tg_params)
        # Xb = tg.fit_transform(X_df, y).astype(np.uint8, copy=False)
        if self.same_binarization:
            tg = self.encoder_
            Xb = tg.transform(X_df).astype(np.uint8, copy=False)
        else:
            tg, Xb = self._fit_threshold_guesser(X_df, y, sample_weight=sample_weight, seed=seed)
        dummy_bb = np.zeros(len(y), dtype=np.uint8)

        sep = WeightedSeparable()
        sep.fit(
            Xb,
            y,
            dummy_bb,
            lambda_leaf=float(sep_params["lambda_leaf"]),
            eta_defer=float(huge_eta),
            depth_budget=int(sep_params["depth_budget"]),
            lookahead_k=int(sep_params["lookahead_k"]),
            sample_weights=sample_weight,
        )

        raw_train = sep.predict(Xb, bb_pred=None, placeholder=self.placeholder).astype(np.uint8, copy=False)
        if np.any(raw_train == np.uint8(self.placeholder)):
            raise RuntimeError("Pseudo WeightedSEP black box deferred on training data despite huge eta.")

        return _PseudoWeightedSEPBlackBox(tg=tg, sep=sep, placeholder=self.placeholder)

    # when we want to replace the black box with an interpretable tree at the end
    def _fit_elimination_model(self, X_df: pd.DataFrame, y: np.ndarray, sample_weight, seed: int) -> _PseudoWeightedSEPBlackBox:
        n = len(y)
        huge_eta = _huge_eta_from_n(n)
        params = dict(self.eliminate_black_box_params)

        sep_params = dict(params.get("sep_params", {}))
        sep_params.setdefault("lambda_leaf", self.lambda_leaf)
        sep_params.setdefault("depth_budget", self.depth_budget)
        sep_params.setdefault("lookahead_k", self.lookahead_k)

        # tg = ThresholdGuessBinarizer(**tg_params)
        # Xb = tg.fit_transform(X_df, y).astype(np.uint8, copy=False)
        if self.same_binarization and self.encoder_ is not None:
            tg = self.encoder_
            Xb = tg.transform(X_df).astype(np.uint8, copy=False)
        else:
            tg, Xb = self._fit_threshold_guesser(X_df, y, sample_weight=sample_weight, seed=seed)
        dummy_bb = np.zeros(len(y), dtype=np.uint8)

        sep = WeightedSeparable()
        sep.fit(
            Xb,
            y,
            dummy_bb,
            lambda_leaf=float(sep_params["lambda_leaf"]),
            eta_defer=float(huge_eta),
            depth_budget=int(sep_params["depth_budget"]),
            lookahead_k=int(sep_params["lookahead_k"]),
            sample_weights=sample_weight,
        )

        raw_train = sep.predict(Xb, bb_pred=None, placeholder=self.placeholder).astype(np.uint8, copy=False)
        if np.any(raw_train == np.uint8(self.placeholder)):
            raise RuntimeError("Elimination WeightedSEP still deferred on training data despite huge eta.")

        return _PseudoWeightedSEPBlackBox(tg=tg, sep=sep, placeholder=self.placeholder)

    def _fit_defer_stage(
        self,
        *,
        X_df_stage: pd.DataFrame,
        Xb_stage: np.ndarray,
        y: np.ndarray,
        bb_pred: np.ndarray,
        lambda_leaf: float,
        eta_defer: float,
        depth_budget: int,
        lookahead_k: int,
        sample_weights,
        seed: int,
    ):
        kind = getattr(self, "defer_tree_type", "weightedsep")

        if self.task_ == "classification" and self.is_binary_classification_:
            if kind in {"sklearn", "sklearn_cart", "cart"}:
                params = dict(getattr(self, "sklearn_defer_tree_params", {}))
                params.setdefault("max_depth", int(depth_budget))
                params.setdefault("min_samples_leaf", 1)
                params.setdefault("random_state", int(seed))
                params.setdefault("criterion", "gini")
                params.setdefault("eta", float(eta_defer))
                params.setdefault("ccp_alpha", 0.0)

                stage = SklearnDeferTree(**params)
                stage.fit(
                    X_df_stage,
                    y,
                    bb_pred,
                    sample_weights=sample_weights,
                )
                return stage

            if kind == "weightedsep":
                stage = WeightedSeparable()
                stage.fit(
                    Xb_stage,
                    y,
                    bb_pred,
                    lambda_leaf=float(lambda_leaf),
                    eta_defer=float(eta_defer),
                    depth_budget=int(depth_budget),
                    lookahead_k=int(lookahead_k),
                    sample_weights=sample_weights,
                )
                return stage

            if kind in {"licketysplit", "lickety", "ls"}:
                stage = LicketySPLITClassifierStage(
                    lambda_leaf=float(lambda_leaf),
                    eta_defer=float(eta_defer),
                    depth_budget=int(depth_budget),
                    lookahead_k=int(lookahead_k),
                    placeholder=int(self.placeholder),
                )
                stage.fit(
                    Xb_stage,
                    y,
                    bb_pred=bb_pred,
                    sample_weights=sample_weights,
                )
                return stage

            raise ValueError(f"Unknown defer_tree_type={kind!r}")

        if self.task_ == "classification":
            # multiclass: do not use WeightedSeparable or SklearnDeferTree (for now)
            stage = LicketySPLITClassifierStage(
                lambda_leaf=float(lambda_leaf),
                eta_defer=float(eta_defer),
                depth_budget=int(depth_budget),
                lookahead_k=int(lookahead_k),
                placeholder=int(self.placeholder),
            )
            stage.fit(
                Xb_stage,
                y,
                bb_pred=bb_pred,
                sample_weights=sample_weights,
            )
            return stage

        if self.task_ == "regression":
            stage = LicketySPLITRegressorStage(
                lambda_leaf=float(lambda_leaf),
                eta_defer=float(eta_defer),
                depth_budget=int(depth_budget),
                lookahead_k=int(lookahead_k),
            )
            stage.fit(
                Xb_stage,
                y,
                bb_pred=bb_pred,
                sample_weights=sample_weights,
            )
            return stage

        if self.task_ == "linear_regression":
            params = dict(getattr(self, "claritree_params", {}))
            params.setdefault("lambda_leaf", float(lambda_leaf))
            params.setdefault("eta_defer", float(eta_defer))
            params.setdefault("depth_budget", int(depth_budget))

            stage = CLARITreeRegressorStage(**params)
            stage.fit(
                X_df_stage,
                y,
                bb_pred=bb_pred,
                sample_weights=sample_weights,
            )
            return stage
        
        
    # def _eventual_mask_single(self, model: WeightedSeparable, Xb: np.ndarray) -> np.ndarray:
    #     raw = model.predict(Xb, bb_pred=None, placeholder=self.placeholder).astype(np.uint8, copy=False)
    #     return self._is_defer_prediction(raw)

    def _raw_stage_predict(self, stage, X_df_stage, Xb_stage):
        if isinstance(stage, SklearnDeferTree):
            raw = stage.predict(X_df_stage, bb_pred=None, placeholder=self.placeholder)
        elif self.task_ == "linear_regression":
            raw = stage.predict(X_df_stage, bb_pred=None, placeholder=np.nan)
        elif self.task_ == "regression":
            raw = stage.predict(Xb_stage, bb_pred=None, placeholder=np.nan)
        else:
            raw = stage.predict(Xb_stage, bb_pred=None, placeholder=self.placeholder)

        if self.task_ == "classification":
            return np.asarray(raw, dtype=np.int64)
        return np.asarray(raw, dtype=np.float64)

    def _eventual_mask_single(self, stage, X_df_stage, Xb_stage):
        raw = self._raw_stage_predict(stage, X_df_stage, Xb_stage)
        return self._is_defer_prediction(raw)

    # def _stage_leaf_stats(self, stage: WeightedSeparable) -> Tuple[int, int]:
    #     paths, actions = stage.get_tree_paths()
    #     l = len(actions)
    #     d = sum(1 for a in actions if int(a) == -1)
    #     return int(l), int(d)

    # def _stage_leaf_count(self, stage: WeightedSeparable) -> int:
    #     return self._raw_leaf_count_weightedsep_stage(stage)

    def _stage_leaf_stats(self, stage) -> Tuple[int, int]:
        if hasattr(stage, "leaf_actions"):
            actions = stage.leaf_actions()
        else:
            _, actions = stage.get_tree_paths()

        l = len(actions)
        d = sum(1 for a in actions if int(a) == -1)
        return int(l), int(d)


    def _stage_leaf_count(self, stage) -> int:
        if hasattr(stage, "leaf_actions"):
            return int(len(stage.leaf_actions()))
        paths, actions = stage.get_tree_paths()
        return int(len(actions))

    def _formula_black_box_leaf_count(self, model) -> int:
        # count the black box in the online formula only if it is a pseudo WeightedSEP tree.
        # otherwise just set l_W = 1.
        if isinstance(model, _PseudoWeightedSEPBlackBox):
            return int(self._raw_leaf_count_weightedsep_stage(model.sep))
        return 1

    # zeros out columns that don't provide any insight in differentiatng the deferred regions (predictively equivalent pruning of features)
    def _zero_out_constant_binarized_columns_from_states(
        self,
        Xb: np.ndarray,
        encoder: ThresholdGuessBinarizer,
        feasible_states: List[IntervalState],
    ) -> Tuple[np.ndarray, np.ndarray]:
        Xb_mod = Xb.copy()
        n_bin = len(encoder.thresholds_)
        zero_mask = np.zeros(n_bin, dtype=bool)

        if len(feasible_states) == 0:
            return Xb_mod, zero_mask

        for j, (raw_feat_idx, thr) in enumerate(encoder.thresholds_):
            feat_name = str(encoder.feature_names_in_[raw_feat_idx])
            thr = float(thr)

            all_true = all(st.implies_true(feat_name, thr) for st in feasible_states)
            all_false = all(st.implies_false(feat_name, thr) for st in feasible_states)

            if all_true or all_false:
                Xb_mod[:, j] = 0
                zero_mask[j] = True

        return Xb_mod, zero_mask

    def _out_of_fold_black_box_predictions(
        self,
        X_df: pd.DataFrame,
        y: np.ndarray,
        sample_weight,
        seed: int,
        stratify_mask: Optional[np.ndarray] = None,
        n_splits: int = 5,
    ) -> np.ndarray:

        n = len(y)
        if self.task_ == "classification":
            oof_pred = np.zeros(n, dtype=np.int64)
        else:
            oof_pred = np.zeros(n, dtype=np.float64)

        if self.task_ == "classification":
            stratify_labels = y if stratify_mask is None else np.asarray(stratify_mask, dtype=np.int64)

            vals, counts = np.unique(stratify_labels, return_counts=True)
            if len(vals) >= 2 and counts.min() >= 2:
                k = min(int(n_splits), int(counts.min()))
                splitter = StratifiedKFold(
                    n_splits=k,
                    shuffle=True,
                    random_state=seed,
                )
                splits = splitter.split(X_df, stratify_labels)
            else:
                k = min(int(n_splits), n)
                splitter = KFold(
                    n_splits=k,
                    shuffle=True,
                    random_state=seed,
                )
                splits = splitter.split(X_df)
        else:
            k = min(int(n_splits), n)
            splitter = KFold(
                n_splits=k,
                shuffle=True,
                random_state=seed,
            )
            splits = splitter.split(X_df)

        vals, counts = np.unique(stratify_labels, return_counts=True)

        if len(vals) >= 2 and counts.min() >= 2:
            k = min(int(n_splits), int(counts.min()))
            splitter = StratifiedKFold(
                n_splits=k,
                shuffle=True,
                random_state=seed,
            )
            splits = splitter.split(X_df, stratify_labels)
        else:
            k = min(int(n_splits), n)
            splitter = KFold(
                n_splits=k,
                shuffle=True,
                random_state=seed,
            )
            splits = splitter.split(X_df)

        for train_idx, val_idx in splits:
            X_train = X_df.iloc[train_idx]
            y_train = y[train_idx]

            if sample_weight is None:
                w_train = None
            else:
                w_train = np.asarray(sample_weight)[train_idx]

            fold_model = self._fit_black_box(
                X_train,
                y_train,
                sample_weight=w_train,
                seed=seed,
            )

            oof_pred[val_idx] = self._predict_black_box(
                fold_model,
                X_df.iloc[val_idx],
            )

        return oof_pred

    def _safe_history_token(self, x: Any) -> str:
        s = str(x)
        out = []
        for ch in s:
            if ch.isalnum():
                out.append(ch)
            elif ch in {"_", "-"}:
                out.append("_")
            else:
                out.append("_")
        return "".join(out)

    def _all_leaf_history_labels_for_stage(self, stage_idx: int) -> List[str]:
        root = self.get_stage_tree(stage_idx)

        labels: List[str] = []
        leaf_counter = 0

        def rec(node: RuleTreeNode) -> None:
            nonlocal leaf_counter

            if node is None:
                return

            if node.is_leaf() or (node.left is None and node.right is None):
                action = node.action

                if action == "defer":
                    label = f"hist_s{stage_idx}_defer_leaf_{leaf_counter}"
                else:
                    if self.task_ == "classification":
                        label = (
                            f"hist_s{stage_idx}_pred_leaf_{leaf_counter}_"
                            f"action_{self._safe_history_token(action)}"
                        )
                    else:
                        label = f"hist_s{stage_idx}_pred_leaf_{leaf_counter}"

                labels.append(label)
                leaf_counter += 1
                return

            rec(node.left)
            rec(node.right)

        rec(root)
        return labels

    def _stage_defer_path_label_pairs(
        self,
        stage_idx: int,
    ) -> List[Tuple[List[RuleLiteral], str]]:
        root = self.get_stage_tree(stage_idx)

        out: List[Tuple[List[RuleLiteral], str]] = []
        leaf_counter = 0

        def rec(node: RuleTreeNode, cur: List[RuleLiteral]) -> None:
            nonlocal leaf_counter

            if node is None:
                return

            if node.is_leaf() or (node.left is None and node.right is None):
                if node.action == "defer":
                    label = f"hist_s{stage_idx}_defer_leaf_{leaf_counter}"
                    out.append((list(cur), label))

                leaf_counter += 1
                return

            feat = str(node.feature_name)
            thr = float(node.threshold)

            cur.append(RuleLiteral(feat, thr, True))
            rec(node.left, cur)
            cur.pop()

            cur.append(RuleLiteral(feat, thr, False))
            rec(node.right, cur)
            cur.pop()

        rec(root, [])
        return out

    def _mark_state_after_defer_leaf(
        self,
        state: IntervalState,
        *,
        stage_idx: int,
        defer_leaf_label: str,
    ) -> IntervalState:
        out = state.clone()

        next_stage_idx = int(stage_idx) + 1

        # this state reaches the next stage
        actual_col = f"hist_actual_reached_stage_{next_stage_idx}"
        out._set_known_binary(actual_col, 1)

        # previous-stage marginal leaf one-hot
        all_prev_leaf_labels = self._all_leaf_history_labels_for_stage(stage_idx)
        for lab in all_prev_leaf_labels:
            marginal_col = f"hist_prev_s{stage_idx}_leaf_{lab}"
            val = 1 if lab == defer_leaf_label else 0

            out._set_known_binary(marginal_col, val)
            out._set_known_binary(f"hist_actual_x_{marginal_col}", val)

        # realized defer tuple
        out.defer_tuple = tuple(out.defer_tuple) + (str(defer_leaf_label),)

        # extra joint tuple columns, if they exist, for the next stage
        extra_prefix = f"hist_defer_tuple_to_s{next_stage_idx}__"
        tuple_col = extra_prefix + "__".join(out.defer_tuple)

        hist_cols = self.history_feature_columns_by_stage_.get(next_stage_idx, [])
        for col in hist_cols:
            if str(col).startswith(extra_prefix):
                out._set_known_binary(str(col), 1 if str(col) == tuple_col else 0)

        return out


    def _defer_tuple_labels_for_stage(
        self,
        X_df: pd.DataFrame,
        stage_idx: int,
        reached_mask: np.ndarray,
    ) -> pd.Series:
        reached_mask = np.asarray(reached_mask, dtype=bool)
        if stage_idx <= 0:
            return pd.Series([None] * len(X_df), index=X_df.index, dtype=object)

        labels_per_prev_stage: List[np.ndarray] = []

        for prev_idx in range(stage_idx):
            # If a previous stage itself used history features, rebuild the feature
            # frame that that previous stage expects.
            if self.add_history and not self.same_binarization and prev_idx > 0:
                X_prev = self._feature_frame_for_stage(X_df, prev_idx)
            else:
                X_prev = X_df

            labels = self._leaf_labels_for_stage(prev_idx, X_prev)
            labels = np.asarray(labels, dtype=object)
            labels_per_prev_stage.append(labels)

        out = np.empty(len(X_df), dtype=object)
        out[:] = None

        active_idx = np.flatnonzero(reached_mask)

        for i in active_idx:
            pieces = []
            ok = True

            for prev_idx, labels in enumerate(labels_per_prev_stage):
                lab = str(labels[i])

                # Only defer leaves are allowed in these tuple features.
                # If this ever fails for a reached sample, something is inconsistent.
                if "_defer_leaf_" not in lab:
                    ok = False
                    break

                pieces.append(lab)

            if ok:
                out[i] = f"hist_defer_tuple_to_s{stage_idx}__" + "__".join(pieces)

        return pd.Series(out, index=X_df.index, dtype=object)
    
    def _leaf_labels_for_stage(
        self,
        stage_idx: int,
        X_df: pd.DataFrame,
    ) -> np.ndarray:
        # return one categorical routing label per sample for stage_idx.

        root = self.get_stage_tree(stage_idx)
        
        needed_features = set()

        def collect_split_features(node: RuleTreeNode) -> None:
            if node is None:
                return
            if node.is_leaf() or (node.left is None and node.right is None):
                return
            needed_features.add(str(node.feature_name))
            collect_split_features(node.left)
            collect_split_features(node.right)

        collect_split_features(root)

        missing_features = sorted(f for f in needed_features if f not in X_df.columns)
        if missing_features:
            raise RuntimeError(
                f"Stage {stage_idx} tree needs missing columns: {missing_features}. "
                f"Available first columns: {list(X_df.columns)[:30]}"
            )

        labels_by_node_id: Dict[int, str] = {}
        leaf_counter = 0

        def assign(node: RuleTreeNode) -> None:
            nonlocal leaf_counter
            if node is None:
                return

            if node.is_leaf() or (node.left is None and node.right is None):
                action = node.action

                if action == "defer":
                    label = f"hist_s{stage_idx}_defer_leaf_{leaf_counter}"
                else:
                    if self.task_ == "classification":
                        label = (
                            f"hist_s{stage_idx}_pred_leaf_{leaf_counter}_"
                            f"action_{self._safe_history_token(action)}"
                        )
                    else:
                        label = f"hist_s{stage_idx}_pred_leaf_{leaf_counter}"

                labels_by_node_id[id(node)] = label
                leaf_counter += 1
                return

            assign(node.left)
            assign(node.right)

        assign(root)

        def traverse_to_leaf(node: RuleTreeNode, row: pd.Series) -> RuleTreeNode:
            cur = node
            while cur is not None and not (cur.is_leaf() or (cur.left is None and cur.right is None)):
                feat = str(cur.feature_name)
                thr = float(cur.threshold)
                val = row[feat]

                if pd.isna(val):
                    go_left = True if cur.missing_go_to_left is None else bool(cur.missing_go_to_left)
                else:
                    go_left = bool(val <= thr)

                cur = cur.left if go_left else cur.right

            return cur

        out = np.empty(len(X_df), dtype=object)
        for i in range(len(X_df)):
            leaf = traverse_to_leaf(root, X_df.iloc[i])
            out[i] = labels_by_node_id[id(leaf)]

        return out


    def _append_history_features_for_stage(
        self,
        X_df: pd.DataFrame,
        stage_idx: int,
        reached_mask: np.ndarray,
        *,
        fit: bool,
    ) -> pd.DataFrame:
        # add history features for stage_idx.
        # stage_idx = 0 gets no history.
        # what samples are real, what defer leaf did things just go into, and the cross product of these two

        if not self.add_history or stage_idx <= 0:
            return X_df

        reached_mask = np.asarray(reached_mask, dtype=bool)
        if reached_mask.ndim != 1 or reached_mask.shape[0] != len(X_df):
            raise ValueError("reached_mask must be shape (n_samples,)")

        prev_stage_idx = stage_idx - 1

        carried_standard_cols: List[str] = []

        if self.standard_history_not_memoryless and stage_idx > 1:
            X_prev_history = self._feature_frame_for_stage(
                X_df,
                stage_idx - 1,
            )

            carried_standard_cols = [
                c
                for c in self.history_feature_columns_by_stage_.get(
                    stage_idx - 1, []
                )
                if not str(c).startswith("hist_defer_tuple_to_s")
            ]
        else:
            X_prev_history = None

        if self.add_history and not self.same_binarization and prev_stage_idx > 0:
            if X_prev_history is not None:
                X_prev = X_prev_history
            else:
                X_prev = self._feature_frame_for_stage(X_df, prev_stage_idx)
        else:
            X_prev = X_df

        prev_labels = self._leaf_labels_for_stage(prev_stage_idx, X_prev)

        

        actual_col = f"hist_actual_reached_stage_{stage_idx}"
        label_prefix = f"hist_prev_s{prev_stage_idx}_leaf"

        # out = X_df.copy()

        # if X_prev_history is not None:
        #     for col in carried_standard_cols:
        #         out[col] = X_prev_history[col].to_numpy(
        #             dtype=np.uint8,
        #             copy=False,
        #         )

        # out[actual_col] = reached_mask.astype(np.uint8)

        new_cols = {}

        if X_prev_history is not None:
            for col in carried_standard_cols:
                new_cols[col] = X_prev_history[col].to_numpy(
                    dtype=np.uint8,
                    copy=False,
                )

        new_cols[actual_col] = reached_mask.astype(np.uint8)

        out = pd.concat(
            [
                X_df,
                pd.DataFrame(new_cols, index=X_df.index),
            ],
            axis=1,
        )

        leaf_ohe = pd.get_dummies(
            pd.Series(prev_labels, index=out.index),
            prefix=label_prefix,
            dtype=np.uint8,
        )

        extra_prefix = f"hist_defer_tuple_to_s{stage_idx}"

        if self.add_extra_history:
            tuple_labels = self._defer_tuple_labels_for_stage(
                X_df,
                stage_idx,
                reached_mask,
            )

            extra_ohe = pd.get_dummies(
                tuple_labels,
                dtype=np.uint8,
            )

            # keep only actual tuple columns
            extra_ohe = extra_ohe[
                [c for c in extra_ohe.columns if str(c).startswith(extra_prefix + "__")]
            ].astype(np.uint8)
        else:
            extra_ohe = pd.DataFrame(index=out.index)

        if fit:
            dummy_cols = list(leaf_ohe.columns)
            extra_cols = list(extra_ohe.columns)

            hist_cols = list(carried_standard_cols)
            hist_cols.append(actual_col)

            for col in dummy_cols:
                hist_cols.append(col)
                hist_cols.append(f"hist_actual_x_{col}")

            # extra defer-tuple history columns
            hist_cols.extend(extra_cols)

            self.history_feature_columns_by_stage_[stage_idx] = hist_cols
        else:
            hist_cols = self.history_feature_columns_by_stage_.get(stage_idx, [])

            dummy_cols = [
                c for c in hist_cols
                if c.startswith(label_prefix + "_")
                and not c.startswith("hist_actual_x_")
            ]

            extra_cols = [
                c for c in hist_cols
                if c.startswith(extra_prefix + "__")
            ]

            # create missing training-time dummy columns as zeros
            # and drop unseen prediction-time dummy columns.
            leaf_ohe = leaf_ohe.reindex(columns=dummy_cols, fill_value=0).astype(np.uint8)
            extra_ohe = extra_ohe.reindex(columns=extra_cols, fill_value=0).astype(np.uint8)

        # for col in dummy_cols:
        #     out[col] = leaf_ohe[col].to_numpy(dtype=np.uint8)
        #     out[f"hist_actual_x_{col}"] = (
        #         out[actual_col].to_numpy(dtype=np.uint8)
        #         * out[col].to_numpy(dtype=np.uint8)
        #     )

        # for col in list(extra_ohe.columns):
        #     out[col] = extra_ohe[col].to_numpy(dtype=np.uint8)

        actual_arr = out[actual_col].to_numpy(
            dtype=np.uint8,
            copy=False,
        )

        new_cols = {}

        for col in dummy_cols:
            col_arr = leaf_ohe[col].to_numpy(
                dtype=np.uint8,
                copy=False,
            )

            new_cols[col] = col_arr
            new_cols[f"hist_actual_x_{col}"] = (
                actual_arr * col_arr
            )

        for col in extra_ohe.columns:
            new_cols[col] = extra_ohe[col].to_numpy(
                dtype=np.uint8,
                copy=False,
            )

        if new_cols:
            out = pd.concat(
                [
                    out,
                    pd.DataFrame(new_cols, index=out.index),
                ],
                axis=1,
            )

        if fit:
            return out

        # base_cols = list(X_df.columns)

        # for col in hist_cols:
        #     if col not in out.columns:
        #         out[col] = np.uint8(0)

        # return out[base_cols + hist_cols]

        base_cols = list(X_df.columns)

        return out.reindex(
            columns=base_cols + hist_cols,
            fill_value=np.uint8(0),
        )

    def _mask_reaching_stage_from_cache(
        self,
        stage_idx: int,
        reaching_masks: Dict[int, np.ndarray],
        n: int,
    ) -> np.ndarray:
        if stage_idx == 0:
            return np.ones(n, dtype=bool)

        if stage_idx not in reaching_masks:
            raise RuntimeError(
                f"Internal history bug: reaching mask for stage {stage_idx} "
                f"has not been computed yet."
            )

        return np.asarray(reaching_masks[stage_idx], dtype=bool)

    def _stage_feature_frame_for_stage_model(
        self,
        X_df: pd.DataFrame,
        stage_idx: int,
    ) -> pd.DataFrame:   
        X_base = _to_dataframe(X_df)

        if self.add_history and not self.same_binarization and stage_idx > 0:
            return self._feature_frame_for_stage(X_base, stage_idx)

        return X_base
        
    def _feature_frame_for_stage(
        self,
        X_df: pd.DataFrame,
        stage_idx: int,
    ) -> pd.DataFrame:
        # prediction-time feature frame for stage_idx.
        # stage j was trained on raw X plus the history block for stage j.
        # this method builds earlier reaching masks iteratively, not recursively.

        X_base = _to_dataframe(X_df)

        if not self.add_history or stage_idx <= 0:
            return X_base

        n = len(X_base)

        # reaching_masks[j] = samples that reach stage j
        reaching_masks: Dict[int, np.ndarray] = {
            0: np.ones(n, dtype=bool),
        }

        active = np.arange(n, dtype=np.int64)

        # build reaching masks for stages 1, ..., stage_idx.
        for j in range(stage_idx):
            if active.size == 0:
                reaching_masks[j + 1] = np.zeros(n, dtype=bool)
                continue

            if self.add_history and not self.same_binarization and j > 0:
                reached_j = self._mask_reaching_stage_from_cache(j, reaching_masks, n)
                X_j_full = self._append_history_features_for_stage(
                    X_base,
                    j,
                    reached_j,
                    fit=False,
                )
            else:
                X_j_full = X_base

            X_j_active = X_j_full.iloc[active]

            if self.task_ == "linear_regression":
                Xb_j_active = np.zeros((len(X_j_active), 0), dtype=np.uint8)
            elif self.same_binarization:
                Xb_j_active = self.encoder_.transform(X_j_active).astype(np.uint8, copy=False)
            else:
                Xb_j_active = self.stage_encoders_[j].transform(X_j_active).astype(
                    np.uint8,
                    copy=False,
                )

            raw_j = self._raw_stage_predict(self.stages_[j], X_j_active, Xb_j_active)
            is_def_j = self._is_defer_prediction(raw_j)

            active = active[is_def_j]

            reached_next = np.zeros(n, dtype=bool)
            reached_next[active] = True
            reaching_masks[j + 1] = reached_next

        reached_stage = self._mask_reaching_stage_from_cache(stage_idx, reaching_masks, n)

        return self._append_history_features_for_stage(
            X_base,
            stage_idx,
            reached_stage,
            fit=False,
        )

    def _stage_design_matrices_for_predict(
        self,
        X_df: pd.DataFrame,
        stage_idx: int,
        active: np.ndarray,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        # return the dataframe and binarized matrix expected by stage_idx.

        X_stage_full = self._stage_feature_frame_for_stage_model(X_df, stage_idx)

        X_active = X_stage_full.iloc[active]

        if self.task_ == "linear_regression":
            Xb_active = np.zeros((len(X_active), 0), dtype=np.uint8)
        elif self.same_binarization:
            Xb_active = self.encoder_.transform(X_active).astype(np.uint8, copy=False)
        else:
            Xb_active = self.stage_encoders_[stage_idx].transform(X_active).astype(
                np.uint8,
                copy=False,
            )

        return X_active, Xb_active

    def _feature_frame_for_black_box_model(
        self,
        X_df: pd.DataFrame,
        history_stage_idx: Optional[int],
    ) -> pd.DataFrame:
        if self.add_history and history_stage_idx is not None:
            return self._feature_frame_for_stage(X_df, history_stage_idx)
        return _to_dataframe(X_df)

    def _mask_reaching_stage(
        self,
        stage_idx: int,
        X_df: pd.DataFrame,
    ) -> np.ndarray:
        # returns mask of samples that reach stage_idx.
        # stage_idx = 0 means all samples reach the first stage.
        # stage_idx = len(self.stages_) means samples deferred by all stages.

        X_base = _to_dataframe(X_df)
        n = len(X_base)

        if stage_idx < 0 or stage_idx > len(self.stages_):
            raise ValueError("stage_idx out of range.")

        active = np.arange(n, dtype=np.int64)

        for j in range(stage_idx):
            if active.size == 0:
                break

            X_j_full = self._stage_feature_frame_for_stage_model(X_base, j)

            X_j_active = X_j_full.iloc[active]

            if self.task_ == "linear_regression":
                Xb_j_active = np.zeros((len(X_j_active), 0), dtype=np.uint8)
            elif self.same_binarization:
                Xb_j_active = self.encoder_.transform(X_j_active).astype(np.uint8, copy=False)
            else:
                Xb_j_active = self.stage_encoders_[j].transform(X_j_active).astype(
                    np.uint8,
                    copy=False,
                )

            raw_j = self._raw_stage_predict(self.stages_[j], X_j_active, Xb_j_active)
            is_def_j = self._is_defer_prediction(raw_j)

            active = active[is_def_j]

        reached = np.zeros(n, dtype=bool)
        reached[active] = True
        return reached

    def _predict_suffix_from_stage(
        self,
        start_stage_idx: int,
        X_df: pd.DataFrame,
    ) -> np.ndarray:
        # predict using stages start_stage_idx, ..., K-1, then final black box. if idx is 2, we skip 0 and 1.
        # predicts on every sample in X_df
        n = len(X_df)

        if start_stage_idx < 0 or start_stage_idx > len(self.stages_):
            raise ValueError("start_stage_idx out of range.")

        final = np.zeros(
            n,
            dtype=np.int64 if self.task_ == "classification" else np.float64,
        )
        active = np.arange(n, dtype=np.int64)

        for j in range(start_stage_idx, len(self.stages_)):
            if active.size == 0:
                break

            X_active, Xb_active = self._stage_design_matrices_for_predict(
                X_df,
                j,
                active,
            )

            raw = self._raw_stage_predict(self.stages_[j], X_active, Xb_active)

            is_def = self._is_defer_prediction(raw)
            handled = ~is_def

            if np.any(handled):
                handled_idx = active[handled]
                final[handled_idx] = self._clean_stage_predictions(raw[handled])

            active = active[is_def]

        if active.size > 0:
            if isinstance(self.final_black_box_, _FixedPredictionBlackBox):
                bb_full = self._predict_black_box(self.final_black_box_, X_df)
                final[active] = self._clean_stage_predictions(bb_full[active])

            else:
                if self.add_history and self.final_black_box_history_stage_idx_ is not None:
                    X_final_full = self._feature_frame_for_stage(
                        X_df,
                        self.final_black_box_history_stage_idx_,
                    )
                    X_final_active = X_final_full.iloc[active]
                else:
                    X_final_active = X_df.iloc[active]

                bb_pred_active = self._predict_black_box(
                    self.final_black_box_,
                    X_final_active,
                )
                final[active] = self._clean_stage_predictions(bb_pred_active)

        return final

    def _compute_backfit_weights_for_stage(
        self,
        X_df: pd.DataFrame,
        reached_mask: np.ndarray,
        feasible_states: List[IntervalState],
    ) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
        # weight 1 on reached_mask, based on deferred distance otherwise
   
        reached_mask = np.asarray(reached_mask, dtype=bool)

        if int(reached_mask.sum()) <= 0:
            raise ValueError("No samples reach this stage.")

        # if every sample reaches the stage, just use uniform weights.
        if int(reached_mask.sum()) == len(reached_mask):
            return np.ones(len(reached_mask), dtype=np.float64), None

        if getattr(self, "set_weights_using_bb_leaves", False):
            if self.projection:
                raise ValueError(
                    "projection=True is not compatible with "
                    "set_weights_using_bb_leaves=True."
                )

            weights, _, X_df_proj = _compute_stage_weights_from_black_box_leaf_similarity(
                X_df=X_df,
                deferred_mask=reached_mask,
                bb_model=self.final_black_box_,
                c=self.c,
                gamma=self.gamma,
            )
            return weights, X_df_proj

        weights, _, X_df_proj = _compute_stage_weights_soft_deferred_region_scaled_nondeferred(
            X_df=X_df,
            deferred_mask=reached_mask,
            feasible_states=feasible_states,
            c=self.c,
            gamma=self.gamma,
            quantile_maps=self.quantile_maps_,
            return_projected_df=bool(self.projection),
        )
        return weights, X_df_proj

    def _lambda_for_backfit_stage(
        self,
        weights: np.ndarray,
        reached_mask: np.ndarray,
        n: int,
    ) -> float:
        if self.defer_lambda_strength == 0:
            return self.lambda_leaf
        if self.defer_lambda_strength == 1:
            return self.lambda_leaf * float(reached_mask.sum()) / float(n)
        if self.defer_lambda_strength == 2:
            return self.lambda_leaf * float(weights.sum()) / float(n)
        raise ValueError("defer_lambda_strength must be in {0, 1, 2}")

    def _weighted_one_stage_defer_objective(
        self,
        *,
        stage,
        encoder,
        X_df_eval: pd.DataFrame,
        y: np.ndarray,
        suffix_pred: np.ndarray,
        weights: np.ndarray,
        lambda_leaf: float,
    ) -> float:
        if self.task_ == "linear_regression":
            Xb_eval = np.zeros((len(X_df_eval), 0), dtype=np.uint8)
        else:
            Xb_eval = encoder.transform(X_df_eval).astype(np.uint8, copy=False)

        raw = self._raw_stage_predict(stage, X_df_eval, Xb_eval)

        is_def = self._is_defer_prediction(raw)

        if self.task_ == "classification":
            pred = self._clean_stage_predictions(raw.copy())
            suffix_clean = self._clean_stage_predictions(suffix_pred)
            pred[is_def] = suffix_clean[is_def]
            y_arr = np.asarray(y, dtype=np.int64)
            loss = (pred != y_arr).astype(np.float64)
        else:
            pred = np.asarray(raw, dtype=np.float64).copy()
            suffix_clean = np.asarray(suffix_pred, dtype=np.float64)
            pred[is_def] = suffix_clean[is_def]
            y_arr = np.asarray(y, dtype=np.float64)
            loss = (pred - y_arr) ** 2

        defer_penalty = is_def.astype(np.float64) * float(self.eta_defer)
        leaves = self._stage_leaf_count(stage)

        return (
            float(lambda_leaf) * float(leaves - 1)
            + float(np.sum(weights * (loss + defer_penalty)))
        )

    def _refit_one_backfit_stage(
        self,
        stage_idx: int,
        X_df: pd.DataFrame,
        y: np.ndarray,
        *,
        seed: int,
    ) -> bool:
        # throw away existing T_j and retrain it with fallback = current suffix T_{j+1:K};B.
        # accept the replacement only if it improves the weighted one-stage defer objective.
        n = len(y)

        reached_mask = self._mask_reaching_stage(stage_idx, X_df)
        if int(reached_mask.sum()) <= 0:
            return False

        feasible_states = self._get_feasible_deferred_states_through_stage(stage_idx)

        weights, X_df_proj = self._compute_backfit_weights_for_stage(
            X_df=X_df,
            reached_mask=reached_mask,
            feasible_states=feasible_states,
        )

        X_df_stage_for_tg = X_df_proj if self.projection and X_df_proj is not None else X_df

        # fallback for T_j is the current suffix T_{j+1:K};B.
        suffix_pred = self._predict_suffix_from_stage(
            stage_idx + 1,
            X_df_stage_for_tg,
        )

        lambda_t = self._lambda_for_backfit_stage(
            weights=weights,
            reached_mask=reached_mask,
            n=n,
        )

        old_stage = self.stages_[stage_idx]

        # lnear regression / CLARITree uses raw X, not threshold-binarized X.
        # so old_enc and new enc_t should be None.
        if self.task_ == "linear_regression":
            old_enc = None
        else:
            old_enc = self.encoder_ if self.same_binarization else self.stage_encoders_[stage_idx]

        old_obj = self._weighted_one_stage_defer_objective(
            stage=old_stage,
            encoder=old_enc,
            X_df_eval=X_df_stage_for_tg,
            y=y,
            suffix_pred=suffix_pred,
            weights=weights,
            lambda_leaf=lambda_t,
        )

        if self.task_ == "linear_regression":
            enc_t = None
            Xb_stage = np.zeros((len(X_df_stage_for_tg), 0), dtype=np.uint8)

        else:
            if self.same_binarization:
                enc_t = self.encoder_
                Xb_stage = enc_t.transform(X_df_stage_for_tg).astype(np.uint8, copy=False)
            else:
                enc_t, Xb_stage = self._fit_threshold_guesser(
                    X_df_stage_for_tg,
                    y,
                    sample_weight=weights,
                    seed=seed,
                )

            if self.drop_no_signal_columns:
                Xb_stage, _ = self._zero_out_constant_binarized_columns_from_states(
                    Xb=Xb_stage,
                    encoder=enc_t,
                    feasible_states=feasible_states,
                )

        new_stage = self._fit_defer_stage(
            X_df_stage=X_df_stage_for_tg,
            Xb_stage=Xb_stage,
            y=y,
            bb_pred=suffix_pred,
            lambda_leaf=lambda_t,
            eta_defer=self.eta_defer,
            depth_budget=self.depth_budget,
            lookahead_k=self.lookahead_k,
            sample_weights=weights,
            seed=seed,
        )

        new_obj = self._weighted_one_stage_defer_objective(
            stage=new_stage,
            encoder=enc_t,
            X_df_eval=X_df_stage_for_tg,
            y=y,
            suffix_pred=suffix_pred,
            weights=weights,
            lambda_leaf=lambda_t,
        )

        tol = 1e-12

        if new_obj <= old_obj + tol:
            self.stages_[stage_idx] = new_stage

            if self.task_ == "linear_regression":
                if not self.same_binarization:
                    self.stage_encoders_[stage_idx] = None
            elif not self.same_binarization:
                self.stage_encoders_[stage_idx] = enc_t

            return True

        return False    

    def _refit_final_black_box_after_backfit(
        self,
        X_df: pd.DataFrame,
        y: np.ndarray,
        *,
        seed: int,
    ) -> None:
        # refit the final fallback model after a backfitting pass, but only if self.retrain_black_box is on.
        
        if not self.retrain_black_box:
            return

        if len(self.stages_) == 0:
            return

        final_deferred = self._mask_reaching_stage(len(self.stages_), X_df)
        self.final_deferred_mask_ = final_deferred.copy()

        if int(final_deferred.sum()) <= 0:
            return

        feasible_states = self._get_feasible_deferred_states_through_stage(len(self.stages_))

        if getattr(self, "set_weights_using_bb_leaves", False):
            if self.projection:
                raise ValueError(
                    "projection=True is not compatible with "
                    "set_weights_using_bb_leaves=True."
                )

            weights, _, X_df_proj = _compute_stage_weights_from_black_box_leaf_similarity(
                X_df=X_df,
                deferred_mask=final_deferred,
                bb_model=self.final_black_box_,
                c=self.c,
                gamma=self.gamma,
            )
        else:
            weights, _, X_df_proj = _compute_stage_weights_soft_deferred_region_scaled_nondeferred(
                X_df=X_df,
                deferred_mask=final_deferred,
                feasible_states=feasible_states,
                c=self.c,
                gamma=self.gamma,
                quantile_maps=self.quantile_maps_,
                return_projected_df=bool(self.projection),
            )

        X_df_for_bb = X_df_proj if self.projection and X_df_proj is not None else X_df

        self.final_black_box_ = self._fit_black_box(
            X_df_for_bb,
            y,
            sample_weight=weights,
            seed=seed,
        )

    def _backfit_backward_pass(
        self,
        X_df: pd.DataFrame,
        y: np.ndarray,
        *,
        start_stage_idx: Optional[int] = None,
        end_stage_idx: int = 0,
        seed_base: int = 0,
        refit_black_box_at_beginning: bool = True,
        refit_black_box_at_end: bool = True,
    ) -> None:
        # Refit stages in reverse order.
        # by default: optionally refit B, then K-1, K-2, ..., 0, then optionally refit B again.
        
        K = len(self.stages_)
        if K == 0:
            return

        if refit_black_box_at_beginning:
            self._refit_final_black_box_after_backfit(
                X_df,
                y,
                seed=seed_base + 800000,
            )

        if start_stage_idx is None:
            start_stage_idx = K - 1

        start_stage_idx = min(int(start_stage_idx), K - 1)
        end_stage_idx = max(int(end_stage_idx), 0)

        if start_stage_idx < end_stage_idx:
            return

        for j in range(start_stage_idx, end_stage_idx - 1, -1):
            self._refit_one_backfit_stage(
                j,
                X_df,
                y,
                seed=seed_base + 1000 + j,
            )

        if refit_black_box_at_end:
            self._refit_final_black_box_after_backfit(
                X_df,
                y,
                seed=seed_base + 900000,
            )

    def _backfit_forward_pass(
        self,
        X_df: pd.DataFrame,
        y: np.ndarray,
        *,
        start_stage_idx: int = 0,
        end_stage_idx: Optional[int] = None,
        seed_base: int = 0,
        refit_black_box_at_beginning: bool = True,
        refit_black_box_at_end: bool = True,
    ) -> None:
        # refit stages in forward order.
        # by default: optionally refit B, then 0, 1, ..., K-1, then optionally refit B again.
        
        K = len(self.stages_)
        if K == 0:
            return

        if refit_black_box_at_beginning:
            self._refit_final_black_box_after_backfit(
                X_df,
                y,
                seed=seed_base + 850000,
            )

        if end_stage_idx is None:
            end_stage_idx = K - 1

        start_stage_idx = max(int(start_stage_idx), 0)
        end_stage_idx = min(int(end_stage_idx), K - 1)

        if start_stage_idx > end_stage_idx:
            return

        for j in range(start_stage_idx, end_stage_idx + 1):
            self._refit_one_backfit_stage(
                j,
                X_df,
                y,
                seed=seed_base + 2000 + j,
            )

        if refit_black_box_at_end:
            self._refit_final_black_box_after_backfit(
                X_df,
                y,
                seed=seed_base + 950000,
            )

    def _run_posthoc_backfitting(
        self,
        X_df: pd.DataFrame,
        y: np.ndarray,
    ) -> None:
        #optional post-training backfitting.
        # none        : no backfitting
        # backward    : repeated backward sweeps
        # forward     : repeated forward sweeps
        # alternating : repeated backward/forward sweeps

        # black-box refit policy:
        # backward    : refit black box at beginning of each backward pass, not at end
        # forward     : refit black box at end of each forward pass, not at beginning

        mode = str(getattr(self, "backfit_mode", "none")).lower()
        sweeps = int(getattr(self, "backfit_sweeps", 1))

        if mode == "none" or sweeps <= 0:
            return

        K = len(self.stages_)
        if K <= 0:
            return

        if mode in {"backward", "backwards"}:
            for s in range(sweeps):
                seed_base = self.random_state + 1_000_000 * (s + 1)

                self._backfit_backward_pass(
                    X_df,
                    y,
                    start_stage_idx=K - 1,
                    end_stage_idx=0,
                    seed_base=seed_base,
                    refit_black_box_at_beginning=True,
                    refit_black_box_at_end=False,
                )

        elif mode in {"forward", "forwards"}:
            for s in range(sweeps):
                seed_base = self.random_state + 1_000_000 * (s + 1)

                self._backfit_forward_pass(
                    X_df,
                    y,
                    start_stage_idx=0,
                    end_stage_idx=K - 1,
                    seed_base=seed_base,
                    refit_black_box_at_beginning=False,
                    refit_black_box_at_end=True,
                )

        elif mode in {"alternating", "alternate"}:
            refit_bb = bool(getattr(self, "retrain_black_box", False))

            for s in range(sweeps):
                seed_base = self.random_state + 1_000_000 * (s + 1)

                # backward pass:
                # if retrain_black_box=True, this is: refit B, then K-1, K-2, ..., 0
                # if retrain_black_box=False, the beginning refit call is harmless/no-op, K-1, K-2, ..., 0
                self._backfit_backward_pass(
                    X_df,
                    y,
                    start_stage_idx=K - 1,
                    end_stage_idx=0,
                    seed_base=seed_base,
                    refit_black_box_at_beginning=True,
                    refit_black_box_at_end=False,
                )

                # forward pass:
                # after backward ends at stage 0, always skip stage 0.
                # if retrain_black_box=True: go 1, 2, ..., K-1 (K-1 is the last tree)
                # because the next backward pass will first refresh B, so K-1's
                # fallback may change.
                # if retrain_black_box=False: go 1, 2, ..., K-2
                # because B is fixed, and the immediately preceding backward pass already refit K-1 and didn't go any further
                forward_end = K - 1 if refit_bb else K - 2

                if 1 <= forward_end:
                    self._backfit_forward_pass(
                        X_df,
                        y,
                        start_stage_idx=1,
                        end_stage_idx=forward_end,
                        seed_base=seed_base + 100_000,
                        refit_black_box_at_beginning=False,
                        refit_black_box_at_end=False,
                    )
        else:
            raise ValueError(f"Unknown backfit_mode={mode!r}")

        self.final_deferred_mask_ = self._mask_reaching_stage(len(self.stages_), X_df)

    def _fit_top_down(
        self,
        X_df: pd.DataFrame,
        y: np.ndarray,
        Xb: np.ndarray,
        bb_pred_override: Optional[np.ndarray] = None,
    ) -> None:
        if self.c is None:
            raise ValueError("For training_method='top_down', c must be provided.")
        if self.defer_lambda_strength is None:
            raise ValueError("For training_method='top_down', defer_lambda_strength must be provided.")

        n = len(y)
        self.quantile_maps_ = _compute_empirical_quantile_maps(X_df)
        if bb_pred_override is not None:
            bb_pred = np.asarray(bb_pred_override).copy()
            bb_model = _FixedPredictionBlackBox(predictions=bb_pred.copy())
            root_bb_model = bb_model
        else:
            bb_model = self._fit_black_box(X_df, y, sample_weight=None, seed=self.random_state)
            root_bb_model = bb_model

            if getattr(self, "out_of_fold_predictions", False):
                bb_pred = self._out_of_fold_black_box_predictions(
                    X_df=X_df,
                    y=y,
                    sample_weight=None,
                    seed=self.random_state,
                    stratify_mask=None,
                    n_splits=self.out_of_fold_n_splits,
                )
            else:
                bb_pred = self._predict_black_box(bb_model, X_df)

        # online formula state for uncompressed single-tree leaf count
        online_LW = 0
        online_prod_d = 1 # to change, don't worry
        online_lW = self._formula_black_box_leaf_count(bb_model)

        if self.task_ == "linear_regression":
            enc1 = None
            Xb_stage = np.zeros((len(X_df), 0), dtype=np.uint8)
        elif self.same_binarization:
            enc1 = self.encoder_
            Xb_stage = Xb
        else:
            enc1, Xb_stage = self._fit_threshold_guesser(
                X_df, y, sample_weight=None, seed=self.random_state
            )

        self.stage_encoders_.append(enc1)
        # stage1 = WeightedSeparable()
        # stage1.fit(
        #     Xb_stage,
        #     y,
        #     bb_pred,
        #     lambda_leaf=self.lambda_leaf,
        #     eta_defer=self.eta_defer,
        #     depth_budget=self.depth_budget,
        #     lookahead_k=self.lookahead_k,
        #     sample_weights=None,
        # )
        # self.stages_.append(stage1)

        # current_deferred = self._eventual_mask_single(stage1, Xb_stage)
        stage1 = self._fit_defer_stage(
            X_df_stage=X_df,
            Xb_stage=Xb_stage,
            y=y,
            bb_pred=bb_pred,
            lambda_leaf=self.lambda_leaf,
            eta_defer=self.eta_defer,
            depth_budget=self.depth_budget,
            lookahead_k=self.lookahead_k,
            sample_weights=None,
            seed=self.random_state,
        )
        self.stages_.append(stage1)

        current_deferred = self._eventual_mask_single(stage1, X_df, Xb_stage)
        
        final_bb_model = bb_model
        max_total_stages = 10**9 if self.max_stages == 0 else self.max_stages
        hit_iteration_limit = False

        if self.max_total_stage_leaves is not None:
            total_leaves = self._raw_total_stage_leaves_so_far()
            if isinstance(final_bb_model, _PseudoWeightedSEPBlackBox):
                total_leaves += self._raw_leaf_count_weightedsep_stage(final_bb_model.sep)
            if total_leaves > self.max_total_stage_leaves:
                self.stop_reason_ = "leaf_budget_exceeded"
                max_total_stages = 0 # skip the for loop, we just need to update the black box

        if self.max_expanded_leaves is not None:
            l_stage1, d_stage1 = self._stage_leaf_stats(stage1)

            # current chain is just stage1 -> current black box
            online_LW = (l_stage1 - d_stage1) + d_stage1 * online_lW # l_stage1 if no pseudo black box, otherwise exactly as it should be.
            online_prod_d = d_stage1

            if online_LW > self.max_expanded_leaves:
                self.stop_reason_ = "expanded_leaf_budget_exceeded"
                self.final_deferred_mask_ = current_deferred.copy()
                self.final_black_box_ = final_bb_model
                max_total_stages = 0  # skip the loop, same style as raw leaf budget
        
        if self.max_leaves_per_tree is not None:
            # stage1_leaves = self._raw_leaf_count_weightedsep_stage(stage1)
            stage1_leaves = self._raw_leaf_count_stage(stage1)
            if stage1_leaves > self.max_leaves_per_tree:
                self.stop_reason_ = "per_tree_leaf_budget_exceeded"
                # current_deferred = self._eventual_mask_single(stage1, Xb_stage)
                final_bb_model = bb_model
                max_total_stages = 0

        X_df_stage_for_tg = X_df

        bb_model_history_stage_idx = None
        final_black_box_history_stage_idx = None

        for stage_idx in range(1, max_total_stages):
            if int(current_deferred.sum()) <= 0:
                self.stop_reason_ = "empty_deferred_set"
                break

            feasible_states = self._get_feasible_deferred_states_through_stage(len(self.stages_)) # on the first iteration of the loop, we do the stage before the loop, which is correct
    
            # weights, _, X_df_proj = _compute_stage_weights_soft_deferred_region_scaled_nondeferred(
            #     X_df=X_df,
            #     deferred_mask=current_deferred,
            #     feasible_states=feasible_states,
            #     c=self.c,
            #     gamma=self.gamma,
            #     quantile_maps=self.quantile_maps_,
            #     return_projected_df=bool(self.projection),
            # )

            # X_df_stage_for_tg = X_df_proj if self.projection else X_df

            if getattr(self, "set_weights_using_bb_leaves", False):
                if self.projection:
                    raise ValueError(
                        "projection=True is not currently compatible with "
                        "set_weights_using_bb_leaves=True."
                    )

                X_df_for_leaf_weights = self._feature_frame_for_black_box_model(
                    X_df,
                    bb_model_history_stage_idx,
                )

                weights, _, X_df_proj = _compute_stage_weights_from_black_box_leaf_similarity(
                    X_df=X_df_for_leaf_weights,
                    deferred_mask=current_deferred,
                    bb_model=bb_model,
                    c=self.c,
                    gamma=self.gamma,
                )

            else:
                distance_states = _strip_history_bounds_from_states(feasible_states)

                weights, _, X_df_proj = _compute_stage_weights_soft_deferred_region_scaled_nondeferred(
                    X_df=X_df,
                    deferred_mask=current_deferred,
                    feasible_states=distance_states,
                    c=self.c,
                    gamma=self.gamma,
                    quantile_maps=self.quantile_maps_,
                    return_projected_df=bool(self.projection),
                )

            # X_df_stage_for_tg = X_df_proj if self.projection else X_df
            X_df_stage_base = X_df_proj if self.projection else X_df

            # history-augmented dataframe. For stage_idx=1, this adds history from stage 0.
            X_df_stage_with_history = self._append_history_features_for_stage(
                X_df_stage_base,
                stage_idx,
                current_deferred,
                fit=True,
            )

            # black box sees history if add_history=True.
            X_df_for_bb = X_df_stage_with_history if self.add_history else X_df_stage_base

            # bb_model = self._fit_black_box(X_df_stage_for_tg, y, sample_weight=weights, seed=self.random_state + stage_idx)
            # # bb_pred = self._predict_black_box(bb_model, X_df_stage_for_tg)
            # if getattr(self, "out_of_fold_predictions", False):
            #     bb_pred = self._out_of_fold_black_box_predictions(
            #         X_df=X_df_stage_for_tg,
            #         y=y,
            #         sample_weight=weights,
            #         seed=self.random_state + stage_idx,
            #         stratify_mask=current_deferred,
            #         n_splits=self.out_of_fold_n_splits,
            #     )
            # else:
            #     bb_pred = self._predict_black_box(bb_model, X_df_stage_for_tg)

            # tree/threshold-guesser sees history only when same_binarization=False.
            if self.add_history and not self.same_binarization:
                X_df_stage_for_tg = X_df_stage_with_history
            else:
                X_df_stage_for_tg = X_df_stage_base

            if self.retrain_black_box:
                bb_model = self._fit_black_box(
                    X_df_for_bb,
                    y,
                    sample_weight=weights,
                    seed=self.random_state + stage_idx,
                )

                # if add_history=True, this black box was trained on
                # _feature_frame_for_stage(X, stage_idx).
                # otherwise it was trained on raw/projected X.
                bb_model_history_stage_idx = stage_idx if self.add_history else None

                if getattr(self, "out_of_fold_predictions", False):
                    bb_pred = self._out_of_fold_black_box_predictions(
                        X_df=X_df_for_bb,
                        y=y,
                        sample_weight=weights,
                        seed=self.random_state + stage_idx,
                        stratify_mask=current_deferred,
                        n_splits=self.out_of_fold_n_splits,
                    )
                else:
                    bb_pred = self._predict_black_box(bb_model, X_df_for_bb)

            else:
                bb_model = root_bb_model
                bb_model_history_stage_idx = None

                # root black box was trained on raw X, not history X.
                bb_pred = self._predict_black_box(root_bb_model, X_df_stage_base)

            defer_frac = float(current_deferred.mean())
            if self.defer_lambda_strength == 0:
                lambda_t = self.lambda_leaf
            elif self.defer_lambda_strength == 1:
                lambda_t = self.lambda_leaf * float(current_deferred.sum()) / float(n)
            elif self.defer_lambda_strength == 2:
                lambda_t = self.lambda_leaf * float(weights.sum()) / float(n)
            else:
                raise ValueError("defer_lambda_strength must be in {0, 1, 2}")

            if self.task_ == "linear_regression":
                # CLARITree regression stages consume raw X_df_stage_for_tg.
                enc_t = None
                Xb_stage = np.zeros((len(X_df_stage_for_tg), 0), dtype=np.uint8)

            elif self.same_binarization:
                enc_t = self.encoder_
                Xb_stage = enc_t.transform(X_df_stage_for_tg).astype(np.uint8, copy=False)

            else:
                enc_t, Xb_stage = self._fit_threshold_guesser(
                    X_df_stage_for_tg,
                    y,
                    sample_weight=weights,
                    seed=self.random_state + stage_idx,
                )

                if self.drop_no_signal_columns:
                    Xb_stage, zero_mask = self._zero_out_constant_binarized_columns_from_states(
                        Xb=Xb_stage,
                        encoder=enc_t,
                        feasible_states=feasible_states,
                    )

            # stage = WeightedSeparable()
            # stage.fit(
            #     Xb_stage,
            #     y,
            #     bb_pred,
            #     lambda_leaf=lambda_t,
            #     eta_defer=self.eta_defer,
            #     depth_budget=self.depth_budget,
            #     lookahead_k=self.lookahead_k,
            #     sample_weights=weights,
            # )
            stage = self._fit_defer_stage(
                X_df_stage=X_df_stage_for_tg,
                Xb_stage=Xb_stage,
                y=y,
                bb_pred=bb_pred,
                lambda_leaf=lambda_t,
                eta_defer=self.eta_defer,
                depth_budget=self.depth_budget,
                lookahead_k=self.lookahead_k,
                sample_weights=weights,
                seed=self.random_state + stage_idx,
            )

            stage_deferred = self._eventual_mask_single(stage, X_df_stage_for_tg, Xb_stage)
            next_deferred = current_deferred & stage_deferred

            if int(next_deferred.sum()) == int(current_deferred.sum()):
                current_deferred = next_deferred
                self.stop_reason_ = "no_shrinkage"
                break

            self.stages_.append(stage)
            final_bb_model = bb_model
            final_black_box_history_stage_idx = bb_model_history_stage_idx
            self.stage_encoders_.append(enc_t)

            if self.max_total_stage_leaves is not None:
                total_leaves = self._raw_total_stage_leaves_so_far()
                if isinstance(final_bb_model, _PseudoWeightedSEPBlackBox):
                    total_leaves += self._raw_leaf_count_weightedsep_stage(final_bb_model.sep)
                if total_leaves > self.max_total_stage_leaves:
                    # current_deferred = self._eventual_mask_single(stage, Xb_stage)
                    current_deferred = next_deferred
                    self.stop_reason_ = "leaf_budget_exceeded"
                    break
                    
            if self.max_expanded_leaves is not None:
                l_stage, d_stage = self._stage_leaf_stats(stage)

                # compute new black box leaf count
                new_lW = self._formula_black_box_leaf_count(final_bb_model)

                # online update: watch out for the difference in lW and LW. l being leaf, and L being the online subchain answer.
                delta = online_prod_d * ((l_stage - d_stage) + d_stage * new_lW - online_lW)
                online_LW = online_LW + delta

                # update product and lW
                online_prod_d *= d_stage
                online_lW = new_lW

                if online_LW > self.max_expanded_leaves:
                    # current_deferred = self._eventual_mask_single(stage, Xb_stage)
                    current_deferred = next_deferred
                    self.stop_reason_ = "expanded_leaf_budget_exceeded"
                    break
            
            if self.max_leaves_per_tree is not None:
                #stage_leaves = self._raw_leaf_count_weightedsep_stage(stage)
                stage_leaves = self._raw_leaf_count_stage(stage)
                if stage_leaves > self.max_leaves_per_tree:
                    # current_deferred = self._eventual_mask_single(stage, Xb_stage)
                    current_deferred = next_deferred
                    self.stop_reason_ = "per_tree_leaf_budget_exceeded"
                    break


            # stage_deferred = self._eventual_mask_single(stage, Xb_stage)
   
            current_deferred = next_deferred

            if stage_idx == max_total_stages - 1:
                hit_iteration_limit = True
                self.stop_reason_ = "iteration_limit"

        else:
            hit_iteration_limit = True
            self.stop_reason_ = "iteration_limit"

        if self.stop_reason_ is None:
            if int(current_deferred.sum()) <= 0:
                self.stop_reason_ = "empty_deferred_set"
            else:
                self.stop_reason_ = "iteration_limit"
                hit_iteration_limit = True

        self.final_deferred_mask_ = current_deferred.copy()

        if (
            self.retrain_black_box
            and (
                hit_iteration_limit
                or self.stop_reason_ == "leaf_budget_exceeded"
                or self.stop_reason_ == "expanded_leaf_budget_exceeded"
                or self.stop_reason_ == "per_tree_leaf_budget_exceeded"
            )
            and int(current_deferred.sum()) > 0
        ):
            feasible_states = self._get_feasible_deferred_states_through_stage(len(self.stages_))
            # final_weights, _, final_X_df_proj = _compute_stage_weights_soft_deferred_region_scaled_nondeferred(
            #     X_df=X_df,
            #     deferred_mask=current_deferred,
            #     feasible_states=feasible_states,
            #     c=self.c,
            #     gamma=self.gamma,
            #     quantile_maps=self.quantile_maps_,
            #     return_projected_df=bool(self.projection),
            # )
            # final_X_df_for_bb = final_X_df_proj if self.projection else X_df
            if getattr(self, "set_weights_using_bb_leaves", False):
                if self.projection:
                    raise ValueError(
                        "projection=True is not currently compatible with "
                        "set_weights_using_bb_leaves=True."
                    )

                X_df_for_final_leaf_weights = self._feature_frame_for_black_box_model(
                    X_df,
                    final_black_box_history_stage_idx,
                )

                final_weights, _, final_X_df_proj = _compute_stage_weights_from_black_box_leaf_similarity(
                    X_df=X_df_for_final_leaf_weights,
                    deferred_mask=current_deferred,
                    bb_model=final_bb_model,
                    c=self.c,
                    gamma=self.gamma,
                )
            else:
                final_distance_states = _strip_history_bounds_from_states(feasible_states)

                final_weights, _, final_X_df_proj = _compute_stage_weights_soft_deferred_region_scaled_nondeferred(
                    X_df=X_df,
                    deferred_mask=current_deferred,
                    feasible_states=final_distance_states,
                    c=self.c,
                    gamma=self.gamma,
                    quantile_maps=self.quantile_maps_,
                    return_projected_df=bool(self.projection),
                )

            # final_X_df_for_bb = final_X_df_proj if self.projection else X_df

            # final_bb_model = self._fit_black_box(
            #     final_X_df_for_bb,
            #     y,
            #     sample_weight=final_weights,
            #     seed=self.random_state + len(self.stages_) + 100000,
            # )

            final_X_df_base = final_X_df_proj if self.projection else X_df
            final_stage_idx = len(self.stages_)

            final_X_df_with_history = self._append_history_features_for_stage(
                final_X_df_base,
                final_stage_idx,
                current_deferred,
                fit=True,
            )

            final_X_df_for_bb = (
                final_X_df_with_history
                if self.add_history
                else final_X_df_base
            )

            final_bb_model = self._fit_black_box(
                final_X_df_for_bb,
                y,
                sample_weight=final_weights,
                seed=self.random_state + len(self.stages_) + 100000,
            )

            final_black_box_history_stage_idx = final_stage_idx if self.add_history else None

        if self.add_history and self.eliminate_black_box:
            raise ValueError("add_history=True is not currently supported with eliminate_black_box=True.")

        if self.eliminate_black_box and int(current_deferred.sum()) > 0:
            feasible_states = self._get_feasible_deferred_states_through_stage(len(self.stages_))
            # elim_weights, _, elim_X_df_proj = _compute_stage_weights_soft_deferred_region_scaled_nondeferred(
            #     X_df=X_df,
            #     deferred_mask=current_deferred,
            #     feasible_states=feasible_states,
            #     c=self.c,
            #     gamma=self.gamma,
            #     quantile_maps=self.quantile_maps_,
            #     return_projected_df=bool(self.projection),
            # )
            # elim_X_df_for_bb = elim_X_df_proj if self.projection else X_df
            if getattr(self, "set_weights_using_bb_leaves", False):
                if self.projection:
                    raise ValueError(
                        "projection=True is not currently compatible with "
                        "set_weights_using_bb_leaves=True."
                    )

                elim_weights, _, elim_X_df_proj = _compute_stage_weights_from_black_box_leaf_similarity(
                    X_df=X_df,
                    deferred_mask=current_deferred,
                    bb_model=final_bb_model,
                    c=self.c,
                    gamma=self.gamma,
                )
            else:
                elim_distance_states = _strip_history_bounds_from_states(feasible_states)

                elim_weights, _, elim_X_df_proj = _compute_stage_weights_soft_deferred_region_scaled_nondeferred(
                    X_df=X_df,
                    deferred_mask=current_deferred,
                    feasible_states=elim_distance_states,
                    c=self.c,
                    gamma=self.gamma,
                    quantile_maps=self.quantile_maps_,
                    return_projected_df=bool(self.projection),
                )

            elim_X_df_for_bb = elim_X_df_proj if self.projection else X_df

            final_bb_model = self._fit_elimination_model(
                elim_X_df_for_bb,
                y,
                sample_weight=elim_weights,
                seed=self.random_state + len(self.stages_) + 200000,
            )

        self.final_black_box_ = final_bb_model
        self.final_black_box_history_stage_idx_ = final_black_box_history_stage_idx
        self._run_posthoc_backfitting(X_df, y)

    # will be depreciated: don't call
    def _fit_bottom_up(self, X_df: pd.DataFrame, y: np.ndarray, Xb: np.ndarray) -> None:
        base_model = self._fit_black_box(X_df, y, sample_weight=None, seed=self.random_state)
        r = self._predict_black_box(base_model, X_df).astype(np.uint8, copy=False)
        self.final_black_box_ = base_model

        max_total_stages = 10**9 if self.max_stages == 0 else self.max_stages
        online_LW = self._formula_black_box_leaf_count(base_model)

        for stage_idx in range(max_total_stages):
            stage = WeightedSeparable()
            stage.fit(
                Xb,
                y,
                r,
                lambda_leaf=self.lambda_leaf,
                eta_defer=self.eta_defer,
                depth_budget=self.depth_budget,
                lookahead_k=self.lookahead_k,
                sample_weights=None,
            )

            # self.stages_.append(stage)
            self.stages_.insert(0, stage) # for bottom up we must append to front

            raw = stage.predict(Xb, bb_pred=None, placeholder=self.placeholder).astype(np.uint8, copy=False)
            is_def = self._is_defer_prediction(raw)

            r_hat = raw.copy()
            r_hat[is_def] = r[is_def]
            r_hat = r_hat & np.uint8(1)

            if np.array_equal(r_hat, r):
                self.stop_reason_ = "no_prediction_change"
                break

            r = r_hat

            if stage_idx == max_total_stages - 1:
                self.stop_reason_ = "iteration_limit"

            if self.max_total_stage_leaves is not None:
                total_leaves = self._raw_total_stage_leaves_so_far()
                if isinstance(self.final_black_box_, _PseudoWeightedSEPBlackBox):
                    total_leaves += self._raw_leaf_count_weightedsep_stage(self.final_black_box_.sep)
                if total_leaves > self.max_total_stage_leaves:
                    self.stop_reason_ = "leaf_budget_exceeded"
                    break
            if self.max_expanded_leaves is not None:
                l_stage, d_stage = self._stage_leaf_stats(stage)
                online_LW = (l_stage - d_stage) + d_stage * online_LW

                if online_LW > self.max_expanded_leaves:
                    self.stop_reason_ = "expanded_leaf_budget_exceeded"
                    break

            if self.max_leaves_per_tree is not None:
                stage_leaves = self._raw_leaf_count_weightedsep_stage(stage)
                if stage_leaves > self.max_leaves_per_tree:
                    self.stop_reason_ = "per_tree_leaf_budget_exceeded"
                    break

        if self.stop_reason_ is None:
            self.stop_reason_ = "iteration_limit"

        if self.stages_:
            active = np.ones(len(y), dtype=bool)
            for stage in self.stages_:
                stage_def = self._eventual_mask_single(stage, Xb)
                active &= stage_def
            self.final_deferred_mask_ = active
        else:
            self.final_deferred_mask_ = np.zeros(len(y), dtype=bool)

# stopped review here on 3/28 11:28pm

class _MDTCompressionMixin:
    def _check_is_fit(self):
        if self.final_black_box_ is None:
            raise RuntimeError("Model is not fit.")
        if self.same_binarization:
            if self.encoder_ is None:
                raise RuntimeError("Model is not fit.")
        else:
            if len(self.stage_encoders_) != len(self.stages_):
                raise RuntimeError("Stage encoders are missing.")

    def _coerce_X_for_predict(self, X) -> pd.DataFrame:
        X_df = _to_dataframe(X)
        if self.X_columns_ is not None:
            if list(X_df.columns) != self.X_columns_:
                if X_df.shape[1] != len(self.X_columns_):
                    raise ValueError("Feature columns do not match training columns.")
                X_df.columns = self.X_columns_
        return X_df

    def get_final_black_box_features(self, X) -> pd.DataFrame:
 
        if self.final_black_box_ is None:
            raise RuntimeError("Model is not fit.")

        X_df = _to_dataframe(X)

        if not self.add_history or self.final_black_box_history_stage_idx_ is None:
            return X_df

        if self.projection:
            raise NotImplementedError(
                "get_final_black_box_features is not currently supported "
                "with projection=True."
            )

        return self._feature_frame_for_stage(
            X_df,
            self.final_black_box_history_stage_idx_,
        )

    def _initial_interval_state(self) -> IntervalState:
        return IntervalState(
            feature_to_group=getattr(self, "feature_to_group_", {}),
            feature_to_pos=getattr(self, "feature_to_pos_", {}),
            group_feature_names=getattr(self, "group_feature_names_", {}),
            categorical_eps=getattr(self, "categorical_eps", CATEGORICAL_EPS_DEFAULT),
            history_feature_columns_by_stage=getattr(self, "history_feature_columns_by_stage_", {}),
        )

    def _has_pseudo_weightedsep_final_stage(self) -> bool:
        return isinstance(self.final_black_box_, _PseudoWeightedSEPBlackBox)

    def _has_true_black_box(self, *, include_pseudo_weightedsep_final_stage: bool = True) -> bool:
        if self.final_black_box_ is None:
            return False
        if include_pseudo_weightedsep_final_stage and self._has_pseudo_weightedsep_final_stage():
            return False
        return True

    # take the binary featured tree into a real continuous feature tree
    def _decode_pseudo_weightedsep_tree(self, pseudo_bb: Optional[_PseudoWeightedSEPBlackBox] = None) -> Optional[RuleTreeNode]:
        if pseudo_bb is None:
            pseudo_bb = self.final_black_box_
        if not isinstance(pseudo_bb, _PseudoWeightedSEPBlackBox):
            return None

        tg = pseudo_bb.tg # stores mapping of binary feature to continuous and threshold
        sep = pseudo_bb.sep # works in binary feature space
        paths, actions = sep.get_tree_paths()

        decoded_paths: List[List[Tuple[str, float, bool]]] = []
        decoded_actions: List[Any] = []
        for p, a in zip(paths, actions):
            new_p = []
            for signed_f in p:
                bin_idx = abs(int(signed_f)) - 1 # unsign and 0 index
                raw_feat_idx, thr = tg.thresholds_[bin_idx]
                feat_name = str(tg.feature_names_in_[raw_feat_idx])
                truth_val = bool(signed_f > 0)
                new_p.append((feat_name, float(thr), truth_val))
            decoded_paths.append(new_p) # the continuous feature leaf paths
            if int(a) == -1:
                raise RuntimeError("Pseudo WeightedSEP stage unexpectedly contains a defer leaf.")
            decoded_actions.append(int(a))

        return _TreeOps.tree_from_decoded_paths(decoded_paths, decoded_actions)

    # all trees in the chain (maybe including final pseudo stage)
    def _get_all_stage_roots(self, *, include_pseudo_weightedsep_final_stage: bool = True) -> List[RuleTreeNode]:
        # self._check_is_fit()
        roots = [self.get_stage_tree(i).clone() for i in range(len(self.stages_))]
        if include_pseudo_weightedsep_final_stage and self._has_pseudo_weightedsep_final_stage():
            pseudo_root = self._decode_pseudo_weightedsep_tree()
            if pseudo_root is not None:
                roots.append(pseudo_root)
        return roots

    # extract actual trees from the black box
    def _get_black_box_ensemble_raw_trees(self) -> List[RuleTreeNode]:
        # self._check_is_fit()
        model = self.final_black_box_

        if isinstance(model, _PseudoWeightedSEPBlackBox):
            root = self._decode_pseudo_weightedsep_tree(model)
            return [] if root is None else [root]

        if isinstance(model, RandomForestClassifier):
            return [
                _TreeOps.build_rule_tree_from_sklearn_tree(est, self.X_columns_)
                for est in model.estimators_
            ]

        if isinstance(model, RandomForestRegressor):
            return [
                _TreeOps.build_rule_tree_from_sklearn_tree(est, self.X_columns_)
                for est in model.estimators_
            ]

        if isinstance(model, GradientBoostingClassifier):
            trees: List[RuleTreeNode] = []
            ests = np.asarray(model.estimators_, dtype=object)

            if ests.ndim == 1:
                for est in ests:
                    trees.append(
                        _TreeOps.build_rule_tree_from_sklearn_tree(est, self.X_columns_)
                    )
            elif ests.ndim == 2:
                # Binary:     shape = (n_estimators, 1)
                # Multiclass: shape = (n_estimators, n_classes)
                # Include every class-specific tree.
                for i in range(ests.shape[0]):
                    for k in range(ests.shape[1]):
                        trees.append(
                            _TreeOps.build_rule_tree_from_sklearn_tree(
                                ests[i, k],
                                self.X_columns_,
                            )
                        )
            else:
                raise RuntimeError(
                    f"Unexpected GradientBoostingClassifier estimators_ ndim={ests.ndim}"
                )

            return trees

        if isinstance(model, GradientBoostingRegressor):
            trees: List[RuleTreeNode] = []
            ests = np.asarray(model.estimators_, dtype=object)

            if ests.ndim == 1:
                for est in ests:
                    trees.append(
                        _TreeOps.build_rule_tree_from_sklearn_tree(est, self.X_columns_)
                    )
            elif ests.ndim == 2:
                for i in range(ests.shape[0]):
                    for k in range(ests.shape[1]):
                        trees.append(
                            _TreeOps.build_rule_tree_from_sklearn_tree(
                                ests[i, k],
                                self.X_columns_,
                            )
                        )
            else:
                raise RuntimeError(
                    f"Unexpected GradientBoostingRegressor estimators_ ndim={ests.ndim}"
                )

            return trees

        if isinstance(model, (XGBClassifier, XGBRegressor)):
            booster = model.get_booster()
            df = booster.trees_to_dataframe()

            if df.empty:
                return []

            out: List[RuleTreeNode] = []
            for tree_id in sorted(df["Tree"].unique().tolist()):
                df_tree = df[df["Tree"] == tree_id].copy()
                out.append(_TreeOps.build_rule_tree_from_xgb_dataframe(df_tree))

            return out

        if isinstance(model, FIGSClassifier):
            if not hasattr(model, "trees_"):
                raise RuntimeError("FIGS model has no trees_ attribute after fitting.")

            return [
                _TreeOps.build_rule_tree_from_figs_tree(tree, self.X_columns_)
                for tree in model.trees_
            ]

        raise ValueError(
            f"Unsupported final black-box type for tree extraction: {type(model).__name__}"
        )

    # just accessing the weightedsep leaf paths method
    def _stage_paths_and_actions(self, stage_index: int) -> Tuple[List[List[int]], List[int]]:
        # self._check_is_fit()
        if stage_index < 0 or stage_index >= len(self.stages_):
            raise IndexError("stage_index out of range")
        stage = self.stages_[stage_index]
        paths, second = stage.get_tree_paths()

        if self.task_ in {"regression", "linear_regression"} and hasattr(stage, "leaf_actions"):
            actions = stage.leaf_actions()
        else:
            actions = second

        return list(paths), list(actions)

    # basically the same as _decode_pseudo_weightedsep_tree
    def decode_stage_paths(self, stage_index: int) -> Tuple[List[List[RuleLiteral]], List[Any]]:
        # self._check_is_fit()

        # appended pseudo-WeightedSEP final stage
        if stage_index == len(self.stages_) and self._has_pseudo_weightedsep_final_stage():
            root = self._decode_pseudo_weightedsep_tree()
            if root is None:
                raise RuntimeError("Failed to decode pseudo WeightedSEP stage.")
            return self._decode_rule_tree_to_paths(root)

        stage = self.stages_[stage_index]
        if isinstance(stage, SklearnDeferTree):
            paths, actions = self._stage_paths_and_actions(stage_index)

            decoded_paths: List[List[RuleLiteral]] = []
            decoded_actions: List[Any] = []

            if self.X_columns_ is None:
                raise RuntimeError("X_columns_ is missing; cannot decode SklearnDeferTree paths.")

            for p, a in zip(paths, actions):
                new_p: List[RuleLiteral] = []

                for feat_idx, thr, truth_val in p:
                    feat_name = str(self.X_columns_[int(feat_idx)])
                    new_p.append(RuleLiteral(feat_name, float(thr), bool(truth_val)))

                decoded_paths.append(new_p)
                decoded_actions.append("defer" if int(a) == -1 else int(a))

            return decoded_paths, decoded_actions
        
        # if self.same_binarization:
        #     encoder = self.encoder_
        # else:
        #     encoder = self.stage_encoders_[stage_index]

        # paths, actions = self._stage_paths_and_actions(stage_index)
        # decoded_paths: List[List[RuleLiteral]] = []
        # decoded_actions: List[Any] = []

        # for p, a in zip(paths, actions):
        #     new_p: List[RuleLiteral] = []
        #     for signed_f in p:
        #         bin_idx = abs(int(signed_f)) - 1
        #         raw_feat_idx, thr = encoder.thresholds_[bin_idx]
        #         feat_name = str(encoder.feature_names_in_[raw_feat_idx])
        #         truth_val = bool(signed_f > 0)
        #         new_p.append(RuleLiteral(feat_name, float(thr), truth_val))
        #     decoded_paths.append(new_p)
        #     decoded_actions.append("defer" if int(a) == -1 else int(a))

        # return decoded_paths, decoded_actions

        if self.same_binarization:
            encoder = self.encoder_
        else:
            encoder = self.stage_encoders_[stage_index]

        stage = self.stages_[stage_index]
        paths, second = stage.get_tree_paths()

        if self.task_ in {"regression", "linear_regression"} and hasattr(stage, "leaf_actions"):
            actions = stage.leaf_actions()
            values = second
        else:
            actions = second
            values = [None] * len(actions)

        decoded_paths: List[List[RuleLiteral]] = []
        decoded_actions: List[Any] = []

        for leaf_idx, (p, a, v) in enumerate(zip(paths, actions, values)):
            new_p: List[RuleLiteral] = []

            if self.task_ == "linear_regression":
                if hasattr(stage, "export_leaf_paths"):
                    leaf = stage.export_leaf_paths()[leaf_idx]

                    for cond in leaf.get("conditions", []):
                        clari_feat_idx = int(cond["feature_idx"])

                        # CLARITree internally prepends an intercept column.
                        # internal feature 0 is intercept and should never appear in a split.
                        raw_feat_idx = clari_feat_idx - 1

                        if raw_feat_idx < 0 or raw_feat_idx >= len(self.X_columns_):
                            raise RuntimeError(
                                "Invalid CLARITree split feature index. "
                                f"Got internal feature_idx={clari_feat_idx}, which maps to "
                                f"raw feature index {raw_feat_idx}, but X has "
                                f"{len(self.X_columns_)} columns."
                            )

                        feat_name = str(self.X_columns_[raw_feat_idx])
                        thr = float(cond["threshold"])
                        truth_val = cond["op"] == "<="
                        new_p.append(RuleLiteral(feat_name, thr, truth_val))
                else:
                    raise RuntimeError("linear_regression stage must expose export_leaf_paths().")

            else:
                for signed_f in p:
                    bin_idx = abs(int(signed_f)) - 1
                    raw_feat_idx, thr = encoder.thresholds_[bin_idx]
                    feat_name = str(encoder.feature_names_in_[raw_feat_idx])
                    truth_val = bool(signed_f > 0)
                    new_p.append(RuleLiteral(feat_name, float(thr), truth_val))

            decoded_paths.append(new_p)

            a_int = int(a)
            if a_int == -1:
                decoded_actions.append("defer")
            elif self.task_ == "regression":
                decoded_actions.append(float(v))
            elif self.task_ == "linear_regression":
                if a_int == 0:
                    decoded_actions.append(float(v))
                elif a_int == 1:
                    decoded_actions.append(
                        LinearLeafAction(
                            label=f"L_{stage_index}_{leaf_idx}",
                            stage_index=stage_index,
                            leaf_index=leaf_idx,
                        )
                    )
                else:
                    decoded_actions.append(f"unknown_{a_int}")
            else:
                decoded_actions.append(int(a_int))

        return decoded_paths, decoded_actions

    # dfs to get root to leaf paths from tree
    def _decode_rule_tree_to_paths(self, root: RuleTreeNode) -> Tuple[List[List[RuleLiteral]], List[Any]]:
        paths: List[List[RuleLiteral]] = []
        actions: List[Any] = []

        def dfs(node: RuleTreeNode, cur: List[RuleLiteral]) -> None:
            if node.is_leaf() or (node.left is None and node.right is None):
                paths.append(list(cur))
                actions.append(node.action)
                return
            cur.append(RuleLiteral(str(node.feature_name), float(node.threshold), True))
            dfs(node.left, cur)
            cur.pop()
            cur.append(RuleLiteral(str(node.feature_name), float(node.threshold), False))
            dfs(node.right, cur)
            cur.pop()

        dfs(root, [])
        return paths, actions

    # gives tuple form
    def get_decoded_stage_path_tuples(self, stage_index: int) -> Tuple[List[List[Tuple[str, float, bool]]], List[Any]]:
        paths, actions = self.decode_stage_paths(stage_index)
        out_paths = [[(lit.feature_name, lit.threshold, lit.truth_value) for lit in p] for p in paths]
        return out_paths, actions

    # def _raw_leaf_count_weightedsep_stage(self, stage: WeightedSeparable) -> int:
    #     paths, actions = stage.get_tree_paths()
    #     return len(actions)

    # def _raw_total_stage_leaves_so_far(self) -> int:
    #     total = 0
    #     for stage in self.stages_:
    #         total += self._raw_leaf_count_weightedsep_stage(stage)

    #     return int(total)

    def _raw_leaf_count_stage(self, stage) -> int:
        if hasattr(stage, "leaf_actions"):
            return int(len(stage.leaf_actions()))
        paths, actions = stage.get_tree_paths()
        return int(len(actions))


    def _raw_leaf_count_weightedsep_stage(self, stage: WeightedSeparable) -> int:
        paths, actions = stage.get_tree_paths()
        return int(len(actions))


    def _raw_total_stage_leaves_so_far(self) -> int:
        total = 0
        for stage in self.stages_:
            total += self._raw_leaf_count_stage(stage)

        return int(total)

    def get_stage_tree(self, stage_index: int) -> RuleTreeNode:
        decoded_paths, actions = self.get_decoded_stage_path_tuples(stage_index)
        return _TreeOps.tree_from_decoded_paths(decoded_paths, actions)

    def build_single_tree_representation(
        self,
        *,
        compress: bool = True,
        include_pseudo_weightedsep_final_stage: bool = True,
        memory_efficient: bool = True,
    ) -> RuleTreeNode:
        # self._check_is_fit()
        roots = self._get_all_stage_roots(
            include_pseudo_weightedsep_final_stage=include_pseudo_weightedsep_final_stage # this can include the pseudobb weightedsep tree if desired
        )
        if not roots:
            raise RuntimeError("No stages are present.")

        has_true_black_box = self._has_true_black_box(
            include_pseudo_weightedsep_final_stage=include_pseudo_weightedsep_final_stage
        )
        
        # truly memory-efficient path:
        # do a single DFS over the chain, carrying interval state and stage index.
        # we build only the output tree we need, and when we hit a defer leaf we recurse
        # directly into the next stage root under the same state.
        if memory_efficient and compress:
            defer_leaf_label_maps: List[Dict[int, str]] = []
            for s, root_s in enumerate(roots):
                cur_map: Dict[int, str] = {}
                leaf_counter = 0

                def rec_label(node: RuleTreeNode, stage_idx=s):
                    nonlocal leaf_counter

                    if node is None:
                        return

                    if node.is_leaf() or (node.left is None and node.right is None):
                        if node.action == "defer":
                            cur_map[id(node)] = f"hist_s{stage_idx}_defer_leaf_{leaf_counter}"
                        leaf_counter += 1
                        return

                    rec_label(node.left, stage_idx)
                    rec_label(node.right, stage_idx)

                rec_label(root_s)
                defer_leaf_label_maps.append(cur_map)

            def rec(node: Optional[RuleTreeNode], stage_idx: int, state: IntervalState) -> Optional[RuleTreeNode]:
                if node is None:
                    return None

                # leaf
                if node.is_leaf():
                    if node.action != "defer":
                        return RuleTreeNode(action=node.action) # path terminates here, just put a prediction leaf

                    # defer leaf: continue into the next stage under the same path constraints
                    next_stage_idx = stage_idx + 1 
                    if next_stage_idx < len(roots): # if we have a defer leaf, we need to put the simplified subtree (next_stage_idx tree onward) right here, instead of the defer leaf
                        defer_label = defer_leaf_label_maps[stage_idx][id(node)]
                        next_state = self._mark_state_after_defer_leaf(
                            state,
                            stage_idx=stage_idx,
                            defer_leaf_label=defer_label,
                        )

                        return rec(roots[next_stage_idx], next_stage_idx, next_state)

                    # (else) no more explicit stages left
                    if has_true_black_box:
                        return RuleTreeNode(action="black_box") # can't simplify this and display it

                    # should only really happen if caller excluded final pseudo stage and there is no true BB
                    # note that before we call rec we bundle the pseudo weighted separable as the last stage of the tree so we handle it
                    return RuleTreeNode(action="defer")

                feat = str(node.feature_name)
                thr = float(node.threshold)

                # logical simplification on the fly
                if state.implies_true(feat, thr):  # split always goes left
                    return rec(node.left, stage_idx, state) # instead of this node, we just put the simplified left subtree (we go simplify it now)

                if state.implies_false(feat, thr):  # split always goes right
                    return rec(node.right, stage_idx, state)

                true_state = state.add_literal(RuleLiteral(feat, thr, True)) # update our state with the information from taking the split, which is not redundant
                false_state = state.add_literal(RuleLiteral(feat, thr, False))

                # build only the needed simplified output node
                return RuleTreeNode(
                    feature_name=feat,
                    threshold=thr,
                    left=rec(node.left, stage_idx, true_state), # left is true, simplify the left subtree as you go growing it with the added information 
                    right=rec(node.right, stage_idx, false_state),
                    action=None, # not a leaf
                    missing_go_to_left=node.missing_go_to_left, # really should get rid of this sometime
                )

            # return rec(roots[0], 0, self._initial_interval_state())
            root = rec(roots[0], 0, self._initial_interval_state())
            root = _TreeOps.prune_trivial_extensions(root)
            return root

        else:

            root = roots[0].clone()
            for nxt in roots[1:]: # take the first tree, put tree2 in its defer leaves, and then, in that combined tree defer leaves, put t3 in each of them, and so on
                root = _TreeOps.replace_defer_leaves_with_subtree(root, nxt)

            # do we have a true black box that needs to be labeled with black box leaves at the end?
            if has_true_black_box:
                root = _TreeOps.replace_defer_leaves_with_subtree(root, None)

            if compress: # now iterate over and tighten bounds, prune redundant subtrees
                root = _TreeOps.simplify_single_tree(root, self._initial_interval_state())
                # root = _TreeOps.prune_trivial_extensions(root) # simplify_single_tree does this at the end

            return root

    # this does not merge everything into one tree. it keeps the chain as a list of stage trees, but compresses each stage using the contexts under which that stage can actually be reached.
    def _extend_states_with_feasible_paths(
        self,
        incoming_states: List[IntervalState],
        defer_path_label_pairs: List[Tuple[List[RuleLiteral], str]],
        *,
        stage_idx: int,
    ) -> List[IntervalState]:
        next_states: List[IntervalState] = []
        # we do a cross product of the earlier paths, and each defer path within this tree, because any sample could roughly go down any other path.
        for base_state in incoming_states:
            for p, defer_leaf_label in defer_path_label_pairs:
                candidate = base_state
                feasible = True
                for lit in p:  # we have a set of x <= 7 True, and we want to check if they are compatible with our base state
                    candidate = candidate.add_literal(lit, propagate=False) # tighten bounds, propagate is false because categorical can override conflicts, but numerical doesn't, we need to see the conflicts
                    lo, hi = candidate.bounds.get(lit.feature_name, (-np.inf, np.inf))  # our current bounds
                    if lo >= hi:  # one was open interval, one was closed interval, so equality is also not possnle
                        feasible = False
                        break
                    if lit.feature_name in candidate.feature_to_group:
                        group_id = candidate.feature_to_group[lit.feature_name]
                        vec = candidate.categorical_vectors.get(group_id)
                        # TODO: if everything is -1 or more than 1 thing is 1, then break
                        # otherwise, we will propagate 
                        if vec is not None:
                            # impossible if every category is forced false
                            if np.all(vec == -1):
                                feasible = False
                                break
                            # impossible if more than one category is forced true
                            if np.sum(vec == 1) > 1:
                                feasible = False
                                break
                            # we should propagate here, the literal does not conflict so we want to populate what this implies!
                            candidate._propagate_group_vector(vec)
                # if we made it through every literal without a conflict, we are good
                # propagating after determining something is feasible will never make it unfeasible, we're just filling in what it implies
                if feasible and candidate.is_feasible():
                    candidate = self._mark_state_after_defer_leaf(
                        candidate,
                        stage_idx=stage_idx,
                        defer_leaf_label=defer_leaf_label,
                    )

                    if candidate.is_feasible():
                        next_states.append(candidate)
        return next_states

    
    # this does not merge everything into one tree. it keeps the chain as a list of stage trees, but compresses each stage using the contexts under which that stage can actually be reached.
    def build_dag_stage_trees(
        self,
        *,
        compress: bool = True,
        include_pseudo_weightedsep_final_stage: bool = True,
    ) -> List[RuleTreeNode]:
        # self._check_is_fit()
        stage_roots = self._get_all_stage_roots( # again can include the pseudo bb
            include_pseudo_weightedsep_final_stage=include_pseudo_weightedsep_final_stage
        )
        if not stage_roots:
            return []
        if not compress:
            return [r.clone() for r in stage_roots]

        compressed_roots: List[RuleTreeNode] = []
        incoming_states: List[IntervalState] = [self._initial_interval_state()]
        for stage_idx, raw_root in enumerate(stage_roots):
            simp_root = _TreeOps.simplify_dag_tree(raw_root.clone(), incoming_states)
            compressed_roots.append(simp_root)

            if stage_idx >= len(self.stages_):
                incoming_states = []
                continue

            defer_path_label_pairs = self._stage_defer_path_label_pairs(stage_idx)

            if len(defer_path_label_pairs) == 0:
                incoming_states = []
            else:
                incoming_states = self._extend_states_with_feasible_paths(
                    incoming_states,
                    defer_path_label_pairs,
                    stage_idx=stage_idx,
                )
        return compressed_roots

    # like the above, but it also tells you what contexts are left over at the end for the black box to get simplified more
    def _build_dag_stage_trees_and_final_states(
        self,
        *,
        compress: bool = True,
        include_pseudo_weightedsep_final_stage: bool = True,
    ) -> Tuple[List[RuleTreeNode], List[IntervalState]]:
        # self._check_is_fit()
        stage_roots = self._get_all_stage_roots(
            include_pseudo_weightedsep_final_stage=include_pseudo_weightedsep_final_stage
        )
        if not stage_roots:
            return [], []

        if not compress:
            incoming_states: List[IntervalState] = [self._initial_interval_state()]

            for stage_idx, raw_root in enumerate(stage_roots):
                # only real MDT stages have defer-history labels.
                # pseudo weightedsep final stage should not extend defer history.
                if stage_idx >= len(self.stages_):
                    incoming_states = []
                    continue

                defer_path_label_pairs = self._stage_defer_path_label_pairs(stage_idx)

                if len(defer_path_label_pairs) == 0:
                    incoming_states = []
                else:
                    incoming_states = self._extend_states_with_feasible_paths(
                        incoming_states,
                        defer_path_label_pairs,
                        stage_idx=stage_idx,
                    )

            if not self._has_true_black_box(
                include_pseudo_weightedsep_final_stage=include_pseudo_weightedsep_final_stage
            ):
                incoming_states = []

            return [r.clone() for r in stage_roots], incoming_states

        compressed_roots: List[RuleTreeNode] = []
        incoming_states: List[IntervalState] = [self._initial_interval_state()]

        for stage_idx, raw_root in enumerate(stage_roots):
            simp_root = _TreeOps.simplify_dag_tree(raw_root.clone(), incoming_states)
            compressed_roots.append(simp_root)

            if stage_idx >= len(self.stages_):
                incoming_states = []
                continue

            defer_path_label_pairs = self._stage_defer_path_label_pairs(stage_idx)

            if len(defer_path_label_pairs) == 0:
                incoming_states = []
            else:
                incoming_states = self._extend_states_with_feasible_paths(
                    incoming_states,
                    defer_path_label_pairs,
                    stage_idx=stage_idx,
                )

        if not self._has_true_black_box(
            include_pseudo_weightedsep_final_stage=include_pseudo_weightedsep_final_stage
        ):
            incoming_states = []

        return compressed_roots, incoming_states

    # for the whole DAG setup, we need just one black box. we simplify each tree in the black box as long as each way to get to it is fine with a subtree being pruned.
    def build_black_box_dag_trees(self, *, compress: bool = True, compress_chain: bool = True) -> List[RuleTreeNode]:
        if not self._has_true_black_box(include_pseudo_weightedsep_final_stage=True):
            return []
        raw_trees = self._get_black_box_ensemble_raw_trees()
        if not compress:
            return [t.clone() for t in raw_trees]
        # what paths can reach the black box?
        _, final_states = self._build_dag_stage_trees_and_final_states(
            compress=compress_chain,
            include_pseudo_weightedsep_final_stage=True,
        )
        if len(final_states) == 0:
            return [t.clone() for t in raw_trees]
        # simplify each black box tree under those states
        return [_TreeOps.simplify_dag_tree(t.clone(), final_states) for t in raw_trees]

    #  returns the single (maybe compressed tree), all the paths that end in black box, and those paths converted to interval states.
    def _get_single_tree_black_box_paths_and_states(self, *, compress_chain: bool = True) -> Tuple[RuleTreeNode, List[List[RuleLiteral]], List[IntervalState]]:
        root = self.build_single_tree_representation(
            compress=compress_chain,
            include_pseudo_weightedsep_final_stage=True,
        )
        bb_paths = _TreeOps.collect_paths_to_action(root, "black_box")
        bb_states = _TreeOps.states_from_paths(
            bb_paths,
            template_state=self._initial_interval_state(),
        )
        return root, bb_paths, bb_states

    # so for each different way to get to the black box, we are making a simplified version of each tree for that way
    # for a single tree, there is exactly 1 path root-to-final-defer-leaf, so this is fine.
    # for dag, it is much more complicated and needs to be investigated.
    def build_black_box_single_tree_context_trees(
        self,
        *,
        compress: bool = True,
        compress_chain: bool = True,
    ) -> Tuple[List[List[RuleTreeNode]], List[IntervalState]]:
        if not self._has_true_black_box(include_pseudo_weightedsep_final_stage=True):
            return [], []
        raw_trees = self._get_black_box_ensemble_raw_trees()
        # get the contexts where the bb is used
        _, _, bb_states = self._get_single_tree_black_box_paths_and_states(compress_chain=compress_chain)
        trees_by_context: List[List[RuleTreeNode]] = []
        # for each leaf that uses the black box, make a simplified version of each tree
        for st in bb_states:
            cur_group: List[RuleTreeNode] = []
            for t in raw_trees:
                cur_group.append(_TreeOps.simplify_single_tree(t.clone(), st) if compress else t.clone())
            trees_by_context.append(cur_group)
        return trees_by_context, bb_states

    # route samples through the stages (but this is single tree representation) and see what black box stage it hit (or -1 if it got a prediction)
    def _route_samples_to_black_box_single_tree_context(self, X, *, compress_chain: bool = True) -> Tuple[np.ndarray, List[IntervalState]]:
        X_df = self._coerce_X_for_predict(X)
        root, _, bb_states = self._get_single_tree_black_box_paths_and_states(compress_chain=compress_chain)
        sig_to_idx = {_TreeOps.state_signature(st): idx for idx, st in enumerate(bb_states)}
        out = np.full(len(X_df), -1, dtype=np.int64)
        for i in range(len(X_df)):
            action, _, path = _TreeOps.traverse_rule_tree(root, X_df.iloc[i])
            if action == "black_box":
                st = _TreeOps.state_from_path(path, template_state=self._initial_interval_state())
                out[i] = sig_to_idx[_TreeOps.state_signature(st)]
        return out, bb_states

    # sim for dag
    def _route_samples_to_black_box_dag_context(self, X, *, compress_chain: bool = True) -> Tuple[np.ndarray, List[IntervalState]]:
        X_df = self._coerce_X_for_predict(X)
        dag_roots, final_states = self._build_dag_stage_trees_and_final_states(
            compress=compress_chain,
            include_pseudo_weightedsep_final_stage=True,
        )
        sig_to_idx = {_TreeOps.state_signature(st): idx for idx, st in enumerate(final_states)}
        out = np.full(len(X_df), -1, dtype=np.int64)
        for i in range(len(X_df)):
            row = X_df.iloc[i]
            st = self._initial_interval_state()
            reached_bb = True
            for root in dag_roots:
                action, _, path = _TreeOps.traverse_rule_tree(root, row)
                st = st.add_path(path)
                if action != "defer":
                    reached_bb = False
                    break
            if reached_bb:
                sig = _TreeOps.state_signature(st)
                if sig in sig_to_idx:
                    out[i] = sig_to_idx[sig]
        return out, final_states

    # return the feasible deferred states (single tree representation) after going through the first stage_count MDT stages.
    def _get_feasible_deferred_states_through_stage(
        self,
        stage_count: int,
    ) -> List[IntervalState]:

        # stage_count = 0 returns [initial_state]
        if stage_count < 0 or stage_count > len(self.stages_):
            raise ValueError("stage_count out of range.")

        incoming_states: List[IntervalState] = [self._initial_interval_state()]

        for stage_idx in range(stage_count):
            defer_path_label_pairs = self._stage_defer_path_label_pairs(stage_idx)

            if len(defer_path_label_pairs) == 0:
                incoming_states = []
                break

            # incoming states start out blank but then we'll do the filtered cross product as we go to the next tree
            incoming_states = self._extend_states_with_feasible_paths(
                incoming_states,
                defer_path_label_pairs,
                stage_idx=stage_idx,
            )

            if len(incoming_states) == 0:
                break

        return incoming_states

class _MDTAnalysisMixin:
    @staticmethod
    def _collect_leaves_in_order(node: RuleTreeNode, leaves: List[RuleTreeNode]) -> None:
        if node is None:
            return
        if node.is_leaf() or (node.left is None and node.right is None):
            leaves.append(node)
            return
        _MDTAnalysisMixin._collect_leaves_in_order(node.left, leaves)
        _MDTAnalysisMixin._collect_leaves_in_order(node.right, leaves)

    @staticmethod
    def _assign_positions_tree(root: RuleTreeNode) -> Dict[int, Tuple[float, float]]:
        positions: Dict[int, Tuple[float, float]] = {}
        leaves: List[RuleTreeNode] = []
        _MDTAnalysisMixin._collect_leaves_in_order(root, leaves)
        if not leaves:
            leaves = [root]
        leaf_x = {id(leaf): i for i, leaf in enumerate(leaves)}

        def dfs(node: RuleTreeNode, depth: int) -> None:
            if node is None:
                return
            if node.is_leaf() or (node.left is None and node.right is None):
                x = leaf_x[id(node)]
                positions[id(node)] = (x, -depth)
                return
            dfs(node.left, depth + 1)
            dfs(node.right, depth + 1)
            xl, _ = positions[id(node.left)]
            xr, _ = positions[id(node.right)]
            positions[id(node)] = (0.5 * (xl + xr), -depth)

        dfs(root, 0)
        return positions

    @staticmethod
    def _shrink_segment(x1, y1, x2, y2, r1, r2):
        dx = x2 - x1
        dy = y2 - y1
        dist = (dx * dx + dy * dy) ** 0.5
        if dist == 0:
            return x1, y1, x2, y2
        ux = dx / dist
        uy = dy / dist
        return (x1 + ux * r1, y1 + uy * r1, x2 - ux * r2, y2 - uy * r2)

    @staticmethod
    def plot_rule_tree(root: RuleTreeNode, *, figsize=(12, 7), ax=None, title: Optional[str] = None, show: bool = True):
        if root is None:
            raise ValueError("root is None")

        pos_local = _MDTAnalysisMixin._assign_positions_tree(root)
        x_scale = 6.5
        y_scale = 5.0
        pos = {}
        xs = []
        ys = []

        def collect(node: RuleTreeNode, node_map: Dict[int, RuleTreeNode]):
            if node is None:
                return
            node_map[id(node)] = node
            collect(node.left, node_map)
            collect(node.right, node_map)

        node_map: Dict[int, RuleTreeNode] = {}
        collect(root, node_map)

        for node_id, (x, y) in pos_local.items():
            xx = x * x_scale
            yy = y * y_scale
            pos[node_id] = (xx, yy)
            xs.append(xx)
            ys.append(yy)

        minx, maxx = min(xs), max(xs)
        x_center = 0.5 * (minx + maxx)
        for node_id, (xx, yy) in list(pos.items()):
            pos[node_id] = (xx - x_center, yy)

        if ax is None:
            total_w = (max(xs) - min(xs)) if xs else 10.0
            total_h = (max(ys) - min(ys)) if ys else 6.0
            fig_w = max(figsize[0], total_w / 2.2)
            fig_h = max(figsize[1], total_h / 1.6)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140)
        else:
            fig = ax.figure

        ax.set_axis_off()
        internal_r = 0.9
        leaf_r = 1.6

        def draw(node: RuleTreeNode):
            x, y = pos[id(node)]
            if not node.is_leaf():
                if node.left is not None:
                    x2, y2 = pos[id(node.left)]
                    r_child = leaf_r if node.left.is_leaf() else internal_r
                    sx, sy, ex, ey = _MDTAnalysisMixin._shrink_segment(x, y, x2, y2, internal_r, r_child)
                    ax.add_line(Line2D([sx, ex], [sy, ey], color="black", linewidth=2.2, zorder=1))
                    mx, my = 0.5 * (sx + ex), 0.5 * (sy + ey)
                    # ax.text(mx, my + 0.2, "T", fontsize=12, ha="center", va="center")
                    draw(node.left)
                if node.right is not None:
                    x2, y2 = pos[id(node.right)]
                    r_child = leaf_r if node.right.is_leaf() else internal_r
                    sx, sy, ex, ey = _MDTAnalysisMixin._shrink_segment(x, y, x2, y2, internal_r, r_child)
                    ax.add_line(Line2D([sx, ex], [sy, ey], color="black", linewidth=2.2, zorder=1))
                    mx, my = 0.5 * (sx + ex), 0.5 * (sy + ey)
                    # ax.text(mx, my + 0.2, "F", fontsize=12, ha="center", va="center")
                    draw(node.right)

            if node.is_leaf():
                if node.action == "defer":
                    face = "#fff2cc"
                    label = "D"
                elif node.action == "black_box":
                    face = "#f4cccc"
                    label = "BB"
                else:
                    face = "#e0ffd8"
                    if isinstance(node.action, LinearLeafAction):
                        label = node.action.label
                    elif isinstance(node.action, float):
                        label = f"{node.action:.4g}"
                    elif isinstance(node.action, np.floating):
                        label = f"{float(node.action):.4g}"
                    else:
                        label = str(node.action)
                circ = Circle((x, y), leaf_r, facecolor=face, edgecolor="black", linewidth=1.6, zorder=10)
                ax.add_patch(circ)
                ax.text(x, y, label, ha="center", va="center", fontsize=40, fontweight="bold", zorder=10000, clip_on=False)
            else:
                face = "#ddeeff"
                circ = Circle((x, y), internal_r, facecolor=face, edgecolor="black", linewidth=1.6, zorder=10)
                ax.add_patch(circ)
                label = f"{node.feature_name}\n≤ {node.threshold:g}"
                ax.annotate(
                    label,
                    xy=(x, y),
                    xycoords="data",
                    xytext=(0, 12),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=30,
                    bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.98),
                    zorder=10000,
                    clip_on=False,
                )

        draw(root)
        all_xy = list(pos.values())
        xs, ys = zip(*all_xy) if all_xy else ([0.0], [0.0])
        ax.set_xlim(min(xs) - 3.0, max(xs) + 3.0)
        ax.set_ylim(min(ys) - 4.0, max(ys) + 4.0)
        ax.margins(x=0.08, y=0.12)
        for t in ax.texts:
            t.set_clip_on(False)
        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        ax.set_title("Rule Tree" if title is None else str(title), fontsize=16, pad=12)
        if show:
            plt.show()
        return fig, ax

    def _fit_threshold_guesser(self, X_df: pd.DataFrame, y: np.ndarray, sample_weight, seed: int):
        tg_params = dict(self.threshold_guess_params or {})
        tg_params.setdefault("n_estimators", 150)
        tg_params.setdefault("max_depth", 2)
        tg_params.setdefault("random_state", seed)
        tg_params.setdefault("column_elimination", False)

        tg = ThresholdGuessBinarizer(**tg_params)

        Xb = tg.fit_transform(X_df, y, sample_weight=sample_weight).astype(np.uint8, copy=False)

        return tg, Xb

    def _resolve_task_and_y(self, y, n: int):
        raw_y = np.asarray(y)

        if self.task is None:
            if _is_binary_01(raw_y):
                self.task_ = "classification"
                self.is_binary_classification_ = True
                y_out = _to_binary_y(raw_y, n)
                self.n_classes_ = 2
                return y_out

            raise ValueError(
                "y is not binary 0/1. Pass task='classification', "
                "task='regression', or task='linear_regression'."
            )

        self.task_ = self.task

        if self.task_ == "classification":
            if _is_binary_01(raw_y):
                self.is_binary_classification_ = True
                y_out = _to_binary_y(raw_y, n)
                self.n_classes_ = 2
                return y_out

            self.is_binary_classification_ = False
            y_out = _to_multiclass_y(raw_y, n, max_classes=50)
            self.n_classes_ = int(y_out.max()) + 1
            return y_out

        self.is_binary_classification_ = False
        self.n_classes_ = None
        return _to_regression_y(raw_y, n)

    def fit(
        self,
        X,
        y,
        bb_pred: Optional[np.ndarray] = None,
        binarizer: Optional[ThresholdGuessBinarizer] = None,
    ) -> "MDTChain":
        X_df = _to_dataframe(X)
        y = self._resolve_task_and_y(y, len(X_df))

        if binarizer is not None and not self.same_binarization:
            raise ValueError(
                "binarizer can only be supplied when same_binarization=True."
            )

        bb_pred_override = None
        if bb_pred is not None:
            if self.task_ == "classification":
                if self.is_binary_classification_:
                    bb_pred_override = _to_binary_pred_vector(bb_pred, len(X_df), name="bb_pred")
                else:
                    bb_pred_override = _to_multiclass_pred_vector(
                        bb_pred,
                        len(X_df),
                        self.n_classes_,
                        name="bb_pred",
                    )
            else:
                bb_pred_override = _to_regression_pred_vector(bb_pred, len(X_df), name="bb_pred")

            self.retrain_black_box = False
            self.out_of_fold_predictions = False
        
            if self.set_weights_using_bb_leaves:
                raise ValueError(
                    "bb_pred=... is not compatible with set_weights_using_bb_leaves=True, "
                    "because a fixed prediction vector has no tree leaves."
                )

        self.X_columns_ = list(X_df.columns)
        self.n_features_in_ = X_df.shape[1]

        self.grouped_categories_, self.feature_to_group_, self.feature_to_pos_, self.group_feature_names_ = (
            _validate_grouped_categories(X_df, self.grouped_categories)
        )

        # self.encoder_ = ThresholdGuessBinarizer(**self.threshold_guess_params)
        # Xb = self.encoder_.fit_transform(X_df, y).astype(np.uint8, copy=False)
        self.stage_encoders_ = []
        self.encoder_ = None
        self.history_feature_columns_by_stage_ = {}
        self.final_black_box_history_stage_idx_ = None

        if self.task_ == "linear_regression":
            # CLARITree regression uses raw X, not threshold-binarized X.
            # Xb is only a dummy because _fit_defer_stage accepts both X_df_stage and Xb_stage.
            Xb = np.zeros((len(X_df), 0), dtype=np.uint8)

        elif self.same_binarization:
            if binarizer is not None:
                self.encoder_ = binarizer
                Xb = self.encoder_.transform(X_df).astype(np.uint8, copy=False)
            else:
                self.encoder_, Xb = self._fit_threshold_guesser(
                    X_df, y, sample_weight=None, seed=self.random_state
                )

        else:
            Xb = None

        self.stages_ = []
        self.final_black_box_ = None
        self.stop_reason_ = None

        # if self.training_method == "top_down":
        #     self._fit_top_down(X_df, y, Xb)
        # else:
        #     self._fit_bottom_up(X_df, y, Xb)
        if self.training_method == "top_down":
            self._fit_top_down(X_df, y, Xb, bb_pred_override=bb_pred_override)
        else:
            if not self.same_binarization:
                raise ValueError("same_binarization=False only makes sense for 'top_down'")
            self._fit_bottom_up(X_df, y, Xb)

        self.n_chain_stages_ = len(self._get_all_stage_roots(include_pseudo_weightedsep_final_stage=True))
        train_pred, train_eventual = self._predict_chain_with_final_bb(X_df)
        self.final_train_predictions_ = train_pred
        self.final_train_eventual_deferral_mask_ = train_eventual
        return self

    def _is_defer_prediction(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw)
        if self.task_ == "classification":
            return raw == int(self.placeholder)
        return np.isnan(raw)


    def _clean_stage_predictions(self, pred: np.ndarray) -> np.ndarray:
        pred = np.asarray(pred)
        if self.task_ == "classification":
            if self.is_binary_classification_:
                return pred.astype(np.uint8, copy=False) & np.uint8(1)
            return pred.astype(np.int64, copy=False)
        return pred.astype(np.float64, copy=False)

    def _predict_chain_with_final_bb(
        self,
        X,
        bb_pred: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # self._check_is_fit()
        X_df = self._coerce_X_for_predict(X)
        n = int(X_df.shape[0])

        bb_pred_full = None
        if bb_pred is not None:
            if self.task_ == "classification":
                if self.is_binary_classification_:
                    bb_pred_full = _to_binary_pred_vector(bb_pred, n, name="bb_pred")
                else:
                    bb_pred_full = _to_multiclass_pred_vector(
                        bb_pred, n, self.n_classes_, name="bb_pred"
                    )
            else:
                bb_pred_full = _to_regression_pred_vector(bb_pred, n, name="bb_pred")

        if self.task_ == "classification":
            final = np.zeros(n, dtype=np.int64)
        else:
            final = np.zeros(n, dtype=np.float64)

        eventual = np.zeros(n, dtype=bool)
        active = np.arange(n, dtype=np.int64)

        for stage_idx, stage in enumerate(self.stages_):
            if active.size == 0:
                break

            X_stage_full = self._stage_feature_frame_for_stage_model(X_df, stage_idx)
            X_active = X_stage_full.iloc[active]

            if self.task_ == "linear_regression":
                Xb_stage = np.zeros((len(X_active), 0), dtype=np.uint8)
            elif self.same_binarization:
                Xb_stage = self.encoder_.transform(X_active).astype(np.uint8, copy=False)
            else:
                Xb_stage = self.stage_encoders_[stage_idx].transform(X_active).astype(
                    np.uint8, copy=False
                )

            raw = self._raw_stage_predict(stage, X_active, Xb_stage)

            is_def = self._is_defer_prediction(raw)
            handled = ~is_def

            if handled.any():
                handled_idx = active[handled]
                final[handled_idx] = self._clean_stage_predictions(raw[handled])

            active = active[is_def]

        if active.size > 0:
            if bb_pred_full is not None:
                final[active] = self._clean_stage_predictions(bb_pred_full[active])
                eventual[active] = True

            elif isinstance(self.final_black_box_, _FixedPredictionBlackBox):
                bb_full = self._predict_black_box(self.final_black_box_, X_df)
                final[active] = self._clean_stage_predictions(bb_full[active])
                eventual[active] = True

            elif self._has_pseudo_weightedsep_final_stage():
                # I would keep pseudo WeightedSEP un-history-augmented for now.
                # If add_history=True, you should probably have forbidden eliminate/pseudo-final.
                pseudo_pred = self._predict_black_box(
                    self.final_black_box_,
                    X_df.iloc[active],
                )
                final[active] = self._clean_stage_predictions(pseudo_pred)
                eventual[active] = False

            else:
                if self.add_history and self.final_black_box_history_stage_idx_ is not None:
                    X_final_full = self._feature_frame_for_stage(
                        X_df,
                        self.final_black_box_history_stage_idx_,
                    )
                    X_final_active = X_final_full.iloc[active]
                else:
                    X_final_active = X_df.iloc[active]

                bb_pred_active = self._predict_black_box(
                    self.final_black_box_,
                    X_final_active,
                )
                final[active] = self._clean_stage_predictions(bb_pred_active)
                eventual[active] = True

        return final, eventual

    def predict(self, X, bb_pred: Optional[np.ndarray] = None) -> np.ndarray:
        pred, _ = self._predict_chain_with_final_bb(X, bb_pred=bb_pred)
        return pred

    def predict_with_eventual_deferral(
        self,
        X,
        bb_pred: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._predict_chain_with_final_bb(X, bb_pred=bb_pred)

    def eventual_deferral_proportion(
        self,
        X=None,
        bb_pred: Optional[np.ndarray] = None,
    ) -> float:
        if X is None:
            if bb_pred is not None:
                raise ValueError("bb_pred can only be passed when X is also passed.")
            if self.final_train_eventual_deferral_mask_ is None:
                raise RuntimeError("Model is not fit.")
            m = self.final_train_eventual_deferral_mask_
            return float(m.mean()) if m.size else 0.0

        _, eventual = self._predict_chain_with_final_bb(X, bb_pred=bb_pred)
        return float(eventual.mean()) if eventual.size else 0.0

    def routing_stats(
        self,
        X,
        bb_pred: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        pred, eventual = self._predict_chain_with_final_bb(X, bb_pred=bb_pred)
        n = int(len(pred))
        n_bb = int(eventual.sum())
        return {
            "N": n,
            "n_eventual_black_box": n_bb,
            "frac_eventual_black_box": (n_bb / n) if n else 0.0,
            "n_handled_by_chain": n - n_bb,
            "frac_handled_by_chain": ((n - n_bb) / n) if n else 0.0,
            "n_stages": len(self._get_all_stage_roots(include_pseudo_weightedsep_final_stage=True)),
            "stop_reason": self.stop_reason_,
            "training_method": self.training_method,
            "pseudo_weightedsep_folded_into_chain": self._has_pseudo_weightedsep_final_stage(),
            "has_true_black_box": self._has_true_black_box(include_pseudo_weightedsep_final_stage=True),
        }

    def plot_stage_tree(self, stage_index: int, *, compressed: bool = False, dag_representation: bool = False, title: Optional[str] = None, show: bool = True):
        if dag_representation:
            trees = self.build_dag_stage_trees(compress=compressed, include_pseudo_weightedsep_final_stage=True)
            root = trees[stage_index]
            ttl = title if title is not None else f"Stage {stage_index + 1} DAG tree"
        else:
            root = self.get_stage_tree(stage_index)
            ttl = title if title is not None else f"Stage {stage_index + 1} raw tree"
        return self.plot_rule_tree(root, title=ttl, show=show)

    def plot_single_tree(self, *, compress: bool = True, include_pseudo_weightedsep_final_stage: bool = True, title: Optional[str] = None, show: bool = True):
        root = self.build_single_tree_representation(
            compress=compress,
            include_pseudo_weightedsep_final_stage=include_pseudo_weightedsep_final_stage,
        )
        ttl = title if title is not None else (
            "Compressed single-tree representation" if compress else "Uncompressed single-tree representation"
        )
        return self.plot_rule_tree(root, title=ttl, show=show)

    def number_of_leaves(
        self,
        *,
        representation: str = "single_tree",
        stage_index: Optional[int] = None,
        compress: bool = True,
        include_pseudo_weightedsep_final_stage: bool = True,
    ) -> int:
        # self._check_is_fit()
        rep = str(representation).lower()

        if rep in {"stage_tree", "raw"}:
            if stage_index is None:
                raise ValueError("stage_index must be provided for representation='raw'/'stage_tree'")
            roots = self._get_all_stage_roots(
                include_pseudo_weightedsep_final_stage=include_pseudo_weightedsep_final_stage
            )
            root = roots[stage_index]
        elif rep == "dag":
            if stage_index is None:
                raise ValueError("stage_index must be provided for representation='dag'")
            dag_roots = self.build_dag_stage_trees(
                compress=compress,
                include_pseudo_weightedsep_final_stage=include_pseudo_weightedsep_final_stage,
            )
            root = dag_roots[stage_index]
        elif rep == "single_tree":
            root = self.build_single_tree_representation(
                compress=compress,
                include_pseudo_weightedsep_final_stage=include_pseudo_weightedsep_final_stage,
            )
        else:
            raise ValueError("representation must be one of {'raw', 'stage_tree', 'dag', 'single_tree'}")

        return _TreeOps.count_leaves_rule_tree(root)

    def leaves_per_stage(
        self,
        *,
        representation: str = "raw",
        compress: bool = True,
        include_pseudo_weightedsep_final_stage: bool = True,
    ) -> List[int]:
        rep = str(representation).lower()
        if rep == "raw":
            roots = self._get_all_stage_roots(
                include_pseudo_weightedsep_final_stage=include_pseudo_weightedsep_final_stage
            )
        elif rep == "dag":
            roots = self.build_dag_stage_trees(
                compress=compress,
                include_pseudo_weightedsep_final_stage=include_pseudo_weightedsep_final_stage,
            )
        else:
            raise ValueError("representation must be one of {'raw', 'dag'}")
        return [_TreeOps.count_leaves_rule_tree(r) for r in roots]

    def total_leaves_across_stages(
        self,
        *,
        representation: str = "raw",
        compress: bool = True,
        include_pseudo_weightedsep_final_stage: bool = True,
    ) -> int:
        return int(sum(self.leaves_per_stage(
            representation=representation,
            compress=compress,
            include_pseudo_weightedsep_final_stage=include_pseudo_weightedsep_final_stage,
        )))

    def black_box_number_of_leaves(
        self,
        *,
        representation: str = "raw",
        compress: bool = True,
        compress_chain: bool = True,
    ) -> int:
        # self._check_is_fit()
        if not self._has_true_black_box(include_pseudo_weightedsep_final_stage=True):
            return 0
        rep = str(representation).lower()
        if rep == "raw":
            trees = self._get_black_box_ensemble_raw_trees()
            return int(sum(_TreeOps.count_leaves_rule_tree(t) for t in trees))
        if rep == "dag":
            trees = self.build_black_box_dag_trees(compress=compress, compress_chain=compress_chain)
            return int(sum(_TreeOps.count_leaves_rule_tree(t) for t in trees))
        if rep == "single_tree":
            trees_by_context, _ = self.build_black_box_single_tree_context_trees(
                compress=compress,
                compress_chain=compress_chain,
            )
            total = 0
            for group in trees_by_context:
                for t in group:
                    total += _TreeOps.count_leaves_rule_tree(t)
            return int(total)
        raise ValueError("representation must be one of {'raw', 'dag', 'single_tree'}")

    def black_box_path_lengths(
        self,
        X,
        *,
        representation: str = "raw",
        compress: bool = True,
        compress_chain: bool = True,
    ) -> np.ndarray:
        # self._check_is_fit()
        X_df = self._coerce_X_for_predict(X)
        n = len(X_df)
        rep = str(representation).lower()

        if not self._has_true_black_box(include_pseudo_weightedsep_final_stage=True):
            return np.zeros(n, dtype=np.int64)

        if rep == "raw":
            trees = self._get_black_box_ensemble_raw_trees()
            _, eventual = self._predict_chain_with_final_bb(X_df)
            per_sample = np.zeros(n, dtype=np.int64)
            bb_idx = np.where(eventual)[0]
            if bb_idx.size > 0:
                X_bb = X_df.iloc[bb_idx]
                total = np.zeros(len(X_bb), dtype=np.int64)
                for t in trees:
                    total += _TreeOps.predict_path_lengths_rule_tree(t, X_bb)
                per_sample[bb_idx] = total
            return per_sample

        if rep == "dag":
            trees = self.build_black_box_dag_trees(compress=compress, compress_chain=compress_chain)
            ctx_idx, _ = self._route_samples_to_black_box_dag_context(X_df, compress_chain=compress_chain)
            per_sample = np.zeros(n, dtype=np.int64)
            bb_idx = np.where(ctx_idx >= 0)[0]
            if bb_idx.size > 0:
                X_bb = X_df.iloc[bb_idx]
                total = np.zeros(len(X_bb), dtype=np.int64)
                for t in trees:
                    total += _TreeOps.predict_path_lengths_rule_tree(t, X_bb)
                per_sample[bb_idx] = total
            return per_sample

        if rep == "single_tree":
            trees_by_context, _ = self.build_black_box_single_tree_context_trees(
                compress=compress,
                compress_chain=compress_chain,
            )
            ctx_idx, _ = self._route_samples_to_black_box_single_tree_context(X_df, compress_chain=compress_chain)
            per_sample = np.zeros(n, dtype=np.int64)
            bb_idx = np.where(ctx_idx >= 0)[0]
            for i in bb_idx:
                group = trees_by_context[int(ctx_idx[i])]
                row = X_df.iloc[i]
                cnt = 0
                for t in group:
                    cnt += _TreeOps.path_length_one_rule_tree(t, row)
                per_sample[i] = cnt
            return per_sample

        raise ValueError("representation must be one of {'raw', 'dag', 'single_tree'}")

    def path_lengths_chain(
        self,
        X,
        *,
        representation: str = "raw",
        compress: bool = True,
        include_black_box: bool = False,
        compress_black_box: bool = True,
        include_pseudo_weightedsep_final_stage: bool = True,
    ) -> np.ndarray:
        # self._check_is_fit()
        X_df = self._coerce_X_for_predict(X)
        rep = str(representation).lower()

        if rep == "single_tree":
            root = self.build_single_tree_representation(
                compress=compress,
                include_pseudo_weightedsep_final_stage=include_pseudo_weightedsep_final_stage,
            )
            counts = _TreeOps.predict_path_lengths_rule_tree(root, X_df)
            if include_black_box and self._has_true_black_box(
                include_pseudo_weightedsep_final_stage=include_pseudo_weightedsep_final_stage
            ):
                counts = counts + self.black_box_path_lengths(
                    X_df,
                    representation="single_tree",
                    compress=compress_black_box,
                    compress_chain=compress,
                )
            return counts

        if rep == "raw":
            stage_roots = self._get_all_stage_roots(
                include_pseudo_weightedsep_final_stage=include_pseudo_weightedsep_final_stage
            )
        elif rep == "dag":
            stage_roots = self.build_dag_stage_trees(
                compress=compress,
                include_pseudo_weightedsep_final_stage=include_pseudo_weightedsep_final_stage,
            )
        else:
            raise ValueError("representation must be one of {'raw', 'dag', 'single_tree'}")

        n = len(X_df)
        counts = np.zeros(n, dtype=np.int64)
        active = np.arange(n, dtype=np.int64)

        for stage_idx, stage_tree in enumerate(stage_roots):
            if active.size == 0:
                break
            counts_active = _TreeOps.predict_path_lengths_rule_tree(stage_tree, X_df.iloc[active])
            counts[active] += counts_active
            next_active = []

            if rep == "raw" and stage_idx < len(self.stages_):
                stage = self.stages_[stage_idx]
                X_active = X_df.iloc[active]

                if self.task_ == "linear_regression":
                    Xb_stage = np.zeros((len(X_active), 0), dtype=np.uint8)
                elif self.same_binarization:
                    Xb_stage = self.encoder_.transform(X_active).astype(np.uint8, copy=False)
                else:
                    Xb_stage = self.stage_encoders_[stage_idx].transform(X_active).astype(np.uint8, copy=False)

                raw = self._raw_stage_predict(stage, X_active, Xb_stage)
                is_def = self._is_defer_prediction(raw)
                next_active = active[is_def].tolist()
            else:
                for global_i in active:
                    action, _, _ = _TreeOps.traverse_rule_tree(stage_tree, X_df.iloc[global_i])
                    if action == "defer":
                        next_active.append(int(global_i))

            active = np.asarray(next_active, dtype=np.int64)

        if include_black_box and active.size > 0 and self._has_true_black_box(
            include_pseudo_weightedsep_final_stage=include_pseudo_weightedsep_final_stage
        ):
            if rep == "raw":
                bb_counts = self.black_box_path_lengths(X_df, representation="raw", compress=False, compress_chain=compress)
            elif rep == "dag":
                bb_counts = self.black_box_path_lengths(
                    X_df,
                    representation="dag",
                    compress=compress_black_box,
                    compress_chain=compress,
                )
            else:
                raise RuntimeError("Unexpected representation")
            counts += bb_counts

        return counts

    def get_all_stage_path_tuples(self) -> List[Tuple[List[List[Tuple[str, float, bool]]], List[Any]]]:
        out = []
        for i in range(len(self._get_all_stage_roots(include_pseudo_weightedsep_final_stage=True))):
            out.append(self.get_decoded_stage_path_tuples(i))
        return out

    def get_stage_defer_paths(self, stage_index: int, *, compressed: bool = False, dag_representation: bool = False) -> List[List[Tuple[str, float, bool]]]:
        if dag_representation:
            trees = self.build_dag_stage_trees(
                compress=compressed,
                include_pseudo_weightedsep_final_stage=True,
            )
            root = trees[stage_index]
        else:
            root = self.get_stage_tree(stage_index)
        paths = _TreeOps.collect_paths_to_action(root, "defer")
        return [[(lit.feature_name, lit.threshold, lit.truth_value) for lit in p] for p in paths]

    def chain_structural_stats(
        self,
        X,
        *,
        compress_single_tree: bool = True,
        compress_dag: bool = True,
        include_black_box_splits: bool = False,
        compress_black_box: bool = True,
    ) -> Dict[str, Any]:
        X_df = self._coerce_X_for_predict(X)
        return {
            "n_stages_including_pseudo_weightedsep": len(
                self._get_all_stage_roots(include_pseudo_weightedsep_final_stage=True)
            ),
            "pseudo_weightedsep_folded_into_chain": self._has_pseudo_weightedsep_final_stage(),
            "has_true_black_box": self._has_true_black_box(include_pseudo_weightedsep_final_stage=True),
            "single_tree_leaves": self.number_of_leaves(
                representation="single_tree",
                compress=compress_single_tree,
                include_pseudo_weightedsep_final_stage=True,
            ),
            "raw_stage_leaves": self.leaves_per_stage(
                representation="raw",
                include_pseudo_weightedsep_final_stage=True,
            ),
            "dag_stage_leaves": self.leaves_per_stage(
                representation="dag",
                compress=compress_dag,
                include_pseudo_weightedsep_final_stage=True,
            ),
            "raw_path_lengths": self.path_lengths_chain(
                X_df,
                representation="raw",
                compress=False,
                include_black_box=include_black_box_splits,
                compress_black_box=False,
                include_pseudo_weightedsep_final_stage=True,
            ),
            "dag_path_lengths": self.path_lengths_chain(
                X_df,
                representation="dag",
                compress=compress_dag,
                include_black_box=include_black_box_splits,
                compress_black_box=compress_black_box,
                include_pseudo_weightedsep_final_stage=True,
            ),
            "single_tree_path_lengths": self.path_lengths_chain(
                X_df,
                representation="single_tree",
                compress=compress_single_tree,
                include_black_box=include_black_box_splits,
                compress_black_box=compress_black_box,
                include_pseudo_weightedsep_final_stage=True,
            ),
            "black_box_leaves_raw": self.black_box_number_of_leaves(
                representation="raw",
                compress=False,
                compress_chain=False,
            ),
            "black_box_leaves_dag": self.black_box_number_of_leaves(
                representation="dag",
                compress=compress_black_box,
                compress_chain=compress_dag,
            ),
            "black_box_leaves_single_tree": self.black_box_number_of_leaves(
                representation="single_tree",
                compress=compress_black_box,
                compress_chain=compress_single_tree,
            ),
        }

class MDTChain(_MDTTrainingMixin, _MDTCompressionMixin, _MDTAnalysisMixin):
    def __init__(
        self,
        *,
        training_method: str = "top_down",
        lambda_leaf: float = 0.001,
        eta_defer: float = 0.25,
        defer_lambda_strength: Optional[float] = None,
        depth_budget: int = 5,
        max_stages: int = 4,
        lookahead_k: int = 1,
        black_box_type: str = "xgboost",
        black_box_params: Optional[Dict[str, Any]] = None,
        threshold_guess_params: Optional[Dict[str, Any]] = None,
        same_binarization: bool = True,
        drop_no_signal_columns: bool = False,
        c: Optional[float] = None,
        gamma: float = 0.0,
        gamma_method: str = "normalized",
        projection: bool = False,
        set_weights_using_bb_leaves: bool = False,
        eliminate_black_box: bool = False,
        retrain_black_box: bool = False,
        correct_xgboost_scale: bool = False,
        backfit_mode: str = "none",
        backfit_sweeps: int = 1,
        eliminate_black_box_params: Optional[Dict[str, Any]] = None,
        max_total_stage_leaves: int = 300,
        max_expanded_leaves: int = 1_000_000,
        max_leaves_per_tree: int = 129,
        grouped_categories: Optional[List[List[int]]] = None,
        categorical_eps: float = CATEGORICAL_EPS_DEFAULT,
        placeholder: int = int(PLACEHOLDER_DEFAULT),
        out_of_fold_predictions: bool = False,
        out_of_fold_n_splits: int = 5,
        defer_tree_type: str = "licketysplit",
        sklearn_defer_tree_params: Optional[Dict[str, Any]] = None,
        claritree_params: Optional[Dict[str, Any]] = None,
        random_state: int = 0,
        task: Optional[str] = "classification",
        add_history: bool = False,
        add_extra_history: bool = False,
        standard_history_not_memoryless: bool = False,
    ):
        self.training_method = str(training_method).lower()
        if self.training_method not in {"top_down", "bottom_up"}:
            raise ValueError("training_method must be 'top_down' or 'bottom_up'")

        self.lambda_leaf = float(lambda_leaf)
        self.eta_defer = float(eta_defer)
        self.depth_budget = int(depth_budget)
        self.max_stages = int(max_stages)
        self.lookahead_k = int(lookahead_k)

        self.black_box_type = str(black_box_type).lower()
        self.black_box_params = dict(black_box_params or {})
        self.threshold_guess_params = dict(
            threshold_guess_params or {
                "n_estimators": 150,
                "max_depth": 2,
                "random_state": 0,
                "column_elimination": False,
            }
        )
        self.same_binarization = bool(same_binarization)
        self.drop_no_signal_columns = bool(drop_no_signal_columns)

        self.c = None if c is None else float(c)
        self.defer_lambda_strength = (
            None if defer_lambda_strength is None else float(defer_lambda_strength)
        )
        self.gamma = float(gamma)
        self.gamma_method = str(gamma_method).lower()
        if self.gamma_method not in {"normalized", "scaled"}:
            raise ValueError("gamma_method must be 'normalized' or 'scaled'")
        self.projection = bool(projection)
        self.set_weights_using_bb_leaves = bool(set_weights_using_bb_leaves)
                
        
        self.eliminate_black_box = bool(eliminate_black_box)
        self.eliminate_black_box_params = dict(eliminate_black_box_params or {})
        self.retrain_black_box = bool(retrain_black_box)
        self.correct_xgboost_scale = bool(correct_xgboost_scale)
        
        self.max_total_stage_leaves = 1000 if max_total_stage_leaves is None else int(max_total_stage_leaves)
        self.max_expanded_leaves = 1_000_000 if max_expanded_leaves is None else int(max_expanded_leaves)
        self.max_leaves_per_tree = None if max_leaves_per_tree is None else int(max_leaves_per_tree)
        
        self.placeholder = int(placeholder)
        self.random_state = int(random_state)

        self.encoder_: Optional[ThresholdGuessBinarizer] = None
        self.stage_encoders_: List[ThresholdGuessBinarizer] = []
        self.stages_: List[Any] = []
        self.final_black_box_: Optional[Any] = None
        self.final_train_eventual_deferral_mask_: Optional[np.ndarray] = None
        self.final_train_predictions_: Optional[np.ndarray] = None
        self.final_deferred_mask_: Optional[np.ndarray] = None
        self.X_columns_: Optional[List[str]] = None
        self.n_features_in_: Optional[int] = None
        self.n_chain_stages_: int = 0
        self.stop_reason_: Optional[str] = None

        self.grouped_categories = grouped_categories
        self.categorical_eps = categorical_eps

        self.add_history = bool(add_history)
        self.history_feature_columns_by_stage_: Dict[int, List[str]] = {}
        self.final_black_box_history_stage_idx_: Optional[int] = None
        self.add_extra_history = bool(add_extra_history)

        self.standard_history_not_memoryless = bool(
            standard_history_not_memoryless
        )

        if self.standard_history_not_memoryless and not self.add_history:
            raise ValueError(
                "standard_history_not_memoryless=True requires add_history=True."
            )
        
        if self.add_extra_history and not self.add_history:
            raise ValueError("add_extra_history=True requires add_history=True.")

        if self.add_extra_history and self.projection:
            raise ValueError("add_extra_history=True is not supported with projection=True.")


        self.out_of_fold_predictions = bool(out_of_fold_predictions)
        self.out_of_fold_n_splits = int(out_of_fold_n_splits)
        if self.out_of_fold_n_splits < 2:
            raise ValueError("out_of_fold_n_splits must be at least 2.")

        self.defer_tree_type = str(defer_tree_type).lower()
        if self.defer_tree_type not in {
            "weightedsep",
            "sklearn",
            "sklearn_cart",
            "cart",
            "licketysplit",
            "lickety",
            "ls",
            "claritree",
            "clari_tree",
            "clari",
        }:
            raise ValueError("defer_tree_type must be one of the valid options")

        self.sklearn_defer_tree_params = dict(sklearn_defer_tree_params or {})
        self.claritree_params = dict(claritree_params or {})

        self.backfit_mode = str(backfit_mode).lower()
        if self.backfit_mode not in {
            "none",
            "backward",
            "backwards",
            "forward",
            "forwards",
            "alternating",
            "alternate",
        }:
            raise ValueError(
                "backfit_mode must be one of: "
                "'none', 'backward', 'forward', or 'alternating'."
            )

        self.backfit_sweeps = int(backfit_sweeps)
        if self.backfit_sweeps < 0:
            raise ValueError("backfit_sweeps must be nonnegative.")

        self.n_classes_ = None
        self.is_binary_classification_ = None

        self.task = None if task is None else str(task).lower()
        if self.task is not None:
            aliases = {
                "classify": "classification",
                "classifier": "classification",
                "multiclass": "classification",
                "reg": "regression",
                "linear": "linear_regression",
                "linear_reg": "linear_regression",
                "claritree": "linear_regression",
            }
            self.task = aliases.get(self.task, self.task)

        if self.task not in {None, "classification", "regression", "linear_regression"}:
            raise ValueError("task must be None, 'classification', 'regression', or 'linear_regression'")

        if self.add_history and self.backfit_mode not in {"none"}:
            raise ValueError("add_history=True is not supported with backfitting yet.")
