from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _as_2d_f64(X, name: str = "X") -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D.")
    if not np.all(np.isfinite(X)):
        raise ValueError(f"{name} must be finite.")
    return np.ascontiguousarray(X)


def _as_2d_u8(X, name: str = "X") -> np.ndarray:
    X = np.asarray(X, dtype=np.uint8)
    if X.ndim != 2:
        raise ValueError(f"{name} must be 2D.")
    return np.ascontiguousarray(X)


def _as_1d_f64(v, n: int, name: str) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1 or v.shape[0] != n:
        raise ValueError(f"{name} must be shape (n_samples,).")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be finite.")
    return np.ascontiguousarray(v)


def _as_1d_i32_labels(y, n: int, name: str = "y", max_classes: int = 50) -> np.ndarray:
    y = np.asarray(y, dtype=np.int32)
    if y.ndim != 1 or y.shape[0] != n:
        raise ValueError(f"{name} must be shape (n_samples,).")
    if np.any(y < 0):
        raise ValueError(f"{name} labels must be nonnegative contiguous integers 0,1,2,...,C-1.")
    if y.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    max_y = int(y.max())
    n_classes = max_y + 1
    if n_classes > int(max_classes):
        raise ValueError(f"{name} has {n_classes} classes, but max_classes={max_classes}.")

    present = np.zeros(n_classes, dtype=bool)
    present[y] = True
    if not bool(np.all(present)):
        raise ValueError(f"{name} labels must be contiguous integers 0,1,2,...,C-1.")

    return np.ascontiguousarray(y)


def _as_1d_i32_pred(v, n: int, n_classes: int, name: str = "bb_pred") -> np.ndarray:
    v = np.asarray(v, dtype=np.int32)
    if v.ndim != 1 or v.shape[0] != n:
        raise ValueError(f"{name} must be shape (n_samples,).")
    if np.any(v < 0) or np.any(v >= int(n_classes)):
        raise ValueError(f"{name} values must be in 0,1,2,...,C-1.")
    return np.ascontiguousarray(v)


def _as_weights(sample_weight, n: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n, dtype=np.float64)

    w = np.asarray(sample_weight, dtype=np.float64)
    if w.ndim != 1 or w.shape[0] != n:
        raise ValueError("sample_weight must be shape (n_samples,).")
    if not np.all(np.isfinite(w)):
        raise ValueError("sample_weight must be finite.")
    if np.any(w < 0.0):
        raise ValueError("sample_weight must be nonnegative.")
    if not np.any(w > 0.0):
        raise ValueError("sample_weight must have positive total weight.")
    return np.ascontiguousarray(w)


def _huge_eta_no_defer_regression(y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
    y = np.asarray(y, dtype=np.float64)
    if sample_weight is None:
        w = np.ones_like(y, dtype=np.float64)
    else:
        w = np.asarray(sample_weight, dtype=np.float64)

    w_sum = float(np.sum(w))
    if w_sum <= 0.0:
        return 1e100

    y_bar = float(np.sum(w * y) / w_sum)
    tss = float(np.sum(w * (y - y_bar) ** 2))
    return max(1e100, 1000.0 * (tss + w_sum + 1.0))


@dataclass
class ConstantRegressorBlackBox:
    value: float

    def predict(self, X) -> np.ndarray:
        return np.full(len(X), float(self.value), dtype=np.float64)


@dataclass
class FixedPredictionRegressorBlackBox:
    predictions: np.ndarray

    def predict(self, X) -> np.ndarray:
        n = len(X)
        if n != len(self.predictions):
            raise ValueError(
                "This model was fit with a fixed regression prediction vector. "
                "For prediction on data with a different number of rows, pass bb_pred=..."
            )
        return np.asarray(self.predictions, dtype=np.float64).copy()


class LicketySPLITClassifierStage:
    """
    WeightedSeparable-style wrapper around licketysplit.LicketySPLITClassifier.

    supports multiclass labels encoded as contiguous integers 0,1,2,...,C-1.
    This expects threshold-binarized uint8 X.

    expected usage:
        stage = LicketySPLITClassifierStage()
        stage.fit(Xb, y, bb_pred=ref, lambda_leaf=..., eta_defer=...,
                  depth_budget=..., lookahead_k=..., sample_weights=w)
        raw = stage.predict(Xb, bb_pred=None, placeholder=99)
        deferred_mask = raw == 99
    """

    def __init__(
        self,
        *,
        lambda_leaf: float = 0.01,
        eta_defer: Optional[float] = None,
        depth_budget: int = 3,
        lookahead_k: int = 1,
        cache_mode="fingerprint",
        cost_caching_enabled: bool = True,
        max_classes: int = 50,
        placeholder: int = 99,
    ):
        self.lambda_leaf = float(lambda_leaf)
        self.eta_defer = eta_defer
        self.depth_budget = int(depth_budget)
        self.lookahead_k = int(lookahead_k)
        self.cache_mode = cache_mode
        self.cost_caching_enabled = bool(cost_caching_enabled)
        self.max_classes = int(max_classes)
        self.placeholder = int(placeholder)

        self._model = None
        self.objective_ = None
        self.uses_defer_ = False

    def fit(
        self,
        X,
        y,
        bb_pred=None,
        *,
        lambda_leaf=None,
        eta_defer=None,
        depth_budget=None,
        lookahead_k=None,
        sample_weight=None,
        sample_weights=None,
    ):
        from licketysplit import LicketySPLITClassifier

        X_arr = _as_2d_u8(X)
        n = int(X_arr.shape[0])
        y_arr = _as_1d_i32_labels(y, n, "y", max_classes=self.max_classes)

        if sample_weight is not None and sample_weights is not None:
            raise ValueError("Pass only one of sample_weight or sample_weights.")
        if sample_weight is None:
            sample_weight = sample_weights
        w_arg = None if sample_weight is None else _as_weights(sample_weight, n)

        lam = self.lambda_leaf if lambda_leaf is None else float(lambda_leaf)
        eta = self.eta_defer if eta_defer is None else eta_defer
        depth = self.depth_budget if depth_budget is None else int(depth_budget)
        k = self.lookahead_k if lookahead_k is None else int(lookahead_k)

        self.classes_ = np.arange(int(y_arr.max()) + 1, dtype=np.int32)
        self.n_classes_ = int(len(self.classes_))

        if int(self.placeholder) < self.n_classes_:
            raise ValueError(
                f"placeholder={self.placeholder} overlaps with class labels 0..{self.n_classes_ - 1}."
            )

        self._model = LicketySPLITClassifier(
            lambda_leaf=lam,
            depth_budget=depth,
            lookahead_k=k,
            eta_defer=eta,
            cache_mode=self.cache_mode,
            cost_caching_enabled=self.cost_caching_enabled,
        )

        if bb_pred is None:
            self.uses_defer_ = False
            obj = self._model.fit(X_arr, y_arr, sample_weight=w_arg)
        else:
            ref = _as_1d_i32_pred(bb_pred, n, self.n_classes_, "bb_pred")
            self.uses_defer_ = True
            obj = self._model.fit(X_arr, y_arr, bb_pred=ref, sample_weight=w_arg)

        self.objective_ = float(
            getattr(self._model, "objective_", obj if obj is not None else self._model.last_objective)
        )
        self.n_features_in_ = int(X_arr.shape[1])
        return self

    def predict(self, X, *, bb_pred=None, placeholder=None) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fit.")

        X_arr = _as_2d_u8(X)
        n = int(X_arr.shape[0])

        if placeholder is None:
            placeholder = self.placeholder
        placeholder = int(placeholder)

        if placeholder < getattr(self, "n_classes_", 0):
            raise ValueError(
                f"placeholder={placeholder} overlaps with class labels 0..{self.n_classes_ - 1}."
            )

        if bb_pred is None:
            return self.predict_placeholder(X_arr, placeholder=placeholder)

        ref = _as_1d_i32_pred(bb_pred, n, self.n_classes_, "bb_pred")
        return np.asarray(self._model.predict(X_arr, bb_pred=ref), dtype=np.int32)

    def predict_placeholder(self, X, placeholder=None) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fit.")

        X_arr = _as_2d_u8(X)

        if placeholder is None:
            placeholder = self.placeholder
        placeholder = int(placeholder)

        if placeholder < getattr(self, "n_classes_", 0):
            raise ValueError(
                f"placeholder={placeholder} overlaps with class labels 0..{self.n_classes_ - 1}."
            )

        return np.asarray(
            self._model.predict_placeholder(X_arr, placeholder=placeholder),
            dtype=np.int32,
        )

    def deferred_mask(self, X, *, placeholder=None) -> np.ndarray:
        raw = self.predict(X, bb_pred=None, placeholder=placeholder)
        if placeholder is None:
            placeholder = self.placeholder
        return raw == int(placeholder)

    def get_tree_paths(self):
        if self._model is None:
            raise RuntimeError("Model has not been fit.")
        return self._model.get_tree_paths()

    def leaf_actions(self):
        if self._model is None:
            raise RuntimeError("Model has not been fit.")
        return list(self._model.leaf_actions())

    def leaf_counts(self):
        if self._model is None:
            raise RuntimeError("Model has not been fit.")
        return dict(self._model.leaf_counts())

    def split_counts(self, X) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fit.")
        return np.asarray(self._model.split_counts(_as_2d_u8(X)), dtype=np.int32)

    def n_leaves(self) -> int:
        counts = self.leaf_counts()
        return int(counts.get("total", len(self.leaf_actions())))




class LicketySPLITRegressorStage:
    """
    WeightedSeparable-style wrapper around licketysplit.LicketySPLITRegressor.

    expected usage:
        stage = LicketySPLITRegressorStage()
        stage.fit(Xb, y, bb_pred=ref, lambda_leaf=..., eta_defer=...,
                  depth_budget=..., lookahead_k=..., sample_weights=w)
        raw = stage.predict(Xb, bb_pred=None, placeholder=np.nan)
        deferred_mask = np.isnan(raw)
    """

    def __init__(
        self,
        *,
        lambda_leaf: float = 0.01,
        eta_defer: Optional[float] = None,
        depth_budget: int = 3,
        lookahead_k: int = 1,
        cache_mode="fingerprint",
        cost_caching_enabled: bool = True,
    ):
        self.lambda_leaf = float(lambda_leaf)
        self.eta_defer = eta_defer
        self.depth_budget = int(depth_budget)
        self.lookahead_k = int(lookahead_k)
        self.cache_mode = cache_mode
        self.cost_caching_enabled = bool(cost_caching_enabled)

        self._model = None
        self.objective_ = None
        self.uses_defer_ = False

    def fit(
        self,
        X,
        y,
        bb_pred=None,
        *,
        lambda_leaf=None,
        eta_defer=None,
        depth_budget=None,
        lookahead_k=None,
        sample_weight=None,
        sample_weights=None,
    ):
        from licketysplit import LicketySPLITRegressor

        X_arr = _as_2d_u8(X)
        n = int(X_arr.shape[0])
        y_arr = _as_1d_f64(y, n, "y")

        if sample_weight is not None and sample_weights is not None:
            raise ValueError("Pass only one of sample_weight or sample_weights.")
        if sample_weight is None:
            sample_weight = sample_weights
        w_arg = None if sample_weight is None else _as_weights(sample_weight, n)

        lam = self.lambda_leaf if lambda_leaf is None else float(lambda_leaf)
        eta = self.eta_defer if eta_defer is None else eta_defer
        depth = self.depth_budget if depth_budget is None else int(depth_budget)
        k = self.lookahead_k if lookahead_k is None else int(lookahead_k)

        self._model = LicketySPLITRegressor(
            lambda_leaf=lam,
            depth_budget=depth,
            lookahead_k=k,
            eta_defer=eta,
            cache_mode=self.cache_mode,
            cost_caching_enabled=self.cost_caching_enabled,
        )

        if bb_pred is None:
            self.uses_defer_ = False
            obj = self._model.fit(X_arr, y_arr, sample_weight=w_arg)
        else:
            ref = _as_1d_f64(bb_pred, n, "bb_pred")
            self.uses_defer_ = True
            obj = self._model.fit(X_arr, y_arr, bb_pred=ref, sample_weight=w_arg)

        self.objective_ = float(getattr(self._model, "objective_", obj if obj is not None else self._model.last_objective))
        self.n_features_in_ = int(X_arr.shape[1])
        return self

    def predict(self, X, *, bb_pred=None, placeholder=np.nan) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fit.")

        X_arr = _as_2d_u8(X)
        n = int(X_arr.shape[0])

        if bb_pred is None:
            return self.predict_placeholder(X_arr, placeholder=placeholder)

        ref = _as_1d_f64(bb_pred, n, "bb_pred")
        return np.asarray(self._model.predict(X_arr, bb_pred=ref), dtype=np.float64)

    def predict_placeholder(self, X, placeholder=np.nan) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fit.")
        X_arr = _as_2d_u8(X)
        return np.asarray(self._model.predict_placeholder(X_arr, placeholder=float(placeholder)), dtype=np.float64)

    def get_tree_paths(self):
        if self._model is None:
            raise RuntimeError("Model has not been fit.")
        return self._model.get_tree_paths()

    def leaf_actions(self):
        if self._model is None:
            raise RuntimeError("Model has not been fit.")
        return list(self._model.leaf_actions())

    def leaf_counts(self):
        if self._model is None:
            raise RuntimeError("Model has not been fit.")
        return dict(self._model.leaf_counts())

    def leaf_values(self):
        if self._model is None:
            raise RuntimeError("Model has not been fit.")
        return list(self._model.leaf_values())

    def split_counts(self, X) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fit.")
        return np.asarray(self._model.split_counts(_as_2d_u8(X)), dtype=np.int32)

    def n_leaves(self) -> int:
        return int(self.leaf_counts().get("total", len(self.leaf_actions())))


class CLARITreeRegressorStage:
    """
    WeightedSeparable-style wrapper around clari_tree.CLARITree.

    this expects raw numeric X, not threshold-binarized X, because CLARITree can use
    continuous columns for linear/ridge leaves.
    """

    def __init__(
        self,
        *,
        kappa: float = 1e-3,
        lambda_leaf: float = 0.01,
        rho: float = 0.0,
        eta_defer: Optional[float] = 0.0,
        depth_budget: int = 3,
        n_thresholds: int = 20,
        thresholds_strategy: str = "quantile",
        verbose: bool = False,
        min_leaf_node_size: int = 0,
    ):
        self.kappa = float(kappa)
        self.lambda_leaf = float(lambda_leaf)
        self.rho = float(rho)
        self.eta_defer = eta_defer
        self.depth_budget = int(depth_budget)
        self.n_thresholds = int(n_thresholds)
        self.thresholds_strategy = str(thresholds_strategy)
        self.verbose = bool(verbose)
        self.min_leaf_node_size = int(min_leaf_node_size)
        self.categorical_idx = []

        self._model = None
        self.objective_ = None
        self.uses_defer_ = False

    def fit(
        self,
        X,
        y,
        bb_pred=None,
        *,
        lambda_leaf=None,
        eta_defer=None,
        depth_budget=None,
        lookahead_k=None, # accepted for interface compatibility; CLARITree ignores it
        sample_weight=None,
        sample_weights=None,
    ):
        from .clari_tree import CLARITree

        X_arr = _as_2d_f64(X)
        n = int(X_arr.shape[0])
        y_arr = _as_1d_f64(y, n, "y")

        if sample_weight is not None and sample_weights is not None:
            raise ValueError("Pass only one of sample_weight or sample_weights.")
        if sample_weight is None:
            sample_weight = sample_weights
        w_arr = _as_weights(sample_weight, n)

        lam = self.lambda_leaf if lambda_leaf is None else float(lambda_leaf)
        eta = self.eta_defer if eta_defer is None else eta_defer
        depth = self.depth_budget if depth_budget is None else int(depth_budget)
        cat = self.categorical_idx

        if bb_pred is None:
            ref = y_arr.copy()
            eta_fit = _huge_eta_no_defer_regression(y_arr, w_arr)
            self.uses_defer_ = False
        else:
            ref = _as_1d_f64(bb_pred, n, "bb_pred")
            eta_fit = 0.0 if eta is None else float(eta)
            self.uses_defer_ = True

        self._model = CLARITree(
            self.kappa,
            depth,
            lam,
            self.rho,
            eta_fit,
            self.n_thresholds,
            self.thresholds_strategy,
            self.verbose,
            self.min_leaf_node_size,
        )

        self.objective_ = float(
            self._model.fit_with_reference_and_weights(
                X_arr,
                y_arr,
                ref,
                w_arr,
                cat,
            )
        )
        self.n_features_in_ = int(X_arr.shape[1])
        self.eta_defer_used_ = float(eta_fit)
        return self

    def predict(self, X, *, bb_pred=None, placeholder=np.nan) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fit.")

        X_arr = _as_2d_f64(X)
        n = int(X_arr.shape[0])

        if bb_pred is None:
            return self.predict_placeholder(X_arr, placeholder=placeholder)

        ref = _as_1d_f64(bb_pred, n, "bb_pred")
        return np.asarray(self._model.predict_with_reference(X_arr, ref), dtype=np.float64)

    def predict_placeholder(self, X, placeholder=np.nan) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model has not been fit.")

        X_arr = _as_2d_f64(X)
        dummy_ref = np.full(X_arr.shape[0], float(placeholder), dtype=np.float64)
        return np.asarray(self._model.predict_with_reference(X_arr, dummy_ref), dtype=np.float64)

    def export_leaf_paths(self):
        if self._model is None:
            raise RuntimeError("Model has not been fit.")
        return list(self._model.export_leaf_paths())

    def get_tree_paths(self):
        leaves = self.export_leaf_paths()
        paths = []
        values = []

        for leaf in leaves:
            path = []
            for cond in leaf.get("conditions", []):
                f1 = int(cond["feature_idx"]) + 1
                path.append(+f1 if cond["op"] == "<=" else -f1)

            paths.append(path)

            leaf_type = leaf.get("leaf_type")
            if leaf_type == "CONSTANT":
                values.append(float(leaf.get("prediction", np.nan)))
            else:
                # DEFER and LINEAR are both not representable by a scalar leaf value here.
                # Use leaf_actions/export_leaf_paths to distinguish them.
                values.append(np.nan)

        return paths, values

    def leaf_actions(self):
        leaves = self.export_leaf_paths()
        actions = []
        for leaf in leaves:
            leaf_type = leaf.get("leaf_type")
            if leaf_type == "DEFER":
                actions.append(-1)
            elif leaf_type == "CONSTANT":
                actions.append(0)
            elif leaf_type == "LINEAR":
                actions.append(1)
            else:
                actions.append(99)
        return actions

    def leaf_counts(self):
        actions = self.leaf_actions()
        return {
            "constant": int(sum(a == 0 for a in actions)),
            "linear": int(sum(a == 1 for a in actions)),
            "defer": int(sum(a == -1 for a in actions)),
            "total": int(len(actions)),
        }

    def split_counts(self, X) -> np.ndarray:
        X_arr = _as_2d_f64(X)
        leaves = self.export_leaf_paths()
        out = np.zeros(X_arr.shape[0], dtype=np.int32)

        for i in range(X_arr.shape[0]):
            depth_for_row = 0
            for leaf in leaves:
                ok = True
                depth = 0
                for cond in leaf.get("conditions", []):
                    j = int(cond["feature_idx"])
                    thr = float(cond["threshold"])
                    op = cond["op"]
                    depth += 1

                    if op == "<=":
                        if not (X_arr[i, j] <= thr):
                            ok = False
                            break
                    else:
                        if not (X_arr[i, j] > thr):
                            ok = False
                            break

                if ok:
                    depth_for_row = depth
                    break

            out[i] = depth_for_row

        return out

    def n_leaves(self) -> int:
        return int(self.leaf_counts()["total"])
