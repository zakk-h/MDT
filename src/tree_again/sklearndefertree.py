import numpy as np
from sklearn.tree import DecisionTreeClassifier


class SklearnDeferTree:
    # train a standard CART tree, then relabel each terminal leaf as either:
    #  - a prediction leaf (weighted majority class of training points in that leaf), or
    #  - a defer leaf (use externally supplied fallback predictions).
    # defer if weighted_bb_error_rate + eta < weighted_leaf_error_rate
    
    def __init__(
        self,
        max_depth=None,
        min_samples_leaf=1,
        random_state=None,
        criterion="gini",
        eta=0.0,
        ccp_alpha=0.0,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.criterion = criterion
        self.eta = eta
        self.ccp_alpha = ccp_alpha

        self.tree_ = None
        self.leaf_actions_ = {} # leaf_id -> "predict" or "defer"
        self.leaf_labels_ = {} # leaf_id -> majority class label
        self.leaf_weight_sums_ = {} # leaf_id -> weighted n_leaf
        self.leaf_error_sums_ = {} # leaf_id -> weighted CART leaf error
        self.leaf_bb_error_sums_ = {} # leaf_id -> weighted BB error
        self.classes_ = None
        self.is_fitted_ = False

    def fit(self, X, y, bb_pred, sample_weights=None, **kwargs):
        X = np.asarray(X)
        y = np.asarray(y)
        bb_pred = np.asarray(bb_pred)

        if len(X) != len(y) or len(y) != len(bb_pred):
            raise ValueError("X, y, and bb_pred must have the same number of rows.")

        if sample_weights is None:
            sample_weights = np.ones(len(y), dtype=np.float64)
        else:
            sample_weights = np.asarray(sample_weights, dtype=np.float64)

        if sample_weights.ndim != 1 or sample_weights.shape[0] != len(y):
            raise ValueError("sample_weights must be shape (n_samples,).")
        if np.any(sample_weights < 0):
            raise ValueError("sample_weights must be nonnegative.")
        if not np.all(np.isfinite(sample_weights)):
            raise ValueError("sample_weights must be finite.")

        self.tree_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            criterion=self.criterion,
            ccp_alpha=self.ccp_alpha,
        )

        self.tree_.fit(X, y, sample_weight=sample_weights)

        self.classes_ = np.unique(y)

        leaf_ids = self.tree_.apply(X)
        unique_leaf_ids = np.unique(leaf_ids)

        self.leaf_actions_.clear()
        self.leaf_labels_.clear()
        self.leaf_weight_sums_.clear()
        self.leaf_error_sums_.clear()
        self.leaf_bb_error_sums_.clear()

        for leaf_id in unique_leaf_ids:
            mask = leaf_ids == leaf_id
            y_leaf = y[mask]
            bb_leaf = bb_pred[mask]
            w_leaf = sample_weights[mask]

            weighted_n_leaf = float(w_leaf.sum())
            self.leaf_weight_sums_[leaf_id] = weighted_n_leaf

            # if a leaf has zero total weight somehow
            if weighted_n_leaf <= 0.0:
                values, counts = np.unique(y_leaf, return_counts=True)
                majority_label = values[np.argmax(counts)]
                self.leaf_labels_[leaf_id] = majority_label
                self.leaf_error_sums_[leaf_id] = 0.0
                self.leaf_bb_error_sums_[leaf_id] = 0.0
                self.leaf_actions_[leaf_id] = "predict"
                continue

            # weighted-majority label in this leaf.
            values = np.unique(y_leaf)
            weighted_counts = np.array(
                [w_leaf[y_leaf == value].sum() for value in values],
                dtype=np.float64,
            )
            majority_label = values[int(np.argmax(weighted_counts))]
            self.leaf_labels_[leaf_id] = majority_label

            # weighted empirical leaf prediction error.
            leaf_error_sum = float(w_leaf[y_leaf != majority_label].sum())

            # weighted empirical black-box error on same leaf samples.
            bb_error_sum = float(w_leaf[y_leaf != bb_leaf].sum())

            self.leaf_error_sums_[leaf_id] = leaf_error_sum
            self.leaf_bb_error_sums_[leaf_id] = bb_error_sum

            # post-hoc defer decision.
            if bb_error_sum + float(self.eta) * weighted_n_leaf < leaf_error_sum:
                self.leaf_actions_[leaf_id] = "defer"
            else:
                self.leaf_actions_[leaf_id] = "predict"

        self.is_fitted_ = True
        return self

    def predict_with_deferral(self, X, bb_pred):
        
        # predict on X, using bb_pred whenever a point lands in a defer leaf. also returns which points are deferred
        self._check_is_fitted()

        X = np.asarray(X)
        bb_pred = np.asarray(bb_pred)

        if len(X) != len(bb_pred):
            raise ValueError("X and bb_pred must have the same number of rows.")

        leaf_ids = self.tree_.apply(X)
        y_pred = np.empty(len(X), dtype=bb_pred.dtype)
        deferred_mask = np.zeros(len(X), dtype=bool)

        for i, leaf_id in enumerate(leaf_ids):
            action = self.leaf_actions_.get(leaf_id, "predict")
            if action == "defer":
                y_pred[i] = bb_pred[i]
                deferred_mask[i] = True
            else:
                y_pred[i] = self.leaf_labels_[leaf_id]

        return y_pred, deferred_mask

    def predict(self, X, bb_pred=None, placeholder=None):
        # can include placeholder instead of real prediction
        self._check_is_fitted()

        X = np.asarray(X)
        leaf_ids = self.tree_.apply(X)

        if bb_pred is not None:
            bb_pred = np.asarray(bb_pred)
            if len(X) != len(bb_pred):
                raise ValueError("X and bb_pred must have the same number of rows.")

            y_pred = np.empty(len(X), dtype=bb_pred.dtype)

            for i, leaf_id in enumerate(leaf_ids):
                action = self.leaf_actions_.get(leaf_id, "predict")
                if action == "defer":
                    y_pred[i] = bb_pred[i]
                else:
                    y_pred[i] = self.leaf_labels_[leaf_id]

            return y_pred

        defer_value = -1 if placeholder is None else placeholder
        out = np.empty(len(X), dtype=np.int64)

        for i, leaf_id in enumerate(leaf_ids):
            action = self.leaf_actions_.get(leaf_id, "predict")
            if action == "defer":
                out[i] = defer_value
            else:
                out[i] = int(self.leaf_labels_[leaf_id])

        return out

    def predict_defer_mask(self, X):
        self._check_is_fitted()

        X = np.asarray(X)
        leaf_ids = self.tree_.apply(X)

        return np.array(
            [self.leaf_actions_.get(leaf_id, "predict") == "defer" for leaf_id in leaf_ids],
            dtype=bool,
        )

    def predict_leaf_actions(self, X):
        self._check_is_fitted()

        X = np.asarray(X)
        leaf_ids = self.tree_.apply(X)

        return np.array(
            [self.leaf_actions_.get(leaf_id, "predict") for leaf_id in leaf_ids],
            dtype=object,
        )

    def deferral_rate(self, X):
        return float(self.predict_defer_mask(X).mean())

    def score(self, X, y, bb_pred):
        y = np.asarray(y)
        y_pred = self.predict(X, bb_pred=bb_pred)
        return float(np.mean(y_pred == y))

    def get_tree_paths(self):
        self._check_is_fitted()

        tree_ = self.tree_.tree_
        children_left = tree_.children_left
        children_right = tree_.children_right
        features = tree_.feature
        thresholds = tree_.threshold

        decoded_paths = []
        actions = []

        def rec(node_id, path):
            left_id = int(children_left[node_id])
            right_id = int(children_right[node_id])

            if left_id == -1 and right_id == -1:
                decoded_paths.append(list(path))

                action = self.leaf_actions_.get(node_id, "predict")
                if action == "defer":
                    actions.append(-1)
                else:
                    actions.append(int(self.leaf_labels_.get(node_id, 0)))
                return

            feat = int(features[node_id])
            thr = float(thresholds[node_id])

            # left branch: x_f <= threshold
            path.append((feat, thr, True))
            rec(left_id, path)
            path.pop()

            # right branch: x_f > threshold
            path.append((feat, thr, False))
            rec(right_id, path)
            path.pop()

        rec(0, [])
        return decoded_paths, actions

    def leaf_count(self):
        self._check_is_fitted()
        return int(len(self.leaf_actions_))

    def defer_leaf_count(self):
        self._check_is_fitted()
        return int(sum(action == "defer" for action in self.leaf_actions_.values()))

    def leaf_statistics(self, X, y, bb_pred, sample_weights=None):
        self._check_is_fitted()

        X = np.asarray(X)
        y = np.asarray(y)
        bb_pred = np.asarray(bb_pred)

        if len(X) != len(y) or len(y) != len(bb_pred):
            raise ValueError("X, y, and bb_pred must have the same number of rows.")

        if sample_weights is None:
            sample_weights = np.ones(len(y), dtype=np.float64)
        else:
            sample_weights = np.asarray(sample_weights, dtype=np.float64)

        if sample_weights.ndim != 1 or sample_weights.shape[0] != len(y):
            raise ValueError("sample_weights must be shape (n_samples,).")

        leaf_ids = self.tree_.apply(X)
        stats = []

        for leaf_id in np.unique(leaf_ids):
            mask = leaf_ids == leaf_id
            y_leaf = y[mask]
            bb_leaf = bb_pred[mask]
            w_leaf = sample_weights[mask]

            weighted_n_leaf = float(w_leaf.sum())
            majority_label = self.leaf_labels_.get(leaf_id, None)
            action = self.leaf_actions_.get(leaf_id, "predict")

            if weighted_n_leaf > 0.0 and majority_label is not None:
                leaf_error_sum = float(w_leaf[y_leaf != majority_label].sum())
                bb_error_sum = float(w_leaf[y_leaf != bb_leaf].sum())
                leaf_error = leaf_error_sum / weighted_n_leaf
                bb_error = bb_error_sum / weighted_n_leaf
            else:
                leaf_error_sum = 0.0
                bb_error_sum = 0.0
                leaf_error = np.nan
                bb_error = np.nan

            stats.append({
                "leaf_id": int(leaf_id),
                "n_leaf": int(mask.sum()),
                "weighted_n_leaf": float(weighted_n_leaf),
                "action": action,
                "majority_label": majority_label,
                "leaf_error": float(leaf_error),
                "bb_error": float(bb_error),
                "leaf_error_sum": float(leaf_error_sum),
                "bb_error_sum": float(bb_error_sum),
                "defer_cost_sum": float(self.eta) * float(weighted_n_leaf),
            })

        return stats

    def _check_is_fitted(self):
        if not self.is_fitted_:
            raise ValueError("This SklearnDeferTree instance is not fitted yet.")