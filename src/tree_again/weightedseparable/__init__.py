import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from ._core import WeightedSeparable as _WeightedSeparableCore
from ._threshold_guessing import ThresholdGuessBinarizer

__all__ = ["WeightedSeparable", "ThresholdGuessBinarizer"]

_PLACEHOLDER_DEFAULT = np.uint8(99)


def _as_u8_X(X) -> np.ndarray:
    X = np.asarray(X, dtype=np.uint8)
    if X.ndim != 2:
        raise ValueError("X must be 2D (n_samples, n_features)")
    return X


def _as_i32_y(y, n: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.int32)
    if y.ndim != 1:
        raise ValueError("y must be 1D (n_samples,)")
    if y.shape[0] != n:
        raise ValueError("y length must match X rows")
    return y


def _as_u8_bb(bb_pred, n: int) -> np.ndarray:
    bb_pred = (np.asarray(bb_pred, dtype=np.uint8) & 1)
    if bb_pred.ndim != 1:
        raise ValueError("bb_pred must be 1D (n_samples,)")
    if bb_pred.shape[0] != n:
        raise ValueError("bb_pred length must match X rows")
    return bb_pred


def _as_f64_weights(sample_weights, n: int) -> np.ndarray:
    sample_weights = np.asarray(sample_weights, dtype=np.float64)
    if sample_weights.ndim != 1:
        raise ValueError("sample_weights must be 1D (n_samples,)")
    if sample_weights.shape[0] != n:
        raise ValueError("sample_weights length must match X rows")
    if not np.all(np.isfinite(sample_weights)):
        raise ValueError("sample_weights must be finite")
    if np.any(sample_weights < 0):
        raise ValueError("sample_weights must be nonnegative")
    return sample_weights


class WeightedSeparable:
    def __init__(self):
        self._model = _WeightedSeparableCore()
        self.objective_: float | None = None

    def fit(
        self,
        X,
        y,
        bb_pred,
        *,
        lambda_leaf: float,
        eta_defer: float,
        depth_budget: int,
        lookahead_k: int = 1,
        sample_weights=None,
    ) -> "WeightedSeparable":
        X = _as_u8_X(X)
        n = int(X.shape[0])
        y = _as_i32_y(y, n)
        bb_pred = _as_u8_bb(bb_pred, n)

        weights_arg = None
        if sample_weights is not None:
            weights_arg = _as_f64_weights(sample_weights, n)

        obj = self._model.fit(
            X,
            y,
            bb_pred,
            float(lambda_leaf),
            float(eta_defer),
            int(depth_budget),
            int(lookahead_k),
            weights_arg,
        )
        self.objective_ = float(obj)
        return self

    def fit_return(self, *args, **kwargs):
        self.fit(*args, **kwargs)
        return float(self.objective_ if self.objective_ is not None else self.last_objective)

    def predict(self, X, bb_pred=None, *, placeholder: int = int(_PLACEHOLDER_DEFAULT)) -> np.ndarray:
        X = _as_u8_X(X)
        n = int(X.shape[0])

        if bb_pred is None:
            return self._model.predict_placeholder(X, int(placeholder))

        bb_pred = _as_u8_bb(bb_pred, n)
        return self._model.predict(X, bb_pred, int(placeholder))

    def routing_stats(self, X, *, placeholder: int = int(_PLACEHOLDER_DEFAULT)) -> dict:
        raw = self.predict(X, bb_pred=None, placeholder=placeholder).astype(np.uint8, copy=False)
        is_def = raw == np.uint8(placeholder)
        n = int(raw.size)
        n_defer = int(is_def.sum())
        n_handled = n - n_defer
        return {
            "N": n,
            "n_defer": n_defer,
            "n_handled": n_handled,
            "frac_defer": (n_defer / n) if n else 0.0,
            "frac_handled": (n_handled / n) if n else 0.0,
        }

    @property
    def last_objective(self) -> float:
        return float(self._model.last_objective())

    def leaf_counts(self) -> dict:
        return dict(self._model.leaf_counts_single_tree())

    def split_counts(self, X) -> np.ndarray:
        X = _as_u8_X(X)
        return np.asarray(self._model.split_counts_single_tree(X), dtype=np.int32)

    def leaf_paths(self) -> list[list[int]]:
        return list(self._model.leaf_paths_single_tree())

    def leaf_actions(self) -> list[int]:
        return list(self._model.leaf_actions_single_tree())

    def get_tree_paths(self):
        paths, actions = self._model.leaf_paths_and_actions_single_tree()
        return paths, actions

    def plot_tree(self, feature_names=None, figsize=(10, 6), ax=None, title=None, show=True):
        paths, actions = self.get_tree_paths()

        class Node:
            __slots__ = ("feature", "left", "right", "action")

            def __init__(self):
                self.feature = None
                self.left = None
                self.right = None
                self.action = None

        def build_tree_from_paths(paths, actions):
            root = Node()
            for path, act in zip(paths, actions):
                cur = root
                for signed_f in path:
                    f = abs(signed_f) - 1
                    go_left = signed_f > 0
                    if cur.feature is None:
                        cur.feature = f
                        cur.left = Node()
                        cur.right = Node()
                    cur = cur.left if go_left else cur.right
                cur.action = int(act)
            return root

        def collect_leaves_in_order(node, leaves):
            if node is None:
                return
            if node.action is not None or (node.left is None and node.right is None):
                leaves.append(node)
                return
            collect_leaves_in_order(node.left, leaves)
            collect_leaves_in_order(node.right, leaves)

        def assign_positions_tree(root):
            positions = {}
            leaves = []
            collect_leaves_in_order(root, leaves)
            if not leaves:
                leaves = [root]
            leaf_x = {leaf: i for i, leaf in enumerate(leaves)}

            def dfs(node, depth):
                if node is None:
                    return
                if node.action is not None or (node.left is None and node.right is None):
                    x = leaf_x[node]
                    positions[node] = (x, -depth)
                    return
                dfs(node.left, depth + 1)
                dfs(node.right, depth + 1)
                x_left, _ = positions[node.left]
                x_right, _ = positions[node.right]
                positions[node] = (0.5 * (x_left + x_right), -depth)

            dfs(root, 0)
            return positions, len(leaves)

        def _shrink_segment(x1, y1, x2, y2, r1, r2):
            dx = x2 - x1
            dy = y2 - y1
            dist = (dx * dx + dy * dy) ** 0.5
            if dist == 0:
                return x1, y1, x2, y2
            ux = dx / dist
            uy = dy / dist
            return (x1 + ux * r1, y1 + uy * r1, x2 - ux * r2, y2 - uy * r2)

        if feature_names is None:
            enc = [abs(v) for p in paths for v in p]
            max_f = (max(enc) - 1) if enc else -1
            feature_names = [f"f{j}" for j in range(max_f + 1)]

        root = build_tree_from_paths(paths, actions)
        pos_local, _ = assign_positions_tree(root)

        x_scale = 4.6
        y_scale = 3.4

        pos = {}
        xs = []
        ys = []
        for node, (x, y) in pos_local.items():
            xx = x * x_scale
            yy = y * y_scale
            pos[node] = (xx, yy)
            xs.append(xx)
            ys.append(yy)

        minx, maxx = min(xs), max(xs)
        x_center = 0.5 * (minx + maxx)
        for node, (xx, yy) in list(pos.items()):
            pos[node] = (xx - x_center, yy)

        if ax is None:
            all_nodes_xy = list(pos.values())
            xs = [p[0] for p in all_nodes_xy] if all_nodes_xy else [0.0]
            ys = [p[1] for p in all_nodes_xy] if all_nodes_xy else [0.0]
            total_w = (max(xs) - min(xs)) if xs else 10.0
            total_h = (max(ys) - min(ys)) if ys else 6.0
            fig_w = max(figsize[0], total_w / 2.2)
            fig_h = max(figsize[1], total_h / 1.6)
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140)
        else:
            fig = ax.figure

        ax.set_axis_off()

        internal_r = 0.46
        leaf_r = 0.56
        SPLIT_FS = 16
        SPLIT_OFFSET_PTS = 12

        def _feat_name(f: int) -> str:
            if feature_names is not None and f < len(feature_names):
                return str(feature_names[f])
            return f"f{f}"

        def draw(node):
            x, y = pos[node]

            if node.feature is not None:
                if node.left is not None:
                    x2, y2 = pos[node.left]
                    r_child = leaf_r if node.left.action is not None else internal_r
                    sx, sy, ex, ey = _shrink_segment(x, y, x2, y2, internal_r, r_child)
                    ax.add_line(Line2D([sx, ex], [sy, ey], color="black", linewidth=2.2, zorder=1))
                    draw(node.left)

                if node.right is not None:
                    x2, y2 = pos[node.right]
                    r_child = leaf_r if node.right.action is not None else internal_r
                    sx, sy, ex, ey = _shrink_segment(x, y, x2, y2, internal_r, r_child)
                    ax.add_line(Line2D([sx, ex], [sy, ey], color="black", linewidth=2.2, zorder=1))
                    draw(node.right)

            if node.action is None:
                face = "#ddeeff"
                radius = internal_r
                circ = Circle((x, y), radius, facecolor=face, edgecolor="black", linewidth=1.6, zorder=10)
                ax.add_patch(circ)

                if node.feature is not None:
                    name = _feat_name(int(node.feature))
                    ax.annotate(
                        name,
                        xy=(x, y),
                        xycoords="data",
                        xytext=(0, SPLIT_OFFSET_PTS),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=SPLIT_FS,
                        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.98),
                        zorder=10000,
                        clip_on=False,
                    )
            else:
                if node.action == -1:
                    face = "#fff2cc"
                    label = "D"
                else:
                    face = "#e0ffd8"
                    label = str(int(node.action))

                circ = Circle((x, y), leaf_r, facecolor=face, edgecolor="black", linewidth=1.6, zorder=10)
                ax.add_patch(circ)
                ax.text(
                    x,
                    y,
                    label,
                    ha="center",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                    zorder=10000,
                    clip_on=False,
                )

        draw(root)

        all_xy = list(pos.values())
        xs, ys = zip(*all_xy) if all_xy else ([0.0], [0.0])

        pad_x = 3.0
        pad_y = 4.0
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
        ax.margins(x=0.08, y=0.12)

        for t in ax.texts:
            t.set_clip_on(False)

        ax.set_aspect("equal", adjustable="box")
        ax.set_axis_off()
        ax.set_title("PRAXIS WeightedSeparable Tree" if title is None else str(title), fontsize=16, pad=12)

        if show:
            plt.show()
        return fig, ax