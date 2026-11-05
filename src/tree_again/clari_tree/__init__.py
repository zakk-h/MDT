from ._core import CLARITree, CLARITreeConst, Greedy, GreedyConst


def _format_feature(feature_idx, feature_names=None):
    if feature_names is None:
        return f"x_{feature_idx}"

    if feature_idx < len(feature_names):
        return str(feature_names[feature_idx])

    raw_idx = feature_idx - 1
    if 0 <= raw_idx < len(feature_names):
        return str(feature_names[raw_idx])

    return f"x_{feature_idx}"


def _format_number(x, precision=4):
    try:
        return f"{float(x):.{precision}g}"
    except Exception:
        return str(x)


def _format_linear_model(leaf, feature_names=None, precision=4, max_terms=6):
    coefs = list(leaf.get("coefficients", []))
    cont = list(leaf.get("continuous_idx", []))

    if len(coefs) == 0:
        return "linear model"

    pieces = [_format_number(coefs[0], precision)]

    shown = 0
    for k, j in enumerate(cont):
        coef_idx = 1 + k
        if coef_idx >= len(coefs):
            break

        b = float(coefs[coef_idx])
        if abs(b) < 1e-12:
            continue

        name = _format_feature(j, feature_names)
        sign = "+" if b >= 0 else "-"
        pieces.append(f"{sign} {_format_number(abs(b), precision)}*{name}")
        shown += 1

        if shown >= max_terms:
            remaining = len(cont) - (k + 1)
            if remaining > 0:
                pieces.append("+ ...")
            break

    return "ŷ = " + " ".join(pieces)


def _format_leaf_label(leaf, feature_names=None, precision=4, max_terms=6):
    leaf_type = leaf.get("leaf_type", "UNKNOWN")
    n = leaf.get("n_instances", None)
    obj = leaf.get("obj", None)

    lines = [str(leaf_type)]

    if leaf_type == "CONSTANT":
        pred = leaf.get("prediction", None)
        lines.append(f"ŷ = {_format_number(pred, precision)}")
    elif leaf_type == "DEFER":
        lines.append("ŷ = reference")
    elif leaf_type == "LINEAR":
        lines.append(_format_linear_model(
            leaf,
            feature_names=feature_names,
            precision=precision,
            max_terms=max_terms,
        ))

    if n is not None:
        lines.append(f"n = {n}")
    if obj is not None:
        lines.append(f"obj = {_format_number(obj, precision)}")

    return "\n".join(lines)


def _insert_leaf(root, leaf):
    cur = root
    for cond in leaf.get("conditions", []):
        split = (cond["feature_idx"], cond["threshold"])
        op = cond["op"]

        if cur.get("split") is None:
            cur["split"] = split

        if cur["split"] != split:
            key = ("extra", split, op)
        else:
            key = op

        if key not in cur["children"]:
            cur["children"][key] = {
                "split": None,
                "children": {},
                "leaf": None,
            }

        cur = cur["children"][key]

    cur["leaf"] = leaf


def _plot_tree(
    self,
    feature_names=None,
    precision=4,
    max_terms=6,
    graph_attr=None,
    node_attr=None,
    edge_attr=None,
):
    try:
        from graphviz import Digraph
    except ImportError as exc:
        raise ImportError(
            "plot_tree requires the graphviz Python package. Install it with `pip install graphviz`."
        ) from exc

    leaves = self.export_leaf_paths()

    g = Digraph()
    g.attr("graph", rankdir="TB")
    g.attr("node", shape="box", style="rounded")
    g.attr("edge")

    if graph_attr:
        g.attr("graph", **graph_attr)
    if node_attr:
        g.attr("node", **node_attr)
    if edge_attr:
        g.attr("edge", **edge_attr)

    if len(leaves) == 0:
        g.node("empty", "No tree currently fit")
        return g

    root = {
        "split": None,
        "children": {},
        "leaf": None,
    }

    for leaf in leaves:
        _insert_leaf(root, leaf)

    counter = {"i": 0}

    def new_id():
        out = f"node_{counter['i']}"
        counter["i"] += 1
        return out

    def add_node(tree_node):
        node_id = new_id()

        if tree_node.get("leaf") is not None:
            label = _format_leaf_label(
                tree_node["leaf"],
                feature_names=feature_names,
                precision=precision,
                max_terms=max_terms,
            )
            g.node(node_id, label, shape="box")
            return node_id

        split = tree_node.get("split")
        if split is None:
            label = "root"
        else:
            feature_idx, threshold = split
            fname = _format_feature(feature_idx, feature_names)
            label = f"{fname} <= {_format_number(threshold, precision)}"

        g.node(node_id, label, shape="ellipse")

        for branch_key, child in tree_node.get("children", {}).items():
            child_id = add_node(child)

            if branch_key == "<=":
                edge_label = "True"
            elif branch_key == ">":
                edge_label = "False"
            elif isinstance(branch_key, tuple) and len(branch_key) == 3:
                edge_label = "True" if branch_key[2] == "<=" else "False"
            else:
                edge_label = str(branch_key)

            g.edge(node_id, child_id, label=edge_label)

        return node_id

    add_node(root)
    return g


Greedy.plot_tree = _plot_tree
CLARITree.plot_tree = _plot_tree


__all__ = ["Greedy", "CLARITree", "GreedyConst", "CLARITreeConst"]