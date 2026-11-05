#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "clari_tree.hpp"
#include "clari_tree_const.hpp"

namespace py = pybind11;

namespace {

template <typename Tree, typename... Options>
void bind_tree_methods(py::class_<Tree, Options...>& cls) {
    cls.def("fit",
            [](Tree& self,
               const Eigen::Ref<const Eigen::MatrixXd>& X,
               const Eigen::Ref<const Eigen::VectorXd>& y,
               const std::vector<int>& categorical_idx) {
                py::gil_scoped_release release;
                return self.fit(X, y, categorical_idx);
            },
            py::arg("X"),
            py::arg("y"),
            py::arg("categorical_idx") = std::vector<int>(),
            "Fit the tree with (X, y). Returns objective loss.")
       .def("predict",
            [](Tree& self, const Eigen::Ref<const Eigen::MatrixXd>& X) {
                py::gil_scoped_release release;
                return self.predict(X);
            },
            py::arg("X"),
            "Predict values for X.")
       .def("print_tree", &Tree::print_tree)
       .def("get_traversed_thresholds",
            [](const Tree& self) {
                py::dict out;
                const auto all_thresholds = self.get_traversed_thresholds();
                for (std::size_t feature = 1; feature < all_thresholds.size(); ++feature) {
                    py::list thresholds;
                    for (double t : all_thresholds[feature]) {
                        thresholds.append(t);
                    }
                    out[py::int_(feature)] = thresholds;
                }
                return out;
            },
            "Return traversed candidate thresholds by feature index.")
       .def("print_traversed_thresholds", &Tree::print_traversed_thresholds)
       .def("get_threshold_pool",
            [](const Tree& self) {
                py::dict out;
                const auto all_thresholds = self.get_threshold_pool();
                for (std::size_t feature = 1; feature < all_thresholds.size(); ++feature) {
                    py::list thresholds;
                    for (double t : all_thresholds[feature]) {
                        thresholds.append(t);
                    }
                    out[py::int_(feature)] = thresholds;
                }
                return out;
            },
            "Return fit-time global threshold pool by feature index.")
       .def("print_threshold_pool", &Tree::print_threshold_pool)
       .def("n_leaves", &Tree::n_leaves);
}

template <typename Tree, typename... Options>
void bind_leaf_path_methods(py::class_<Tree, Options...>& cls) {
    cls.def("print_leaf_paths", &Tree::print_leaf_paths)
       .def("export_leaf_paths",
            [](const Tree& self) {
                py::list out;

                const auto paths = self.export_leaf_paths();

                for (const auto& leaf : paths) {
                    py::dict d;

                    py::list conditions;
                    for (const auto& c : leaf.conditions) {
                        py::dict cd;
                        cd["feature_idx"] = c.feature_idx;
                        cd["threshold"] = c.threshold;
                        cd["op"] = c.is_leq ? "<=" : ">";
                        conditions.append(cd);
                    }

                    d["conditions"] = conditions;
                    d["leaf_type"] =
                        leaf.leaf_type == LeafType::CONSTANT ? "CONSTANT" :
                        leaf.leaf_type == LeafType::DEFER    ? "DEFER" :
                                                                "LINEAR";

                    d["obj"] = leaf.obj;
                    d["n_instances"] = leaf.n_instances;

                    if (leaf.leaf_type == LeafType::CONSTANT) {
                        d["prediction"] = leaf.constant_prediction;
                    } else if (leaf.leaf_type == LeafType::DEFER) {
                        d["prediction"] = "reference";
                    } else {
                        py::list coefs;
                        for (int i = 0; i < leaf.coefficients.size(); ++i) {
                            coefs.append(leaf.coefficients(i));
                        }

                        py::list cont;
                        for (int j : leaf.continuous_idx) {
                            cont.append(j);
                        }

                        d["coefficients"] = coefs;
                        d["continuous_idx"] = cont;
                    }

                    out.append(d);
                }

                return out;
            },
            "Return root-to-leaf paths with split conditions and leaf prediction/defer/linear model.");
}

} // namespace

template <typename Tree>
void bind_three_leaf_methods(py::class_<Tree, Greedy>& cls) {
    cls.def("fit",
            [](Tree& self,
               const Eigen::Ref<const Eigen::MatrixXd>& X,
               const Eigen::Ref<const Eigen::VectorXd>& y,
               const Eigen::Ref<const Eigen::VectorXd>& reference_pred,
               const std::vector<int>& categorical_idx) {
                py::gil_scoped_release release;
                return self.fit(X, y, reference_pred, categorical_idx);
            },
            py::arg("X"),
            py::arg("y"),
            py::arg("reference_pred"),
            py::arg("categorical_idx") = std::vector<int>(),
            "Fit the three-leaf tree with (X, y, reference_pred). Returns objective loss.")
       .def("fit_with_reference",
            [](Tree& self,
               const Eigen::Ref<const Eigen::MatrixXd>& X,
               const Eigen::Ref<const Eigen::VectorXd>& y,
               const Eigen::Ref<const Eigen::VectorXd>& reference_pred,
               const std::vector<int>& categorical_idx) {
                py::gil_scoped_release release;
                return self.fit(X, y, reference_pred, categorical_idx);
            },
            py::arg("X"),
            py::arg("y"),
            py::arg("reference_pred"),
            py::arg("categorical_idx") = std::vector<int>(),
            "Fit the three-leaf tree with (X, y, reference_pred). Returns objective loss.")
       .def("fit_weighted",
            [](Tree& self,
               const Eigen::Ref<const Eigen::MatrixXd>& X,
               const Eigen::Ref<const Eigen::VectorXd>& y,
               const Eigen::Ref<const Eigen::VectorXd>& sample_weight,
               const std::vector<int>& categorical_idx) {
                py::gil_scoped_release release;
                return self.fit_weighted(X, y, sample_weight, categorical_idx);
            },
            py::arg("X"),
            py::arg("y"),
            py::arg("sample_weight"),
            py::arg("categorical_idx") = std::vector<int>(),
            "Fit the tree with sample weights. Returns objective loss.")
       .def("fit_with_reference_and_weights",
            [](Tree& self,
               const Eigen::Ref<const Eigen::MatrixXd>& X,
               const Eigen::Ref<const Eigen::VectorXd>& y,
               const Eigen::Ref<const Eigen::VectorXd>& reference_pred,
               const Eigen::Ref<const Eigen::VectorXd>& sample_weight,
               const std::vector<int>& categorical_idx) {
                py::gil_scoped_release release;
                return self.fit_with_reference_and_weights(X, y, reference_pred, sample_weight, categorical_idx);
            },
            py::arg("X"),
            py::arg("y"),
            py::arg("reference_pred"),
            py::arg("sample_weight"),
            py::arg("categorical_idx") = std::vector<int>(),
            "Fit the three-leaf tree with reference predictions and sample weights. Returns objective loss.")
       .def("predict",
            [](Tree& self,
               const Eigen::Ref<const Eigen::MatrixXd>& X,
               const Eigen::Ref<const Eigen::VectorXd>& reference_pred) {
                py::gil_scoped_release release;
                return self.predict(X, reference_pred);
            },
            py::arg("X"),
            py::arg("reference_pred"),
            "Predict values for X, using reference_pred for defer leaves.")
       .def("predict_with_reference",
            [](Tree& self,
               const Eigen::Ref<const Eigen::MatrixXd>& X,
               const Eigen::Ref<const Eigen::VectorXd>& reference_pred) {
                py::gil_scoped_release release;
                return self.predict(X, reference_pred);
            },
            py::arg("X"),
            py::arg("reference_pred"),
            "Predict values for X, using reference_pred for defer leaves.");
}

void bind_three_leaf_methods(py::class_<Greedy>& cls) {
    cls.def("fit",
            [](Greedy& self,
               const Eigen::Ref<const Eigen::MatrixXd>& X,
               const Eigen::Ref<const Eigen::VectorXd>& y,
               const Eigen::Ref<const Eigen::VectorXd>& reference_pred,
               const std::vector<int>& categorical_idx) {
                py::gil_scoped_release release;
                return self.fit(X, y, reference_pred, categorical_idx);
            },
            py::arg("X"),
            py::arg("y"),
            py::arg("reference_pred"),
            py::arg("categorical_idx") = std::vector<int>(),
            "Fit the three-leaf tree with (X, y, reference_pred). Returns objective loss.")
       .def("fit_with_reference",
            [](Greedy& self,
               const Eigen::Ref<const Eigen::MatrixXd>& X,
               const Eigen::Ref<const Eigen::VectorXd>& y,
               const Eigen::Ref<const Eigen::VectorXd>& reference_pred,
               const std::vector<int>& categorical_idx) {
                py::gil_scoped_release release;
                return self.fit(X, y, reference_pred, categorical_idx);
            },
            py::arg("X"),
            py::arg("y"),
            py::arg("reference_pred"),
            py::arg("categorical_idx") = std::vector<int>(),
            "Fit the three-leaf tree with (X, y, reference_pred). Returns objective loss.")
       .def("fit_weighted",
            [](Greedy& self,
               const Eigen::Ref<const Eigen::MatrixXd>& X,
               const Eigen::Ref<const Eigen::VectorXd>& y,
               const Eigen::Ref<const Eigen::VectorXd>& sample_weight,
               const std::vector<int>& categorical_idx) {
                py::gil_scoped_release release;
                return self.fit_weighted(X, y, sample_weight, categorical_idx);
            },
            py::arg("X"),
            py::arg("y"),
            py::arg("sample_weight"),
            py::arg("categorical_idx") = std::vector<int>(),
            "Fit the tree with sample weights. Returns objective loss.")
       .def("fit_with_reference_and_weights",
            [](Greedy& self,
               const Eigen::Ref<const Eigen::MatrixXd>& X,
               const Eigen::Ref<const Eigen::VectorXd>& y,
               const Eigen::Ref<const Eigen::VectorXd>& reference_pred,
               const Eigen::Ref<const Eigen::VectorXd>& sample_weight,
               const std::vector<int>& categorical_idx) {
                py::gil_scoped_release release;
                return self.fit_with_reference_and_weights(X, y, reference_pred, sample_weight, categorical_idx);
            },
            py::arg("X"),
            py::arg("y"),
            py::arg("reference_pred"),
            py::arg("sample_weight"),
            py::arg("categorical_idx") = std::vector<int>(),
            "Fit the three-leaf tree with reference predictions and sample weights. Returns objective loss.")
       .def("predict",
            [](Greedy& self,
               const Eigen::Ref<const Eigen::MatrixXd>& X,
               const Eigen::Ref<const Eigen::VectorXd>& reference_pred) {
                py::gil_scoped_release release;
                return self.predict(X, reference_pred);
            },
            py::arg("X"),
            py::arg("reference_pred"),
            "Predict values for X, using reference_pred for defer leaves.")
       .def("predict_with_reference",
            [](Greedy& self,
               const Eigen::Ref<const Eigen::MatrixXd>& X,
               const Eigen::Ref<const Eigen::VectorXd>& reference_pred) {
                py::gil_scoped_release release;
                return self.predict(X, reference_pred);
            },
            py::arg("X"),
            py::arg("reference_pred"),
            "Predict values for X, using reference_pred for defer leaves.");
}


PYBIND11_MODULE(_core, m) {
    m.doc() = "clari_tree C++ core bindings";

    py::class_<Greedy> greedy(m, "Greedy");
    greedy.def(py::init<double, Depth, double, int, const std::string&, bool, int>(),
            py::arg("kappa"),
            py::arg("depth"),
            py::arg("lambda_") = 0.0,
            py::arg("n_thresholds") = 20,
            py::arg("thresholds_strategy") = "quantile",
            py::arg("verbose") = true,
            py::arg("min_leaf_node_size") = 0)
        .def(py::init<double, Depth, double, int, bool, int>(),
            py::arg("kappa"),
            py::arg("depth"),
            py::arg("lambda_"),
            py::arg("n_thresholds"),
            py::arg("verbose"),
            py::arg("min_leaf_node_size") = 0)

        // three-leaf constructors.
        .def(py::init<double, Depth, double, double, double, int, const std::string&, bool, int>(),
            py::arg("kappa"),
            py::arg("depth"),
            py::arg("lambda_"),
            py::arg("rho"),
            py::arg("eta"),
            py::arg("n_thresholds") = 20,
            py::arg("thresholds_strategy") = "quantile",
            py::arg("verbose") = true,
            py::arg("min_leaf_node_size") = 0)
        .def(py::init<double, Depth, double, double, double, int, bool, int>(),
            py::arg("kappa"),
            py::arg("depth"),
            py::arg("lambda_"),
            py::arg("rho"),
            py::arg("eta"),
            py::arg("n_thresholds"),
            py::arg("verbose"),
            py::arg("min_leaf_node_size") = 0);

    bind_tree_methods(greedy);
    bind_leaf_path_methods(greedy);
    bind_three_leaf_methods(greedy);

    py::class_<CLARITree, Greedy> clari_tree(m, "CLARITree");
    clari_tree.def(py::init<double, Depth, double, int, const std::string&, bool, int>(),
                py::arg("kappa"),
                py::arg("depth"),
                py::arg("lambda_") = 0.0,
                py::arg("n_thresholds") = 20,
                py::arg("thresholds_strategy") = "quantile",
                py::arg("verbose") = true,
                py::arg("min_leaf_node_size") = 0)
            .def(py::init<double, Depth, double, int, bool, int>(),
                py::arg("kappa"),
                py::arg("depth"),
                py::arg("lambda_"),
                py::arg("n_thresholds"),
                py::arg("verbose"),
                py::arg("min_leaf_node_size") = 0)

            // three-leaf constructors.
            .def(py::init<double, Depth, double, double, double, int, const std::string&, bool, int>(),
                py::arg("kappa"),
                py::arg("depth"),
                py::arg("lambda_"),
                py::arg("rho"),
                py::arg("eta"),
                py::arg("n_thresholds") = 20,
                py::arg("thresholds_strategy") = "quantile",
                py::arg("verbose") = true,
                py::arg("min_leaf_node_size") = 0)
            .def(py::init<double, Depth, double, double, double, int, bool, int>(),
                py::arg("kappa"),
                py::arg("depth"),
                py::arg("lambda_"),
                py::arg("rho"),
                py::arg("eta"),
                py::arg("n_thresholds"),
                py::arg("verbose"),
                py::arg("min_leaf_node_size") = 0);

    bind_tree_methods(clari_tree);
    bind_leaf_path_methods(clari_tree);
    bind_three_leaf_methods(clari_tree);

    py::class_<GreedyConst> greedy_const(m, "GreedyConst");
    greedy_const.def(py::init<int, double, int, const std::string&, bool, int>(),
                     py::arg("depth"),
                     py::arg("lambda_") = 0.0,
                     py::arg("n_thresholds") = 20,
                     py::arg("thresholds_strategy") = "quantile",
                     py::arg("verbose") = true,
                     py::arg("min_leaf_node_size") = 1)
                .def(py::init<int, double, int, bool, int>(),
                     py::arg("depth"),
                     py::arg("lambda_"),
                     py::arg("n_thresholds"),
                     py::arg("verbose"),
                     py::arg("min_leaf_node_size") = 1);
    bind_tree_methods(greedy_const);

    py::class_<CLARITreeConst, GreedyConst> clari_tree_const(m, "CLARITreeConst");
    clari_tree_const.def(py::init<int, double, int, const std::string&, bool, int>(),
                        py::arg("depth"),
                        py::arg("lambda_") = 0.0,
                        py::arg("n_thresholds") = 20,
                        py::arg("thresholds_strategy") = "quantile",
                        py::arg("verbose") = true,
                        py::arg("min_leaf_node_size") = 1)
                    .def(py::init<int, double, int, bool, int>(),
                        py::arg("depth"),
                        py::arg("lambda_"),
                        py::arg("n_thresholds"),
                        py::arg("verbose"),
                        py::arg("min_leaf_node_size") = 1);

    bind_tree_methods(clari_tree_const);
}