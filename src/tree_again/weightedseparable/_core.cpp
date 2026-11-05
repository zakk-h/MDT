#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>

#include "weightedseparable.cpp"

namespace py = pybind11;

static std::vector<std::vector<uint8_t>> numpy_to_rowmajor_u8(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X
) {
    py::buffer_info info = X.request();
    if (info.ndim != 2) throw std::runtime_error("X must be 2D (n_samples, n_features)");

    const int n_samples  = static_cast<int>(info.shape[0]);
    const int n_features = static_cast<int>(info.shape[1]);
    const auto* x_ptr = static_cast<const uint8_t*>(info.ptr);

    std::vector<std::vector<uint8_t>> X_row_major(
        static_cast<size_t>(n_samples),
        std::vector<uint8_t>(static_cast<size_t>(n_features))
    );

    for (int i = 0; i < n_samples; ++i) {
        std::memcpy(
            X_row_major[(size_t)i].data(),
            x_ptr + (size_t)i * (size_t)n_features,
            (size_t)n_features * sizeof(uint8_t)
        );
    }
    return X_row_major;
}

static std::vector<int> numpy_to_vec_i32(
    py::array_t<int, py::array::c_style | py::array::forcecast> y
) {
    py::buffer_info info = y.request();
    if (info.ndim != 1) throw std::runtime_error("y must be 1D (n_samples,)");
    const int n = static_cast<int>(info.shape[0]);
    const auto* p = static_cast<const int*>(info.ptr);
    return std::vector<int>(p, p + n);
}

static std::vector<uint8_t> numpy_to_vec_u8(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> v
) {
    py::buffer_info info = v.request();
    if (info.ndim != 1) throw std::runtime_error("array must be 1D");
    const int n = static_cast<int>(info.shape[0]);
    const auto* p = static_cast<const uint8_t*>(info.ptr);
    return std::vector<uint8_t>(p, p + n);
}

static std::vector<double> numpy_to_vec_f64(
    py::array_t<double, py::array::c_style | py::array::forcecast> v
) {
    py::buffer_info info = v.request();
    if (info.ndim != 1) throw std::runtime_error("sample_weights must be 1D");
    const int n = static_cast<int>(info.shape[0]);
    const auto* p = static_cast<const double*>(info.ptr);
    return std::vector<double>(p, p + n);
}

static py::array_t<uint8_t> vec_u8_to_numpy(const std::vector<uint8_t>& v) {
    py::array_t<uint8_t> out((py::ssize_t)v.size());
    std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(uint8_t));
    return out;
}

static py::array_t<int> vec_i32_to_numpy(const std::vector<int>& v) {
    py::array_t<int> out((py::ssize_t)v.size());
    std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(int));
    return out;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Bindings for PRAXIS_WeightedSeparable (row-major X, optional sample weights)";

    py::class_<PRAXIS_WeightedSeparable>(m, "WeightedSeparable")
        .def(py::init<>())

        .def(
            "fit",
            [](PRAXIS_WeightedSeparable& self,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X,
               py::array_t<int,     py::array::c_style | py::array::forcecast> y,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> bb_pred,
               double lambda_leaf,
               double eta_defer,
               int depth_budget,
               int lookahead_k,
               py::object sample_weights_obj)
            {
                py::buffer_info xinfo = X.request();
                py::buffer_info yinfo = y.request();
                py::buffer_info binfo = bb_pred.request();

                if (xinfo.ndim != 2) throw std::runtime_error("X must be 2D (n_samples, n_features)");
                if (yinfo.ndim != 1) throw std::runtime_error("y must be 1D (n_samples,)");
                if (binfo.ndim != 1) throw std::runtime_error("bb_pred must be 1D (n_samples,)");

                const int n_samples = static_cast<int>(xinfo.shape[0]);
                if ((int)yinfo.shape[0] != n_samples) throw std::runtime_error("y length must match X rows");
                if ((int)binfo.shape[0] != n_samples) throw std::runtime_error("bb_pred length must match X rows");

                if (depth_budget < -128 || depth_budget > 127)
                    throw std::runtime_error("depth_budget out of int8 range [-128,127]");
                if (lookahead_k < 1 || lookahead_k > 127)
                    throw std::runtime_error("lookahead_k out of int8 range [1,127]");

                auto X_row_major = numpy_to_rowmajor_u8(X);
                auto y_vec       = numpy_to_vec_i32(y);
                auto bb_vec      = numpy_to_vec_u8(bb_pred);

                std::vector<double> weights_vec;
                if (!sample_weights_obj.is_none()) {
                    auto weights_arr =
                        py::cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(sample_weights_obj);
                    py::buffer_info winfo = weights_arr.request();
                    if (winfo.ndim != 1) throw std::runtime_error("sample_weights must be 1D");
                    if ((int)winfo.shape[0] != n_samples)
                        throw std::runtime_error("sample_weights length must match X rows");
                    weights_vec = numpy_to_vec_f64(weights_arr);
                }

                self.fit(
                    X_row_major,
                    y_vec,
                    bb_vec,
                    lambda_leaf,
                    eta_defer,
                    (int8_t)depth_budget,
                    (int8_t)lookahead_k,
                    weights_vec
                );

                return self.last_objective();
            },
            py::arg("X"),
            py::arg("y"),
            py::arg("bb_pred"),
            py::arg("lambda_leaf"),
            py::arg("eta_defer"),
            py::arg("depth_budget"),
            py::arg("lookahead_k") = 1,
            py::arg("sample_weights") = py::none()
        )

        .def(
            "predict",
            [](const PRAXIS_WeightedSeparable& self,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> bb_pred,
               int placeholder)
            {
                py::buffer_info xinfo = X.request();
                py::buffer_info binfo = bb_pred.request();
                if (xinfo.ndim != 2) throw std::runtime_error("X must be 2D (n_samples, n_features)");
                if (binfo.ndim != 1) throw std::runtime_error("bb_pred must be 1D (n_samples,)");

                const int n_samples = static_cast<int>(xinfo.shape[0]);
                if ((int)binfo.shape[0] != n_samples) throw std::runtime_error("bb_pred length must match X rows");

                auto X_row_major = numpy_to_rowmajor_u8(X);
                auto bb_vec      = numpy_to_vec_u8(bb_pred);

                auto preds = self.predict(X_row_major, bb_vec, (uint8_t)placeholder);
                return vec_u8_to_numpy(preds);
            },
            py::arg("X"),
            py::arg("bb_pred"),
            py::arg("placeholder") = 99
        )

        .def(
            "predict_placeholder",
            [](const PRAXIS_WeightedSeparable& self,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X,
               int placeholder)
            {
                py::buffer_info xinfo = X.request();
                if (xinfo.ndim != 2) throw std::runtime_error("X must be 2D (n_samples, n_features)");

                auto X_row_major = numpy_to_rowmajor_u8(X);

                std::vector<uint8_t> empty_bb;
                auto preds = self.predict(X_row_major, empty_bb, (uint8_t)placeholder);
                return vec_u8_to_numpy(preds);
            },
            py::arg("X"),
            py::arg("placeholder") = 99
        )

        .def("last_objective", &PRAXIS_WeightedSeparable::last_objective)

        .def(
            "leaf_counts_single_tree",
            [](const PRAXIS_WeightedSeparable& self) {
                auto c = self.leaf_counts_single_tree();
                py::dict d;
                d["predict0"] = c.predict0;
                d["predict1"] = c.predict1;
                d["defer"]    = c.defer;
                d["total"]    = c.total();
                return d;
            }
        )

        .def(
            "split_counts_single_tree",
            [](const PRAXIS_WeightedSeparable& self,
               py::array_t<uint8_t, py::array::c_style | py::array::forcecast> X)
            {
                py::buffer_info xinfo = X.request();
                if (xinfo.ndim != 2) throw std::runtime_error("X must be 2D (n_samples, n_features)");

                auto X_row_major = numpy_to_rowmajor_u8(X);
                auto counts = self.split_counts_single_tree(X_row_major);
                return vec_i32_to_numpy(counts);
            },
            py::arg("X")
        )

        .def(
            "leaf_paths_single_tree",
            [](const PRAXIS_WeightedSeparable& self) {
                return self.leaf_paths_single_tree();
            }
        )

        .def(
            "leaf_actions_single_tree",
            [](const PRAXIS_WeightedSeparable& self) {
                return self.leaf_actions_single_tree();
            }
        )

        .def(
            "leaf_paths_and_actions_single_tree",
            [](const PRAXIS_WeightedSeparable& self) {
                return self.leaf_paths_and_actions_single_tree();
            }
        );
}