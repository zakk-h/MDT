#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdint>
#include <memory>
#include <string>
#include <iostream>
#include <stdexcept>

using namespace std;

#if defined(_MSC_VER)
  #include <intrin.h>
  static inline int popcnt64(uint64_t x) { return static_cast<int>(__popcnt64(x)); }
  static inline int ctz64(uint64_t x) {
      unsigned long idx;
      _BitScanForward64(&idx, x);
      return static_cast<int>(idx);
  }
#else
  static inline int popcnt64(uint64_t x) { return __builtin_popcountll(x); }
  static inline int ctz64(uint64_t x) { return __builtin_ctzll(x); }
#endif

struct Packed {
    vector<uint64_t> w;
    Packed() = default;
    explicit Packed(size_t nwords) : w(nwords, 0ULL) {}

    inline bool any() const { for (uint64_t t : w) if (t) return true; return false; }
    inline int  count() const { int s = 0; for (uint64_t t : w) s += popcnt64(t); return s; }
};

class PRAXIS_WeightedSeparable {
public:
    enum class LeafKind : uint8_t { PREDICT = 0, DEFER = 1 };

    double last_objective() const { return last_objective_; }

    void fit(
        const vector<vector<uint8_t>>& X_row_major,
        const vector<int>& y,
        const vector<uint8_t>& bb_pred,
        double lambda_leaf,
        double eta_defer,
        int8_t depth_budget,
        int8_t lookahead_k = 1,
        const vector<double>& sample_weights = {}
    ) {
        if (X_row_major.empty()) throw runtime_error("fit: X has 0 rows.");
        if (X_row_major[0].empty()) throw runtime_error("fit: X has 0 features.");
        if (y.empty()) throw runtime_error("fit: y is empty.");
        if ((int)X_row_major.size() != (int)y.size()) throw runtime_error("fit: X and y size mismatch.");
        if ((int)bb_pred.size() != (int)y.size()) throw runtime_error("fit: bb_pred and y size mismatch.");
        if (lookahead_k < 1) throw runtime_error("fit: lookahead_k must be >= 1.");
        if ((int)X_row_major[0].size() <= 0) throw runtime_error("fit: invalid feature count.");
        if (!sample_weights.empty() && (int)sample_weights.size() != (int)y.size())
            throw runtime_error("fit: sample_weights and y size mismatch.");

        n_samples  = (int)y.size();
        n_features = (int)X_row_major[0].size();

        for (int i = 0; i < n_samples; ++i) {
            if ((int)X_row_major[(size_t)i].size() != n_features)
                throw runtime_error("fit: inconsistent row lengths in X.");
        }

        sample_weights_.assign((size_t)n_samples, 1.0);
        if (!sample_weights.empty()) {
            sample_weights_ = sample_weights;
        }

        total_weight_ = 0.0;
        for (double w : sample_weights_) {
            if (!std::isfinite(w)) throw runtime_error("fit: sample_weights contains non-finite value.");
            total_weight_ += w;
        }

        n_words   = (n_samples + 63) / 64;
        tail_mask = (n_samples % 64) ? ((1ULL << (n_samples % 64)) - 1ULL) : ~0ULL;

        // match the weighted defer-tree objective's lambda * N * leaves scaling.
        lamN = lambda_leaf * (double)n_samples;
        eta = eta_defer;
        depth_trained = depth_budget;
        k_trained     = lookahead_k;

        X_bits.assign((size_t)n_features, Packed((size_t)n_words));
        for (int f = 0; f < n_features; ++f) {
            auto& col = X_bits[(size_t)f].w;
            for (int i = 0; i < n_samples; ++i) {
                if (X_row_major[(size_t)i][(size_t)f] & 1)
                    col[(size_t)(i >> 6)] |= (1ULL << (i & 63));
            }
            col[(size_t)(n_words - 1)] &= tail_mask;
        }

        Y1 = Packed((size_t)n_words);
        for (int i = 0; i < n_samples; ++i) {
            if ((y[(size_t)i] & 1) == 1)
                Y1.w[(size_t)(i >> 6)] |= (1ULL << (i & 63));
        }
        Y1.w[(size_t)(n_words - 1)] &= tail_mask;

        BBwrong = Packed((size_t)n_words);
        for (int i = 0; i < n_samples; ++i) {
            const uint8_t yi = (uint8_t)(y[(size_t)i] & 1);
            const uint8_t bi = (uint8_t)(bb_pred[(size_t)i] & 1);
            if (yi != bi) BBwrong.w[(size_t)(i >> 6)] |= (1ULL << (i & 63));
        }
        BBwrong.w[(size_t)(n_words - 1)] &= tail_mask;

        Packed root((size_t)n_words);
        for (int w = 0; w < n_words - 1; ++w) root.w[(size_t)w] = ~0ULL;
        root.w[(size_t)(n_words - 1)] = tail_mask;

        double obj = 0.0;
        root_node = build_lickety_tree(root, depth_budget, lookahead_k, obj);
        last_objective_ = obj;
    }

    vector<uint8_t> predict(
        const vector<vector<uint8_t>>& X_row_major,
        const vector<uint8_t>& bb_pred_row = {},
        uint8_t placeholder = (uint8_t)99
    ) const {
        if (!root_node) throw runtime_error("predict: Model not fit.");
        const size_t N = X_row_major.size();
        if (N == 0) return {};
        if ((int)X_row_major[0].size() != n_features) throw runtime_error("predict: feature count mismatch.");

        const bool use_placeholder = bb_pred_row.empty();
        if (!use_placeholder && bb_pred_row.size() != N) throw runtime_error("predict: bb_pred_row size mismatch.");

        const vector<uint8_t> dummy_bb(use_placeholder ? N : 0, 0);
        const vector<uint8_t>& bb = use_placeholder ? dummy_bb : bb_pred_row;

        vector<uint8_t> out(N, 0);

        vector<int> idx; idx.reserve(N);
        for (size_t i = 0; i < N; ++i) idx.push_back((int)i);

        predict_rec(root_node.get(), X_row_major, bb, out, idx, use_placeholder, placeholder);
        return out;
    }

    struct LeafCounts {
        int predict0 = 0;
        int predict1 = 0;
        int defer    = 0;
        int total() const { return predict0 + predict1 + defer; }
    };

    LeafCounts leaf_counts_single_tree() const {
        if (!root_node) return {};
        LeafCounts c;
        count_leaves_rec_(root_node.get(), c);
        return c;
    }

    vector<int> split_counts_single_tree(
        const vector<vector<uint8_t>>& X_row_major
    ) const {
        if (!root_node) throw runtime_error("split_counts_single_tree: Model not fit.");
        const size_t N = X_row_major.size();
        if (N == 0) return {};
        if ((int)X_row_major[0].size() != n_features)
            throw runtime_error("split_counts_single_tree: feature count mismatch.");

        vector<int> out(N, 0);
        for (size_t i = 0; i < N; ++i) {
            out[i] = split_count_one_(root_node.get(), X_row_major[i]);
        }
        return out;
    }

    std::vector<std::vector<int>> leaf_paths_single_tree() const {
        std::vector<std::vector<int>> paths;
        if (!root_node) return paths;
        std::vector<int> cur;
        collect_leaf_paths_rec_(root_node.get(), cur, paths);
        return paths;
    }

    std::vector<int> leaf_actions_single_tree() const {
        std::vector<int> actions;
        if (!root_node) return actions;
        std::vector<int> cur;
        std::vector<std::vector<int>> dummy_paths;
        collect_leaf_paths_and_actions_rec_(root_node.get(), cur, dummy_paths, actions);
        return actions;
    }

    std::pair<std::vector<std::vector<int>>, std::vector<int>> leaf_paths_and_actions_single_tree() const {
        std::vector<std::vector<int>> paths;
        std::vector<int> actions;
        if (!root_node) return {paths, actions};
        std::vector<int> cur;
        collect_leaf_paths_and_actions_rec_(root_node.get(), cur, paths, actions);
        return {paths, actions};
    }

private:
    struct Node {
        int feature = -1;
        LeafKind kind = LeafKind::PREDICT;
        uint8_t pred_class = 0;
        shared_ptr<Node> left;
        shared_ptr<Node> right;
    };

    int n_samples = 0;
    int n_features = 0;
    int n_words = 0;
    uint64_t tail_mask = ~0ULL;

    double lamN = 0.0;
    double eta = 0.0;
    int8_t depth_trained = -1;
    int8_t k_trained = 1;

    vector<double> sample_weights_;
    double total_weight_ = 0.0;

    vector<Packed> X_bits;
    Packed Y1;
    Packed BBwrong;

    shared_ptr<Node> root_node;
    double last_objective_ = 0.0;

    inline void and_bits(const Packed& a, const Packed& b, Packed& out) const {
        for (int i = 0; i < n_words; ++i) out.w[(size_t)i] = a.w[(size_t)i] & b.w[(size_t)i];
        out.w[(size_t)(n_words - 1)] &= tail_mask;
    }

    inline void andnot_bits(const Packed& a, const Packed& b, Packed& out) const {
        for (int i = 0; i < n_words; ++i) out.w[(size_t)i] = a.w[(size_t)i] & ~b.w[(size_t)i];
        out.w[(size_t)(n_words - 1)] &= tail_mask;
    }

    inline double weight_sum_mask(const Packed& mask) const {
        double s = 0.0;
        for (int wi = 0; wi < n_words; ++wi) {
            uint64_t bits = mask.w[(size_t)wi];
            while (bits) {
                const int b = ctz64(bits);
                const int idx = (wi << 6) + b;
                s += sample_weights_[(size_t)idx];
                bits &= (bits - 1ULL);
            }
        }
        return s;
    }

    inline double weight_sum_and(const Packed& a, const Packed& b) const {
        double s = 0.0;
        for (int wi = 0; wi < n_words; ++wi) {
            uint64_t bits = a.w[(size_t)wi] & b.w[(size_t)wi];
            while (bits) {
                const int bit = ctz64(bits);
                const int idx = (wi << 6) + bit;
                s += sample_weights_[(size_t)idx];
                bits &= (bits - 1ULL);
            }
        }
        return s;
    }

    static inline double entropy_bin(double pos_mass, double total_mass) {
        if (total_mass <= 0.0) return 0.0;
        const double eps = 1e-12;
        double p = pos_mass / total_mass;
        p = max(eps, min(1.0 - eps, p));
        return -(p * log2(p) + (1.0 - p) * log2(1.0 - p));
    }

    struct BestLeaf {
        double cost = std::numeric_limits<double>::infinity();
        LeafKind kind = LeafKind::PREDICT;
        uint8_t pred_class = 0;
    };

    BestLeaf best_leaf_option(const Packed& mask) const {
        const double mass = weight_sum_mask(mask);
        if (mass <= 0.0) return BestLeaf{0.0, LeafKind::PREDICT, 0};

        const double pos_mass = weight_sum_and(mask, Y1);
        const double neg_mass = mass - pos_mass;

        const double cost0 = lamN + pos_mass;
        const double cost1 = lamN + neg_mass;

        const double bb_mis_mass = weight_sum_and(mask, BBwrong);
        const double defer_pen = eta * mass;
        const double costD = lamN + defer_pen + bb_mis_mass;

        BestLeaf best{cost0, LeafKind::PREDICT, 0};
        if (cost1 < best.cost) best = BestLeaf{cost1, LeafKind::PREDICT, 1};
        if (costD < best.cost) best = BestLeaf{costD, LeafKind::DEFER, 0};
        return best;
    }

    double greedy_cost(const Packed& mask, int8_t depth) const {
        BestLeaf leaf = best_leaf_option(mask);
        if (depth <= 0) return leaf.cost;

        const int D_count = mask.count();
        if (D_count <= 1) return leaf.cost;

        const int best_f = find_best_split_entropy(mask);
        if (best_f < 0) return leaf.cost;

        Packed L((size_t)n_words), R((size_t)n_words);
        and_bits(mask, X_bits[(size_t)best_f], L);
        andnot_bits(mask, X_bits[(size_t)best_f], R);
        if (!L.any() || !R.any()) return leaf.cost;

        const double split_cost =
            greedy_cost(L, (int8_t)(depth - 1))
            + greedy_cost(R, (int8_t)(depth - 1));

        return std::min(leaf.cost, split_cost);
    }

    double lickety_cost(const Packed& mask, int8_t depth, int8_t k) const {
        BestLeaf leaf = best_leaf_option(mask);
        if (depth <= 0) return leaf.cost;

        const int D_count = mask.count();
        if (D_count <= 1) return leaf.cost;

        int best_feat = -1;
        double best_proxy_children = std::numeric_limits<double>::infinity();

        Packed L((size_t)n_words), R((size_t)n_words);

        for (int f = 0; f < n_features; ++f) {
            and_bits(mask, X_bits[(size_t)f], L);
            andnot_bits(mask, X_bits[(size_t)f], R);
            if (!L.any() || !R.any()) continue;

            double proxy_children = 0.0;
            if (k <= 1) {
                proxy_children = greedy_cost(L, (int8_t)(depth - 1)) + greedy_cost(R, (int8_t)(depth - 1));
            } else {
                const int8_t km1 = (int8_t)(k - 1);
                proxy_children = lickety_cost(L, (int8_t)(depth - 1), km1) + lickety_cost(R, (int8_t)(depth - 1), km1);
            }

            if (proxy_children < best_proxy_children) {
                best_proxy_children = proxy_children;
                best_feat = f;
            }
        }

        if (best_feat < 0) return leaf.cost;

        and_bits(mask, X_bits[(size_t)best_feat], L);
        andnot_bits(mask, X_bits[(size_t)best_feat], R);
        if (!L.any() || !R.any()) return leaf.cost;

        const double split_cost =
            lickety_cost(L, (int8_t)(depth - 1), k)
            + lickety_cost(R, (int8_t)(depth - 1), k);

        return std::min(leaf.cost, split_cost);
    }

    shared_ptr<Node> build_lickety_tree(const Packed& mask, int8_t depth, int8_t k, double& out_cost) const {
        BestLeaf leaf = best_leaf_option(mask);
        if (depth <= 0) { out_cost = leaf.cost; return make_leaf_(leaf); }

        const int D_count = mask.count();
        if (D_count <= 1) { out_cost = leaf.cost; return make_leaf_(leaf); }

        int best_feat = -1;
        double best_proxy_children = std::numeric_limits<double>::infinity();

        Packed bestL((size_t)n_words), bestR((size_t)n_words);
        Packed L((size_t)n_words), R((size_t)n_words);

        for (int f = 0; f < n_features; ++f) {
            and_bits(mask, X_bits[(size_t)f], L);
            andnot_bits(mask, X_bits[(size_t)f], R);
            if (!L.any() || !R.any()) continue;

            double proxy_children = 0.0;
            if (k <= 1) {
                proxy_children = greedy_cost(L, (int8_t)(depth - 1)) + greedy_cost(R, (int8_t)(depth - 1));
            } else {
                const int8_t km1 = (int8_t)(k - 1);
                proxy_children = lickety_cost(L, (int8_t)(depth - 1), km1) + lickety_cost(R, (int8_t)(depth - 1), km1);
            }

            if (proxy_children < best_proxy_children) {
                best_proxy_children = proxy_children;
                best_feat = f;
                bestL.w = L.w;
                bestR.w = R.w;
            }
        }

        if (best_feat < 0) { out_cost = leaf.cost; return make_leaf_(leaf); }

        double left_cost = 0.0, right_cost = 0.0;
        auto left_node  = build_lickety_tree(bestL, (int8_t)(depth - 1), k, left_cost);
        auto right_node = build_lickety_tree(bestR, (int8_t)(depth - 1), k, right_cost);

        const double split_cost = left_cost + right_cost;

        if (leaf.cost <= split_cost) { out_cost = leaf.cost; return make_leaf_(leaf); }

        out_cost = split_cost;
        auto n = make_shared<Node>();
        n->feature = best_feat;
        n->left = left_node;
        n->right = right_node;
        return n;
    }

    int find_best_split_entropy(const Packed& mask) const {
        const int D_count = mask.count();
        if (D_count <= 1) return -1;

        const double D_mass = weight_sum_mask(mask);
        if (D_mass <= 0.0) return -1;

        const double pos_mass = weight_sum_and(mask, Y1);
        const double baseH = entropy_bin(pos_mass, D_mass);

        int best_f = -1;
        double best_gain = -1e300;

        Packed L((size_t)n_words);
        for (int f = 0; f < n_features; ++f) {
            for (int i = 0; i < n_words; ++i) L.w[(size_t)i] = mask.w[(size_t)i] & X_bits[(size_t)f].w[(size_t)i];
            L.w[(size_t)(n_words - 1)] &= tail_mask;

            const int Dl_count = L.count();
            const int Dr_count = D_count - Dl_count;
            if (Dl_count == 0 || Dr_count == 0) continue;

            const double Dl_mass = weight_sum_mask(L);
            const double Dr_mass = D_mass - Dl_mass;
            if (Dl_mass <= 0.0 || Dr_mass <= 0.0) continue;

            const double posl_mass = weight_sum_and(L, Y1);
            const double posr_mass = pos_mass - posl_mass;

            const double wl = Dl_mass / D_mass;
            const double wr = Dr_mass / D_mass;

            const double Hl = entropy_bin(posl_mass, Dl_mass);
            const double Hr = entropy_bin(posr_mass, Dr_mass);

            const double gain = baseH - (wl * Hl + wr * Hr);
            if (gain > best_gain) { best_gain = gain; best_f = f; }
        }
        return best_f;
    }

    static shared_ptr<Node> make_leaf_(const BestLeaf& leaf) {
        auto n = make_shared<Node>();
        n->feature = -1;
        n->kind = leaf.kind;
        n->pred_class = leaf.pred_class;
        return n;
    }

    static void predict_rec(
        const Node* node,
        const vector<vector<uint8_t>>& X_row_major,
        const vector<uint8_t>& bb_pred_row,
        vector<uint8_t>& out,
        const vector<int>& idx,
        bool use_placeholder,
        uint8_t placeholder
    ) {
        if (!node) return;

        if (node->feature < 0) {
            if (node->kind == LeafKind::DEFER) {
                if (use_placeholder) {
                    for (int r : idx) out[(size_t)r] = placeholder;
                } else {
                    for (int r : idx) out[(size_t)r] = (uint8_t)(bb_pred_row[(size_t)r] & 1);
                }
            } else {
                const uint8_t p = (uint8_t)(node->pred_class & 1);
                for (int r : idx) out[(size_t)r] = p;
            }
            return;
        }

        const int f = node->feature;
        vector<int> left_idx;  left_idx.reserve(idx.size());
        vector<int> right_idx; right_idx.reserve(idx.size());

        for (int r : idx) {
            if (X_row_major[(size_t)r][(size_t)f] & 1) left_idx.push_back(r);
            else right_idx.push_back(r);
        }

        if (!left_idx.empty())  predict_rec(node->left.get(),  X_row_major, bb_pred_row, out, left_idx, use_placeholder, placeholder);
        if (!right_idx.empty()) predict_rec(node->right.get(), X_row_major, bb_pred_row, out, right_idx, use_placeholder, placeholder);
    }

    void count_leaves_rec_(const Node* node, LeafCounts& c) const {
        if (!node) return;

        if (node->feature < 0) {
            if (node->kind == LeafKind::DEFER) {
                c.defer += 1;
            } else {
                const uint8_t p = (uint8_t)(node->pred_class & 1);
                if (p == 0) c.predict0 += 1;
                else        c.predict1 += 1;
            }
            return;
        }

        if (node->left)  count_leaves_rec_(node->left.get(), c);
        if (node->right) count_leaves_rec_(node->right.get(), c);
    }

    static int split_count_one_(const Node* node, const vector<uint8_t>& xrow) {
        int s = 0;
        const Node* cur = node;
        while (cur && cur->feature >= 0) {
            const int f = cur->feature;
            ++s;
            cur = (xrow[(size_t)f] & 1) ? cur->left.get() : cur->right.get();
        }
        return s;
    }

    static void collect_leaf_paths_rec_(
        const Node* node,
        std::vector<int>& cur,
        std::vector<std::vector<int>>& out
    ) {
        if (!node) return;

        if (node->feature < 0) {
            out.push_back(cur);
            return;
        }

        const int f1 = node->feature + 1;

        cur.push_back(+f1);
        collect_leaf_paths_rec_(node->left.get(), cur, out);
        cur.pop_back();

        cur.push_back(-f1);
        collect_leaf_paths_rec_(node->right.get(), cur, out);
        cur.pop_back();
    }

    static void collect_leaf_paths_and_actions_rec_(
        const Node* node,
        std::vector<int>& cur,
        std::vector<std::vector<int>>& paths_out,
        std::vector<int>& actions_out
    ) {
        if (!node) return;

        if (node->feature < 0) {
            paths_out.push_back(cur);
            if (node->kind == LeafKind::DEFER) {
                actions_out.push_back(-1);
            } else {
                actions_out.push_back((int)(node->pred_class & 1));
            }
            return;
        }

        const int f1 = node->feature + 1;

        cur.push_back(+f1);
        collect_leaf_paths_and_actions_rec_(node->left.get(), cur, paths_out, actions_out);
        cur.pop_back();

        cur.push_back(-f1);
        collect_leaf_paths_and_actions_rec_(node->right.get(), cur, paths_out, actions_out);
        cur.pop_back();
    }
};