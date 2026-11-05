#ifndef CLARI_TREE_HPP
#define CLARI_TREE_HPP
#pragma once

#include <string>
#include <vector>
#include <set>
#include <Eigen/Dense>
#include <cstddef>
#include <limits>
#include <tuple>

// Depth is an int across the tree API.
using Depth = int;

enum class LeafType {
    CONSTANT = 0,
    LINEAR = 1,
    DEFER = 2
};

//
// ========== Node ==========
// Represents one node in the regression tree
//
class Node {
public:
    Node* left;                   // left child (nullptr if leaf)
    Node* right;                  // right child (nullptr if leaf)
    bool is_leaf;                 // flag for leaf node
    std::size_t n_instances;      // number of samples routed to this node
    double obj;                   // objective value at this node
    double threshold;             // threshold used for splitting
    int feature_idx;              // feature index used for splitting
    Eigen::VectorXd coefficients; // ridge regression coefficients

    LeafType leaf_type; // CONSTANT, LINEAR, or DEFER
    double constant_prediction;   // prediction used by constant leaves

    Node();
    ~Node();

    Node& operator=(const Node& other);   // copy assignment
    Node(const Node& other);              // copy constructor

    // std::string print_tree(int indentation = 0); 
    std::string print_tree(
        int indentation = 0,
        const std::vector<int>& continuous_idx = {}
    ) const; // print subtree
};

struct PathCondition {
    int feature_idx = -1;
    double threshold = 0.0;
    bool is_leq = true;  // true: x_j <= threshold, false: x_j > threshold
};

struct LeafPathExport {
    std::vector<PathCondition> conditions;

    LeafType leaf_type = LeafType::LINEAR;

    // leaf metadata
    double obj = 0.0;
    std::size_t n_instances = 0;

    // constant leaf
    double constant_prediction = 0.0;

    // linear leaf
    Eigen::VectorXd coefficients;

    // for linear leaves, coefficients are:
    // coefficients[0] = intercept
    // coefficients[1 + k] corresponds to continuous_idx[k]
    std::vector<int> continuous_idx;

    // dEFER leaf has no coefficients; prediction is reference_pred
};

//
// ========== Greedy ==========
// Greedy regression tree with ridge regression in each leaf
//
class Greedy {
public:
    Eigen::MatrixXd X;   // feature matrix
    Eigen::VectorXd y;   // target vector
    double kappa;        // ridge regularization parameter
    bool verbose;        // verbosity flag
    Depth depth;         // maximum depth
    unsigned long int n; // number of samples
    unsigned long int m; // number of features
    double lambda;       // penalty on number of leaf nodes
    double scaled_kappa; // scaled by n
    double scaled_lambda;// scaled by TSS
    double rho;          // extra local cost per sample for linear leaves
    double eta;          // extra local cost per sample for defer leaves
    int n_thresholds;    // maximum number of thresholds per continuous feature
    std::string thresholds_strategy; // threshold generation strategy
    std::vector<int> continuous_idx_;  // indices in X (excluding intercept col 0)
    std::vector<int> binary_idx_;      // indices in X (excluding intercept col 0)
    std::vector<int> categorical_idx_; // optional, provided by caller (excluding intercept col 0)
    Eigen::MatrixXd X_reg_;   // [n x p_reg] = [1 | continuous features]
    int p_reg_ = 0;           // = 1 + continuous_idx_.size()
    int p_split_ = 0;         // = X.cols() (original with intercept)
    Eigen::VectorXd x_mean_;  // mean of continuous features (for standardization)
    Eigen::VectorXd x_std_;   // std of continuous features (for standardization)
    double y_mean_ = 0.0;     // mean of y (for centering)
    Eigen::VectorXd reference_pred_centered_; // reference predictions centered by y_mean_
    Eigen::VectorXd defer_resid_sq_;          // (centered y - centered reference)^2
    bool has_reference_pred_ = false;         // true when fit(..., reference_pred, ...) is used
    
    Eigen::VectorXd sample_weight_;
    double total_weight_ = 0.0;

    int min_leaf_node_size;   // requested minimum samples per leaf; <= 0 means auto
    Node* root;          // root node

    std::vector<LeafPathExport> export_leaf_paths() const;

    std::string print_leaf_paths() const;

    Greedy(double kappa, Depth depth, double lambda = 0.0, int n_thresholds = 1, bool verbose = true, int min_leaf_node_size = 0);
    Greedy(double kappa, Depth depth, double lambda, int n_thresholds, const std::string& thresholds_strategy, bool verbose = true, int min_leaf_node_size = 0);

    // three-leaf objective constructors.
    Greedy(double kappa, Depth depth, double lambda, double rho, double eta,
        int n_thresholds = 1, bool verbose = true, int min_leaf_node_size = 0);

    Greedy(double kappa, Depth depth, double lambda, double rho, double eta,
        int n_thresholds, const std::string& thresholds_strategy,
        bool verbose = true, int min_leaf_node_size = 0);

    
    virtual ~Greedy();

    Greedy& operator=(const Greedy& other);
    Greedy(const Greedy& other);

    double fit(Eigen::MatrixXd X, Eigen::VectorXd y, const std::vector<int>& categorical_idx = {});

    // three-leaf fit with reference predictions.
    double fit(Eigen::MatrixXd X,
            Eigen::VectorXd y,
            Eigen::VectorXd reference_pred,
            const std::vector<int>& categorical_idx = {});

    double fit_weighted(Eigen::MatrixXd X,
                    Eigen::VectorXd y,
                    Eigen::VectorXd sample_weight,
                    const std::vector<int>& categorical_idx = {});

    double fit(Eigen::MatrixXd X,
            Eigen::VectorXd y,
            Eigen::VectorXd reference_pred,
            Eigen::VectorXd sample_weight,
            const std::vector<int>& categorical_idx = {});

    double fit_with_reference_and_weights(Eigen::MatrixXd X,
                                        Eigen::VectorXd y,
                                        Eigen::VectorXd reference_pred,
                                        Eigen::VectorXd sample_weight,
                                        const std::vector<int>& categorical_idx = {});

    void fit_coefficients(Node* node,
                        Eigen::MatrixXd X,
                        Eigen::VectorXd y,
                        const std::vector<int>& original_rows);

    virtual double recursive_fit(
        std::vector<std::vector<unsigned long int>>& sorted_indices,
        Eigen::LLT<Eigen::MatrixXd>& llt,
        Eigen::VectorXd& b,
        double y_sum_sq,
        Node* node,
        Depth depth_remaining
    );

    // double loss(Eigen::MatrixXd L, Eigen::VectorXd b, double y_sum_sq);
    double loss(const Eigen::LLT<Eigen::MatrixXd>& llt,
            const Eigen::VectorXd& b,
            double y_sum_sq) const;

    double constant_loss(double weight_sum, double wy_sum, double wy_sum_sq) const;

    double best_three_leaf_objective(double weight_sum,
                                    double wy_sum,
                                    double wy_sum_sq,
                                    const Eigen::LLT<Eigen::MatrixXd>& llt,
                                    const Eigen::VectorXd& b,
                                    double defer_sse) const;

    LeafType best_leaf_type(double weight_sum,
                            double wy_sum,
                            double wy_sum_sq,
                            const Eigen::LLT<Eigen::MatrixXd>& llt,
                            const Eigen::VectorXd& b,
                            double defer_sse) const;

    double sum_weight_from_sorted_indices(
        const std::vector<std::vector<unsigned long int>>& sorted_indices
    ) const;

    double sum_weighted_y_from_sorted_indices(
        const std::vector<std::vector<unsigned long int>>& sorted_indices
    ) const;

    double sum_weighted_defer_sse_from_sorted_indices(
        const std::vector<std::vector<unsigned long int>>& sorted_indices
    ) const;

    Eigen::VectorXd predict(Eigen::MatrixXd X);
    Eigen::VectorXd predict(Eigen::MatrixXd X, Eigen::VectorXd reference_pred);

    double predict_row(Eigen::VectorXd x);
    double predict_row(Eigen::VectorXd x, double reference_value);

    std::string print_tree();
    std::vector<std::vector<double>> get_traversed_thresholds() const;
    std::string print_traversed_thresholds() const;
    std::vector<std::vector<double>> get_threshold_pool() const;
    std::string print_threshold_pool() const;
    std::size_t n_leaves() const;

    // --- NEW helpers---
    static bool is_binary_column(const Eigen::VectorXd& col);
    void detect_feature_types(); // fills *_idx_, sets p_reg_, p_split_

    // Row in regression space [1 | x_cont]
    Eigen::RowVectorXd reg_row(int i) const;

    // Full recompute (regression view) for a subset of rows
    std::tuple<Eigen::LLT<Eigen::MatrixXd>, Eigen::VectorXd, double>
    recompute_stats_from_rows(const std::vector<int>& rows);

protected:
    void resolve_min_leaf_node_size();
    bool children_respect_min_leaf_size(double left_weight, double right_weight) const;
    void reset_traversed_thresholds();
    void record_traversed_threshold(unsigned long int feature_idx, double threshold);
    void build_threshold_pool(const std::vector<std::vector<unsigned long int>>& sorted_indices);
    std::vector<std::set<double>> traversed_thresholds_;
    std::vector<std::vector<double>> threshold_pool_;
    std::size_t resolved_min_leaf_node_size_ = 1;

private:
    static std::size_t count_leaves(const Node* n);
    double recursive_fit_simplified_candidate(
        std::vector<std::vector<unsigned long int>> sorted_indices,
        Eigen::LLT<Eigen::MatrixXd> llt,
        Eigen::VectorXd b,
        double y_sum_sq,
        Node* node,
        Depth depth_remaining
    );

    void export_leaf_paths_rec_(
        const Node* node,
        std::vector<PathCondition>& cur,
        std::vector<LeafPathExport>& out
    ) const;

    static std::string leaf_type_to_string_(LeafType t);

  
};

//
// ========== CLARITree ==========
// Main CLARITree algorithm with recursive splitting strategy
//
class CLARITree : public Greedy {
public:
    CLARITree(double kappa, Depth depth, double lambda = 0.0, int n_thresholds = 1, bool verbose = true, int min_leaf_node_size = 0);
    CLARITree(double kappa, Depth depth, double lambda, int n_thresholds, const std::string& thresholds_strategy, bool verbose = true, int min_leaf_node_size = 0);

    // three-leaf objective constructors.
    CLARITree(double kappa, Depth depth, double lambda, double rho, double eta,
            int n_thresholds = 1, bool verbose = true, int min_leaf_node_size = 0);

    CLARITree(double kappa, Depth depth, double lambda, double rho, double eta,
            int n_thresholds, const std::string& thresholds_strategy,
            bool verbose = true, int min_leaf_node_size = 0);

    double recursive_fit(
        std::vector<std::vector<unsigned long int>>& sorted_indices,
        Eigen::LLT<Eigen::MatrixXd>& llt,
        Eigen::VectorXd& b,
        double y_sum_sq,
        Node* node,
        Depth depth_remaining
    ) override;
};

#endif // CLARI_TREE_HPP
