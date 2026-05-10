#pragma once
#include <vector>
#include <string>
#include <Eigen/Dense>

struct Layer {
    Eigen::MatrixXd W;   // (n_out × n_in)
    Eigen::VectorXd b;   // (n_out)
    std::string activation_fn;

    Layer(int in_dim, int out_dim, const std::string& fn);

    Eigen::VectorXd z(const Eigen::VectorXd& input) const;        // W*x + b
    Eigen::VectorXd activate(const Eigen::VectorXd& z) const;     // σ(z)
    Eigen::VectorXd derivative(const Eigen::VectorXd& z) const;   // σ'(z)
};

class MLP {
public:
    MLP(double lr);
    void input_layer(int nb_input);
    void add_layer(int nb_neurons, std::string activation_fn);

    std::pair<std::vector<Eigen::VectorXd>,
              std::vector<Eigen::VectorXd>> forward_all(const Eigen::VectorXd& input);

    void   train(const std::vector<Eigen::VectorXd>& inputs,
                 const std::vector<Eigen::VectorXd>& labels,
                 int epochs, int verbose);

    Eigen::VectorXd predict(const Eigen::VectorXd& input);
    void save_model(std::string filename);

private:
    int    input_size;
    double learning_rate;
    std::vector<Layer> layers;
};