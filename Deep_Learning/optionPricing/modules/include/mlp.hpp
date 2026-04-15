#ifndef MLP_HPP
#define MLP_HPP

#include <vector>
#include <string>
#include <Eigen/Dense>
#include "Perceptron.hpp"

class MLP {
private:
    std::vector<std::vector<Perceptron>> layers;
    int input_size;
    double learning_rate;

public:
    MLP(double lr);

    void input_layer(int nb_input);
    void add_layer(int nb_neurons, std::string activation_fn);
    
    std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>> forward_all(const Eigen::VectorXd& input);
    
    void train(const std::vector<Eigen::VectorXd>& inputs, const std::vector<Eigen::VectorXd>& labels, int epochs, int verbose = 1);
    
    void save_model(std::string filename);
    
    Eigen::VectorXd predict(const Eigen::VectorXd& input);
};

#endif