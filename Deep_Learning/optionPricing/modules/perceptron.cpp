#include "Perceptron.hpp" 
#include <iostream>
#include <random>
#include <cmath>

Perceptron::Perceptron(int nb_inputs, std::string func_name) : weights(nb_inputs) {
    std::random_device rd;
    std::default_random_engine gen(rd());
    
    double limit = std::sqrt(6.0 / (nb_inputs + 1));
    std::uniform_real_distribution<double> dist(-limit, limit);

    bias = dist(gen);
    for (int i = 0; i < nb_inputs; i++) {
        weights(i) = dist(gen);
    }

    if (func_name == "sigmoid") {
        activation = ActivationFunction::sigmoid;
        derivative = ActivationFunction::sigmoid_derivative;
    } else if (func_name == "relu") {
        activation = ActivationFunction::relu;
        derivative = ActivationFunction::relu_derivative;
    } else if (func_name == "leaky_relu") {
        activation = ActivationFunction::leaky_relu;
        derivative = ActivationFunction::leaky_relu_derivative;
    } else if (func_name == "softplus") {
        activation = ActivationFunction::softplus;
        derivative = ActivationFunction::softplus_derivative;
    } else {
        std::cerr << "Warning: Unknown activation '" << func_name << "'. Defaulting to sigmoid." << std::endl;
        activation = ActivationFunction::sigmoid;
        derivative = ActivationFunction::sigmoid_derivative;
    }
}

double Perceptron::get_z(const Eigen::VectorXd& x) {
    return weights.dot(x) + bias;
}