#include "activationFunction.hpp"
#include <cmath>
#include <algorithm>

// SIGMOID
double ActivationFunction::sigmoid(double x) {
    return (x >= 0) ? (1.0 / (1.0 + std::exp(-x))) : (std::exp(x) / (1.0 + std::exp(x)));
}

double ActivationFunction::sigmoid_derivative(double z) {
    double a = sigmoid(z);
    return a * (1.0 - a);
}

// RELU
double ActivationFunction::relu(double x) {
    return x > 0 ? x : 0;
}

double ActivationFunction::relu_derivative(double z) {
    return z > 0 ? 1.0 : 0.0;
}

// LEAKY RELU
double ActivationFunction::leaky_relu(double x) {
    return x > 0 ? x : 0.01 * x;
}

double ActivationFunction::leaky_relu_derivative(double z) {
    return z > 0 ? 1.0 : 0.01;
}

// SOFTPLUS
double ActivationFunction::softplus(double x) {
    return std::log1p(std::exp(x));
}

double ActivationFunction::softplus_derivative(double z) {
    return sigmoid(z);
}

// HEAVISIDE
double ActivationFunction::heaviside(double x) {
    return (x >= 0.0) ? 1.0 : 0.0;
}

double ActivationFunction::heaviside_derivative(double z) {
    return 0.0;
}

double ActivationFunction::apply(const std::string& name, double x) {
    if (name == "sigmoid") return sigmoid(x);
    if (name == "relu") return relu(x);
    if (name == "leaky_relu") return leaky_relu(x);
    if (name == "softplus") return softplus(x);
    if (name == "heaviside") return heaviside(x);
    return sigmoid(x); // Default
}

double ActivationFunction::derivative(const std::string& name, double x) {
    if (name == "sigmoid") return sigmoid_derivative(x);
    if (name == "relu") return relu_derivative(x);
    if (name == "leaky_relu") return leaky_relu_derivative(x);
    if (name == "softplus") return softplus_derivative(x);
    if (name == "heaviside") return heaviside_derivative(x);
    return sigmoid_derivative(x); // Default
}