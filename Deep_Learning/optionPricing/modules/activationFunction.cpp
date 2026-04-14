#include <cmath>
#include <algorithm>

class ActivationFunction {   
public:
    static double sigmoid(double x) {
        return (x >= 0) ? (1.0 / (1.0 + std::exp(-x))) : (std::exp(x) / (1.0 + std::exp(x)));
    }
    static double sigmoid_derivative(double z) {
        double a = sigmoid(z);
        return a * (1.0 - a);
    }

    static double relu(double x) {
        return x > 0 ? x : 0;
    }

    static double relu_derivative(double z) {
        return z > 0 ? 1.0 : 0.0;
    }

    static double leaky_relu(double x) {
        return x > 0 ? x : 0.01 * x;
    }

    static double leaky_relu_derivative(double z) {
        return z > 0 ? 1.0 : 0.01;
    }

    static double softplus(double x) {
        return std::log1p(std::exp(x));
    }

    static double softplus_derivative(double z) {
        return sigmoid(z);
    }
    
    static double heaviside(double x) {
        return (x >= 0.0) ? 1.0 : 0.0;
    }

    static double heaviside_derivative(double z) {
        return 0.0;
    } 
};