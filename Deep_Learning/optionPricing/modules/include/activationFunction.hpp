#ifndef ACTIVATION_FUNCTION_HPP
#define ACTIVATION_FUNCTION_HPP

#include <cmath>

class ActivationFunction { 
public:
    static double sigmoid(double x);
    static double sigmoid_derivative(double z);

    static double relu(double x);
    static double relu_derivative(double z);

    static double leaky_relu(double x);
    static double leaky_relu_derivative(double z);

    static double softplus(double x);
    static double softplus_derivative(double z);
    
    static double heaviside(double x);
    static double heaviside_derivative(double z);
};

#endif