#ifndef PERCEPTRON_HPP
#define PERCEPTRON_HPP

#include <Eigen/Dense>
#include <functional>
#include <string>
#include "ActivationFunction.hpp"
#include "activationFunction.hpp" 

class Perceptron {   
public:
    Eigen::VectorXd weights;
    double bias;

    std::function<double(double)> activation;
    std::function<double(double)> derivative;

    Perceptron(int nb_inputs, std::string func_name);
    double get_z(const Eigen::VectorXd& x);
};

#endif