#include <iostream>
#include <random>
#include <Eigen/Dense>
#include <functional>
#include <string>

#include "activationFunction.cpp" 

class Perceptron
{   
public:
    Eigen::VectorXd weights;
    int nb_inputs;
    double bias;

    std::function<double(double)> activation;
    std::function<double(double)> derivative;

    Perceptron(int nb_inputs, std::string func_name) : nb_inputs(nb_inputs), weights(nb_inputs)
    {
        std::random_device rd;
        std::default_random_engine generator(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        bias = dist(generator);

        for (int i = 0; i < nb_inputs; i++)
        {
            weights(i) = dist(generator);
        };

        if (func_name == "sigmoid") {
            activation = ActivationFunction::sigmoid;
            derivative = ActivationFunction::sigmoid_derivative;
        } else if (func_name == "relu") {
            activation = ActivationFunction::relu;
            derivative = ActivationFunction::relu_derivative;
        } else if (func_name == "heaviside"){ 
            activation = ActivationFunction::heaviside;
            derivative = [](double x) { return 1.0; };
        }
    };  

    double forward(const Eigen::VectorXd& x)
    {   
        double z = weights.dot(x) + bias;

        return activation(z);
    };

    void update_weights(const Eigen::VectorXd& x, double error, double learning_rate, double y_pred) {
        double gradient = error * derivative(y_pred); 
        
        weights -= learning_rate * gradient * x;
        bias -= learning_rate * gradient;
    }
    
    // Getters and setters 
    Eigen::VectorXd get_weights()
    {
        return weights;
    };

    double get_bias()
    {
        return bias;
    };
};

