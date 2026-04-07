#include <iostream>
#include <random>
#include <Eigen/Dense>

class Perceptron
{   
public:
    Eigen::VectorXd weights;
    int nb_inputs;
    double bias;

    Perceptron(int nb_inputs) : nb_inputs(nb_inputs), weights(nb_inputs)
    {
        std::random_device rd;
        std::default_random_engine generator(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        bias = dist(generator);

        for (int i = 0; i < nb_inputs; i++)
        {
            weights(i) = dist(generator);
        };
    };  

    double forward(const Eigen::VectorXd& x)
    {
        return weights.dot(x) + bias;
    };

    void update_weights(const Eigen::VectorXd& x, double error, double learning_rate)
    {
        weights -= learning_rate * error * x;
        bias -= learning_rate * error;
    };
    
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

