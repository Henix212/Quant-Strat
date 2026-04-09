#include <iostream>
#include <Eigen/Dense>
#include "modules/perceptron.cpp"

int main() {
    Eigen::MatrixXd X(4, 2);
    Eigen::VectorXd Y(4);

    X << 1, 0,
         1, 1,
         0, 1,
         0, 0;

    Y << 1.0, 1.0, 1.0, 0.0;

    Perceptron p(2, "relu");

    double learning_rate = 0.1;
    int epochs = 10000;

    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < X.rows(); i++) {
            Eigen::VectorXd xi = X.row(i);
            double y_true = Y(i);
            
            double y_pred = p.forward(xi);
            
            double error = y_pred - y_true;

            p.update_weights(xi, error, learning_rate, y_pred);
        }
    }

    std::cout << "Poids finaux : " << p.get_weights().transpose() << std::endl;
    std::cout << "Bias final   : " << p.get_bias() << std::endl;
    std::cout << "\nPredictions :" << std::endl;

    for (int i = 0; i < X.rows(); i++) {
        Eigen::VectorXd xi = X.row(i);
        std::cout << "Input: [" << xi.transpose() << "] => Pred: " 
                  << (p.forward(xi) > 0.5 ? 1 : 0) << std::endl;
    }

    return 0;
}