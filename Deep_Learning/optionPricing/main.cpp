#include <iostream>
#include <Eigen/Dense>
#include "modules/perceptron.cpp" // ton perceptron

int main()
{
    Eigen::MatrixXd X(4, 2); 
    Eigen::VectorXd Y(4);    

    X << 1, 0,
         1, 1,
         0, 1,
         0, 0;

    Y << 1, 1, 1, 0; 

    Perceptron p(2);

    double learning_rate = 0.1;
    int epochs = 10000;

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        for (int i = 0; i < X.rows(); i++)
        {
            Eigen::VectorXd xi = X.row(i);   // ith input
            double y_true = Y(i);            // label
            double y_pred = p.forward(xi);   // prédiction

            double error = y_pred - y_true;  // erreur
            p.update_weights(xi, error, learning_rate); // mise à jour poids
        }
    }

    std::cout << "Poids finaux : " << p.get_weights().transpose() << std::endl;
    std::cout << "Bias final  : " << p.get_bias() << std::endl;

    std::cout << "Prédictions : " << std::endl;
    for (int i = 0; i < X.rows(); i++)
    {
        Eigen::VectorXd xi = X.row(i);
        std::cout << "Input: " << xi.transpose()
                  << " => Pred: " << p.forward(xi)
                  << " (Label: " << Y(i) << ")" << std::endl;
    }

    return 0;
}
