#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <iomanip>

#include "modules/mlp.cpp"

int main() {
    double learning_rate = 0.1;
    MLP mlp(learning_rate);

    mlp.input_layer(2);
    mlp.add_layer(4, "sigmoid"); 
    mlp.add_layer(1, "heaviside"); 

    std::vector<Eigen::VectorXd> inputs;
    inputs.push_back((Eigen::VectorXd(2) << 0, 0).finished());
    inputs.push_back((Eigen::VectorXd(2) << 0, 1).finished());
    inputs.push_back((Eigen::VectorXd(2) << 1, 0).finished());
    inputs.push_back((Eigen::VectorXd(2) << 1, 1).finished());

    std::vector<Eigen::VectorXd> labels;
    labels.push_back((Eigen::VectorXd(1) << 0).finished()); 
    labels.push_back((Eigen::VectorXd(1) << 1).finished()); 
    labels.push_back((Eigen::VectorXd(1) << 1).finished()); 
    labels.push_back((Eigen::VectorXd(1) << 0).finished()); 

    std::cout << "--- Debut de l'entrainement ---" << std::endl;

    mlp.train(inputs, labels, 50000, 1); 
    std::cout << "--- Fin de l'entrainement ---" << std::endl;

    std::cout << "\nResultats des tests (Seuil de decision 0.5) :" << std::endl;
    std::cout << std::fixed << std::setprecision(4);

    for (size_t i = 0; i < inputs.size(); ++i) {
        Eigen::VectorXd prediction = mlp.predict(inputs[i]);
        
        double pred = prediction(0);
                
        std::cout << "Entree: [" << inputs[i].transpose() << "] " << std::endl
                  << "Sortie brute: " << pred << std::endl;
    }

    mlp.save_model("poids_reseau.txt");
    std::cout << "\nModele sauvegarde dans 'poids_reseau.txt'" << std::endl;

    return 0;
}