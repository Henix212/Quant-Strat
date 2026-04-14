#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <iomanip>
#include <string>

#include "modules/mlp.cpp"
#include "../../Statistics/Black-Scholes/optionPricing.cpp"

std::string filename = "options_dataset.csv";
int nbSamples = 10000;

std::vector<Eigen::VectorXd> inputs;
std::vector<Eigen::VectorXd> labels;

void load_dataset(const std::string& filename, 
                  std::vector<Eigen::VectorXd>& inputs, 
                  std::vector<Eigen::VectorXd>& labels) {
    
    std::ifstream file(filename);
    std::string line;
    
    if (!file.is_open()) {
        std::cerr << "Erreur : Impossible d'ouvrir le fichier " << filename << std::endl;
        return;
    }

    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        std::vector<double> row_values;

        while (std::getline(ss, value, ',')) {
            row_values.push_back(std::stod(value));
        }

        Eigen::VectorXd input(6);
        for (int i = 0; i < 6; ++i) {
            input(i) = row_values[i];
        }
        inputs.push_back(input);

        Eigen::VectorXd label(1);
        label(0) = row_values[6];
        labels.push_back(label);
    }

    file.close();
    std::cout << inputs.size() << std::endl;
}

int main() {
    const double learning_rate = 0.0001; 
    const int epochs = 500;
    const std::string filename = "options_dataset.csv";

    load_dataset(filename, inputs, labels);

    if (inputs.empty()) return 1;

    Eigen::VectorXd min_v = inputs[0], max_v = inputs[0];

    for (const auto& in : inputs) {
        for (int i = 0; i < 6; ++i) {
            min_v(i) = std::min(min_v(i), in(i));
            max_v(i) = std::max(max_v(i), in(i));
        }
    }

    for (auto& in : inputs) {
        for (int i = 0; i < 6; ++i) {
            double range = max_v(i) - min_v(i);
            in(i) = (range == 0) ? 0 : (in(i) - min_v(i)) / range;
        }
    }

    MLP mlp(learning_rate);

    mlp.input_layer(6);

    mlp.add_layer(32, "softplus"); 
    mlp.add_layer(16, "softplus"); 
    mlp.add_layer(8, "softplus");
    mlp.add_layer(1, "leaky_relu");

    std::cout << "\n--- Training Starts ---" << std::endl;
    mlp.train(inputs, labels, epochs, 1);

    Eigen::VectorXd test(6);
    test << 101.15, 98.01, 0.01, 0.0991, 0.16, 0;
    
    for (int i = 0; i < 6; ++i) {
        test(i) = (test(i) - min_v(i)) / (max_v(i) - min_v(i));
    }

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Prediction for normalized test: " << mlp.predict(test)(0) << std::endl;

    return 0;
}