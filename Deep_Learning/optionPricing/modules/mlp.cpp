#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include "perceptron.cpp"

class MLP {
private:
    std::vector<std::vector<Perceptron>> layers;
    int input_size;
    double learning_rate;

public:
    MLP(double lr) : input_size(0), learning_rate(lr) {}

    void input_layer(int nb_input) { 
        this->input_size = nb_input; 
    }

    void add_layer(int nb_neurons, std::string activation_fn) {
        if (layers.empty() && input_size == 0) {
            std::cerr << "CRITICAL ERROR: Call input_layer() before add_layer()!" << std::endl;
        }

        int in_dim = layers.empty() ? input_size : layers.back().size();
        std::vector<Perceptron> layer;
        for (int i = 0; i < nb_neurons; ++i) {
            layer.emplace_back(in_dim, activation_fn);
        }
        layers.push_back(layer);
    }

    std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>> forward_all(const Eigen::VectorXd& input) {
        std::vector<Eigen::VectorXd> zs, as;
        Eigen::VectorXd current_a = input;

        for (auto& layer : layers) {
            Eigen::VectorXd z_layer(layer.size());
            Eigen::VectorXd a_layer(layer.size());
            for (int i = 0; i < layer.size(); ++i) {
                z_layer(i) = layer[i].get_z(current_a);
                a_layer(i) = layer[i].activation(z_layer(i));
            }
            zs.push_back(z_layer);
            as.push_back(a_layer);
            current_a = a_layer;
        }
        return {zs, as};
    }

    void train(const std::vector<Eigen::VectorXd>& inputs, const std::vector<Eigen::VectorXd>& labels, int epochs, int verbose = 1) {
        for (int e = 0; e < epochs; ++e) {
            double total_loss = 0;
            for (size_t s = 0; s < inputs.size(); ++s) {
                auto [zs, as] = forward_all(inputs[s]);
                std::vector<Eigen::VectorXd> deltas(layers.size());

                int L = layers.size() - 1;
                deltas[L].resize(layers[L].size());
                for (int j = 0; j < layers[L].size(); ++j) {
                    double error = as[L](j) - labels[s](j);
                    deltas[L](j) = error * layers[L][j].derivative(zs[L](j));
                }

                for (int i = L - 1; i >= 0; --i) {
                    deltas[i].resize(layers[i].size());
                    for (int j = 0; j < layers[i].size(); ++j) {
                        double sum_delta = 0;
                        for (int k = 0; k < layers[i+1].size(); ++k) {
                            sum_delta += deltas[i+1](k) * layers[i+1][k].weights(j);
                        }
                        deltas[i](j) = sum_delta * layers[i][j].derivative(zs[i](j));
                    }
                }

                for (int i = 0; i < layers.size(); ++i) {
                    Eigen::VectorXd layer_in = (i == 0) ? inputs[s] : as[i-1];
                    for (int j = 0; j < layers[i].size(); ++j) {
                        layers[i][j].weights -= learning_rate * deltas[i](j) * layer_in;
                        layers[i][j].bias -= learning_rate * deltas[i](j);
                    }
                }
                total_loss += (as[L] - labels[s]).squaredNorm();
            }
            if (verbose && (e % 10 == 0)) 
                std::cout << "Epoch " << e+1 << " - Loss: " << total_loss/inputs.size() << std::endl;
        }
    }

    void save_model(std::string filename) {
        std::ofstream file(filename);
        for (size_t i = 0; i < layers.size(); ++i) {
            file << "--- Layer " << i+1 << " ---\n";
            for (size_t j = 0; j < layers[i].size(); ++j) {
                file << " Neuron " << j+1 << ":\n Weights: " << layers[i][j].weights.transpose() 
                     << "\n Bias: " << layers[i][j].bias << "\n";
            }
        }
    }

    Eigen::VectorXd predict(const Eigen::VectorXd& input) {
        auto [_, as] = forward_all(input);
        return as.back();
    }
};