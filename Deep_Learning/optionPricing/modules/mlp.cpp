#include <iostream>
#include <fstream>
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
        int in_dim = layers.empty() ? input_size : layers.back().size();
        std::vector<Perceptron> layer;
        for (int i = 0; i < nb_neurons; ++i) {
            layer.emplace_back(in_dim, activation_fn);
        }
        layers.push_back(layer);
    }

    std::pair<Eigen::VectorXd, std::vector<Eigen::VectorXd>> forward_all(const Eigen::VectorXd& inputs) {
        std::vector<Eigen::VectorXd> activations;
        Eigen::VectorXd current_in = inputs;

        for (auto& layer : layers) {
            Eigen::VectorXd next_in(layer.size());
            for (int i = 0; i < layer.size(); ++i) {
                next_in(i) = layer[i].forward(current_in);
            }
            activations.push_back(next_in);
            current_in = next_in;
        }
        return {current_in, activations};
    }

    void train(const std::vector<Eigen::VectorXd>& inputs, const std::vector<Eigen::VectorXd>& labels, int epochs, int verbose = 0) {
        for (int e = 0; e < epochs; ++e) {
            double total_loss = 0;

            for (size_t s = 0; s < inputs.size(); ++s) {
                auto [pred, acts] = forward_all(inputs[s]);

                std::vector<Eigen::VectorXd> deltas(layers.size());

                int last = layers.size() - 1;
                deltas[last].resize(layers[last].size());
                for (int j = 0; j < layers[last].size(); ++j) {
                    double local_error = pred(j) - labels[s](j);
                    deltas[last](j) = local_error * layers[last][j].derivative(pred(j));
                }

                for (int i = last - 1; i >= 0; --i) {
                    deltas[i].resize(layers[i].size());
                    for (int j = 0; j < layers[i].size(); ++j) {
                        double sum_delta = 0;
                        for (int k = 0; k < layers[i+1].size(); ++k) {
                            sum_delta += deltas[i+1](k) * layers[i+1][k].weights(j);
                        }
                        deltas[i](j) = sum_delta * layers[i][j].derivative(acts[i](j));
                    }
                }

                for (int i = 0; i < layers.size(); ++i) {
                    Eigen::VectorXd layer_input = (i == 0) ? inputs[s] : acts[i-1];
                    for (int j = 0; j < layers[i].size(); ++j) {
                        layers[i][j].weights -= learning_rate * deltas[i](j) * layer_input;
                        layers[i][j].bias -= learning_rate * deltas[i](j);
                    }
                }
                total_loss += (pred - labels[s]).squaredNorm();
            }
            if (e % 10 == 0 && verbose == 1) std::cout << "Epoch " << e+1 << " - Loss: " << total_loss/inputs.size() << std::endl;
        }
    }

    void save_model(std::string filename) {
        std::fstream file(filename,std::ofstream::out);
        for (size_t i = 0; i < layers.size(); ++i) {
            file << "--- Couche " << i+1 << " ---\n";
            for (size_t j = 0; j < layers[i].size(); ++j) {
                file << " Neurone " << j+1 << ":\n Poids: " << layers[i][j].weights.transpose() << "\n Biais: " << layers[i][j].bias << "\n";
            }
        }
    }

    Eigen::VectorXd predict(const Eigen::VectorXd& input) {
        auto [output, _] = forward_all(input);
        
        return output;
    }
};