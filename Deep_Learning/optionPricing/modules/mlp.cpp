#include "mlp.hpp"
#include "activationFunction.hpp"
#include <iostream>
#include <fstream>
#include <cmath>

Layer::Layer(int in_dim, int out_dim, const std::string& fn)
    : W(Eigen::MatrixXd::Random(out_dim, in_dim) * std::sqrt(2.0 / in_dim)),
      b(Eigen::VectorXd::Zero(out_dim)),
      activation_fn(fn) {}

Eigen::VectorXd Layer::z(const Eigen::VectorXd& input) const {
    return W * input + b;                         
}

Eigen::VectorXd Layer::activate(const Eigen::VectorXd& z) const {
    return z.unaryExpr([&](double v) {
        return ActivationFunction::apply(activation_fn, v);
    }).eval();
}

Eigen::VectorXd Layer::derivative(const Eigen::VectorXd& z) const {
    return z.unaryExpr([&](double v) {
        return ActivationFunction::derivative(activation_fn, v);
    }).eval();
}

MLP::MLP(double lr) : input_size(0), learning_rate(lr) {}

void MLP::input_layer(int nb_input) {
    input_size = nb_input;
}

void MLP::add_layer(int nb_neurons, std::string activation_fn) {
    if (layers.empty() && input_size == 0) {
        std::cerr << "CRITICAL ERROR: Call input_layer() before add_layer()!" << std::endl;
        return;
    }
    int in_dim = layers.empty() ? input_size : (int)layers.back().W.rows();
    layers.emplace_back(in_dim, nb_neurons, activation_fn);
}

std::pair<std::vector<Eigen::VectorXd>,
          std::vector<Eigen::VectorXd>>
MLP::forward_all(const Eigen::VectorXd& input) {
    std::vector<Eigen::VectorXd> zs, as;
    zs.reserve(layers.size());
    as.reserve(layers.size());

    Eigen::VectorXd current = input;
    for (const auto& layer : layers) {
        Eigen::VectorXd z = layer.z(current);      
        zs.push_back(z);
        current = layer.activate(z);               
        as.push_back(current);
    }
    return {zs, as};
}

void MLP::train(const std::vector<Eigen::VectorXd>& inputs,
                const std::vector<Eigen::VectorXd>& labels,
                int epochs, int verbose) {

    const int L = (int)layers.size() - 1;

    for (int e = 0; e < epochs; ++e) {
        double total_loss = 0;

        for (size_t s = 0; s < inputs.size(); ++s) {
            auto [zs, as] = forward_all(inputs[s]);

            std::vector<Eigen::VectorXd> deltas(layers.size());

            deltas[L] = (as[L] - labels[s]).cwiseProduct(layers[L].derivative(zs[L]));

            for (int i = L - 1; i >= 0; --i) {
                deltas[i] = (layers[i+1].W.transpose() * deltas[i+1])
                                .cwiseProduct(layers[i].derivative(zs[i]));
            }

            for (int i = 0; i < (int)layers.size(); ++i) {
                const Eigen::VectorXd& layer_in = (i == 0) ? inputs[s] : as[i-1];

                layers[i].W -= learning_rate * deltas[i] * layer_in.transpose();
                layers[i].b -= learning_rate * deltas[i];
            }

            total_loss += (as[L] - labels[s]).squaredNorm();
        }

        if (verbose && (e % 10 == 0))
            std::cout << "Epoch " << e+1 << " - Loss: "
                      << total_loss / inputs.size() << std::endl;
    }
}

Eigen::VectorXd MLP::predict(const Eigen::VectorXd& input) {
    auto [_, as] = forward_all(input);
    return as.back();
}

void MLP::save_model(std::string filename) {
    std::ofstream file(filename);
    for (size_t i = 0; i < layers.size(); ++i) {
        file << "--- Layer " << i+1 << " ---\n";
        file << " W:\n" << layers[i].W << "\n";
        file << " b: " << layers[i].b.transpose() << "\n";
    }
}