#define _USE_MATH_DEFINES

#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>
#include <vector>
#include <iomanip>

struct callOptionEuParams
{
    double S; // Spot price of the underlying asset
    double K; // Strike price
    double r; // Risk-free rate
    double v; // Volatility
    double T; // Maturity
    double q; // Dividend yield percentage
};

/**
    * @brief Compute the Black-Scholes formula.
    * @param params Parameters of a EU call option.
    * @return The strike price of a EU call option.
*/
double blackScholesOptionPricing(const callOptionEuParams params)
{
    const double d1 = (std::log(params.S / params.K) + (params.r - params.q + (params.v * params.v / 2.0)) * params.T) / (params.v * std::sqrt(params.T));
    const double d2 = d1 - params.v * std::sqrt(params.T);

    auto N = [](double x) {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    };

    const double price = params.S * std::exp(-params.q * params.T) * N(d1) - 
                         params.K * std::exp(-params.r * params.T) * N(d2);

    return price;
}

int create_dataset(int nbSamples, std::string filename) {
    std::ofstream file(filename);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_real_distribution<> distS(10.0, 500.0);    
    std::uniform_real_distribution<> distK_rel(0.7, 1.3);   
    std::uniform_real_distribution<> distR(0.0, 0.05);      
    std::uniform_real_distribution<> distV(0.05, 0.90);     
    std::uniform_real_distribution<> distT(0.01, 2.0);
    std::uniform_real_distribution<> distq(0.0, 5.0);      
    
    file << "S,K,r,v,T,q,Price\n";

    std::cout << "Génération de " << nbSamples << " échantillons..." << std::endl;

    for (int i = 0; i < nbSamples; ++i) {
        callOptionEuParams p;
        p.S = distS(gen);
        p.K = p.S * distK_rel(gen); 
        p.r = distR(gen);
        p.v = distV(gen);
        p.T = distT(gen);
        p.q = distq(gen); 

        double price = blackScholesOptionPricing(p);

        file << p.S << "," 
             << p.K << "," 
             << p.r << "," 
             << p.v << "," 
             << p.T << "," 
             << p.q << "," 
             << price << "\n";
        
        if (i % 10000 == 0) std::cout << "Progrès : " << i << "/" << nbSamples << std::endl;
    }

    file.close();
    std::cout << "Terminé. Dataset sauvegardé dans 'options_dataset.csv'" << std::endl;

    return 0;
}

/**
    * @brief Main loop
*/
double valide_input_bs()
{
    const int nbSimulations = 100000;

    callOptionEuParams params;
    params.S = 101.15;
    params.K = 98.01;
    params.r = 0.01;
    params.v = 0.0991;
    params.T = 0.16;
    params.q = 0;

    double price = blackScholesOptionPricing(params);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "BSM Fair-Value Price: " << price << std::endl;

    return price;
}