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

/**
    * @brief Main loop
*/
int main()
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

    return 0;
}