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

std::vector<double> monteCarloOptionPricing(const int nbSim, const callOptionEuParams params)
{
    double drift = (params.r - params.q - 0.5 * params.v * params.v) * params.T;
    double diffusion = params.v * sqrt(params.T);
    double payOffSum = 0;
    double payoffSquaredSum = 0;

    std::random_device rd;
    std::default_random_engine generator(rd());
    std::normal_distribution<double> distribution(0.0, 1.0);

    for (int i = 0; i < nbSim; i++)
    {
        double Z = distribution(generator);
        double SForward = params.S * exp(drift + diffusion * Z);
        double payoff = std::max(SForward - params.K, 0.0);

        payOffSum += payoff;
        payoffSquaredSum += payoff * payoff;
    }

    double meanPayoff = payOffSum / nbSim;
    double discountFactor = exp(-params.r * params.T);
    double mcPrice = meanPayoff * discountFactor;

    double variance = (payoffSquaredSum / nbSim) - (meanPayoff * meanPayoff);
    double stdError = sqrt(variance / nbSim) * discountFactor;
    double marginError = 1.96 * stdError;

    return {mcPrice,marginError};
}

int main()
{
    const int nbSimulations = 100000;

    callOptionEuParams params;
    params.S = 101.15;
    params.K = 98.01;
    params.r = 0.01;
    params.v = 0.0991;
    params.T = 0.16;
    params.q = 0.03;

    std::vector<double> monteCarloPrice = monteCarloOptionPricing(nbSimulations, params);
 
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Monte Carlo Call Price: " << monteCarloPrice[0] << std::endl;
    std::cout << "Margin error (95\% confidence): " << monteCarloPrice[1] << std::endl;

    return 0;
}
