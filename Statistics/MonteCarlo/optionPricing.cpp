#define _USE_MATH_DEFINES

#include <iostream>
#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>

struct callOptionEuParams
{
    double S; // Spot price of the underlying asset
    double K; // Strike price
    double r; // Risk-free rate
    double v; // Volatility
    double T; // Maturity
};

double monteCarloOptionPricing(const int nbSim, const callOptionEuParams params)
{
    double drift = (params.r - 0.5 * params.v * params.v) * params.T;
    double diffusion = params.v * sqrt(params.T);

    double payOffSum = 0;

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0); // Gaussian distribution with most used parameters

    for (int i = 0; i < nbSim; i++)
    {
        double Z = distribution(generator);
        double SForward = params.T * exp(drift + diffusion * Z);

        payOffSum += std::max(SForward - params.K, 0.0);
    }

    return payOffSum / nbSim * exp(-params.r * params.T);
}

int main()
{
    const int nbSimulations = 10000;

    callOptionEuParams params;
    params.S = 101.15;
    params.K = 98.01;
    params.r = 0.01;
    params.v = 0.0991;
    params.T = 0.16;

    double monteCarloPrice = monteCarloOptionPricing(nbSimulations, params);
 
    std::cout << "\n Monte Carlo Call Price: " << monteCarloPrice;
    std::cout << std::endl;
    
    return 0;
}
