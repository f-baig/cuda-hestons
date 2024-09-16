#include <iostream>
#include "MonteCarloSimulator.h"

int main() {
    double S = 100.0; 
    double K = 100.0;
    double T = 1.0;  
    double r = 0.05;   
    double sigma = 0.2;
    int num_simulations = 1000000;

    MonteCarloSimulator simulator(S, K, T, r, sigma, num_simulations);

    double option_price_cpu = simulator.runSimulation();
    std::cout << "European Call Option Price (CPU): " << option_price_cpu << std::endl;

    double option_price_gpu = simulator.runSimulationCUDA();
    std::cout << "European Call Option Price (GPU): " << option_price_gpu << std::endl;

    return 0;
}
