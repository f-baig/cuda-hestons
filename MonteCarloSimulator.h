#ifndef MONTECARLOSIMULATOR_H
#define MONTECARLOSIMULATOR_H

class MonteCarloSimulator {
public:
    // Constructor
    MonteCarloSimulator(double S, double K, double T, double r, double sigma, int num_simulations);

    // Run CPU-based simulation
    double runSimulation();

    // Run CUDA-accelerated simulation
    double runSimulationCUDA();

private:
    // Variables defined same as general formulas 
    double S;
    double K;
    double T;
    double r;
    double sigma;
    int num_simulations;

    double normalRandom();
    double generatePricePath();

    // CUDA
    void launchCUDASimulation(double* d_results);
};

#endif
