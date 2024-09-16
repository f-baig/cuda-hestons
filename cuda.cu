#include "MonteCarloSimulator.h"
#include <curand_kernel.h>
#include <cmath>
#include <iostream>

__global__ void monteCarloKernel(double* d_results, double S, double K, double T, double r, double sigma, int num_simulations) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < num_simulations) {
        curandState state;
        curand_init(1234, idx, 0, &state);

        double gauss_bm = curand_normal(&state);

        double ST = S * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * gauss_bm);

        // Call
        d_results[idx] = max(ST - K, 0.0); 
    }
}

void MonteCarloSimulator::launchCUDASimulation(double* d_results) {
    int threads_per_block = 256;
    int number_of_blocks = (num_simulations + threads_per_block - 1) / threads_per_block;

    monteCarloKernel<<<number_of_blocks, threads_per_block>>>(d_results, S, K, T, r, sigma, num_simulations);

    cudaDeviceSynchronize();
}

double MonteCarloSimulator::runSimulationCUDA() {
    double* d_results;
    double* h_results = new double[num_simulations];

    cudaMalloc((void**)&d_results, num_simulations * sizeof(double));

    launchCUDASimulation(d_results);

    cudaMemcpy(h_results, d_results, num_simulations * sizeof(double), cudaMemcpyDeviceToHost);

    double payoff_sum = 0.0;
    for (int i = 0; i < num_simulations; ++i) {
        payoff_sum += h_results[i];
    }

    cudaFree(d_results);
    delete[] h_results;

    return (payoff_sum / static_cast<double>(num_simulations)) * exp(-r * T);
}
