#pragma once
#include <bits/stdc++.h>
#include <chrono>
#include <random>


// Eigen Core
#include <Core>

typedef float Scalar;

inline Scalar sigmoid(Scalar x)
{
	return 1.f / (1.f + std::exp(-x));
}

template<size_t InputLayer, size_t OutputLayer, size_t... HiddenLayers>
class NeuralNetwork
{
public:
	static constexpr size_t num_layers = sizeof...(HiddenLayers) + 2; // Will always be the same with the same Layer template parameters, thus it can be constexpr static (I think?)

    NeuralNetwork()
    {
        initialize();
    }

	void forward(Eigen::Matrix<Scalar, 1, InputLayer>& inputMatrix)
	{
		neuronActivations[0] = inputMatrix;

		for (int i = 1; i < num_layers; ++i)
		{
			neuronActivations[i] = (neuronActivations[i - 1] * weights[i - 1] + biases[i - 1]).unaryExpr([](Scalar x) { return sigmoid(x); });
		}
	}

	Scalar cost(const Eigen::Matrix<Scalar, 1, OutputLayer>& out, const Eigen::Matrix<Scalar, 1, OutputLayer>& expected)
	{
		return ((out - expected) * (out - expected)).sum() / OutputLayer;
	}

	std::array<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, num_layers - 1> weights;
	std::array<Eigen::Matrix<Scalar, 1, Eigen::Dynamic>, num_layers - 1> biases;

	std::array<Eigen::Matrix<Scalar, 1, Eigen::Dynamic>, num_layers> neuronActivations;
   
	std::array<size_t, num_layers> topology = { InputLayer, HiddenLayers..., OutputLayer }; 

private:
	void initialize()
	{
		std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count()); 	
		std::normal_distribution<Scalar> weight_dist(0.0f, 1.0f); 


		for (size_t i = 0; i < num_layers - 1; ++i)
		{
			weights[i].resize(topology[i], topology[i + 1]); 
			biases[i].resize(1, topology[i + 1]);           
			neuronActivations[i].resize(1, topology[i]);     


			for (size_t r = 0; r < weights[i].rows(); ++r) {
				for (size_t c = 0; c < weights[i].cols(); ++c) {
					weights[i](r, c) = weight_dist(rng) * 0.1f; 
				}
			}
			for (size_t c = 0; c < biases[i].cols(); ++c) {
				biases[i](0, c) = weight_dist(rng) * 0.1f;
			}
		}
		neuronActivations[num_layers - 1].resize(1, topology[num_layers - 1]); 
	}


	
};

