#pragma once
#include <bits/stdc++.h>
#include <chrono>
#include <random>
#include <fstream>

#include <raylib.h>

// Eigen Core
#include <Core>

typedef float Scalar;

inline Scalar sigmoid(Scalar x)
{
	return 1.f / (1.f + std::exp(-x));
}

template<size_t cols>
inline Eigen::Matrix<Scalar, 1, cols> softmax(Eigen::Matrix<Scalar, 1, cols>& rawOuput)
{

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

	void RenderImageMatrix(const Eigen::Matrix<Scalar, 1, 784>& matrix)
    {
        for (int i = 0; i < 784; ++i)
        {
            int r, c;
            r = i / 28;
            c = i % 28;

            Color color(matrix(0, i) * 255.f, matrix(0, i) * 255.f, matrix(0, i) * 255.f, 255.f);
            
            int squareSize = 10;
            DrawRectangle(c * squareSize, r * squareSize, squareSize, squareSize, color);
        }
    }

	bool is_correct(const Eigen::Matrix<Scalar, 1, OutputLayer>& expected)
	{
		Eigen::Index predicted_index, expected_index;
		
		output().maxCoeff(&predicted_index);
		
		expected.maxCoeff(&expected_index);
		
		return predicted_index == expected_index;
	}

	void forward(Eigen::Matrix<Scalar, 1, InputLayer>& inputMatrix)
	{
		neuronActivations[0] = inputMatrix;

		for (int i = 1; i < num_layers - 1; ++i)
		{
			neuronActivations[i] = (neuronActivations[i - 1] * weights[i - 1] + biases[i - 1]).unaryExpr([](Scalar x) { return sigmoid(x); });
		}

		neuronActivations[num_layers - 1] = soft(neuronActivations[i - 1] * weights[i - 1] + biases[i - 1]);
	}

	void backpropagate(const Eigen::Matrix<Scalar, 1, OutputLayer>& expected, Scalar learningRate)
	{
		Eigen::Matrix<Scalar, 1, Eigen::Dynamic> error = 2 * (neuronActivations[num_layers - 1] - expected);
		
		for (int i = num_layers - 2; i >= 0; --i)
		{
			Eigen::Matrix<Scalar, 1, Eigen::Dynamic> activation_derivative = neuronActivations[i + 1].array() * (1 - neuronActivations[i + 1].array());
			Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> delta = error.array() * activation_derivative.array();
			
			weights[i] -= learningRate * (neuronActivations[i].transpose() * delta);
			biases[i] -= learningRate * delta;
			
			error = delta * weights[i].transpose();
		}
	}

	void train(const Eigen::MatrixXf& images, const Eigen::MatrixXf& labels, Scalar learningRate, int index)
	{
		Eigen::Matrix<Scalar, 1, InputLayer> inputMatrix = images.row(index%images.rows());
		Eigen::Matrix<Scalar, 1, OutputLayer> expectedMatrix = labels.row(index%(int)labels.rows());

		forward(inputMatrix);

		std::cout << "Current Error - " << error(expectedMatrix) << " - " << (is_correct(expectedMatrix) ? "true" : "false") << "\n";
		RenderImageMatrix(inputMatrix);
		backpropagate(expectedMatrix, learningRate);
	}

	Eigen::Matrix<Scalar, 1, OutputLayer> output()
	{
		return neuronActivations[num_layers - 1];
	}

	Scalar error(const Eigen::Matrix<Scalar, 1, OutputLayer>& expected)
	{
		auto out = this->output();
		return (out - expected).array().square().sum();
	}

	void saveWeights(const std::string& filename)
	{
		std::ofstream file(filename, std::ios::binary);
		if (!file.is_open())
		{
			std::cerr << "Error: Could not open file for saving weights.\n";
			return;
		}

		for (size_t i = 0; i < num_layers - 1; ++i)
		{
			size_t rows = weights[i].rows();
			size_t cols = weights[i].cols();
			file.write(reinterpret_cast<const char*>(&rows), sizeof(size_t));
			file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
			file.write(reinterpret_cast<const char*>(weights[i].data()), rows * cols * sizeof(Scalar));
		}

		for (size_t i = 0; i < num_layers - 1; ++i)
		{
			size_t cols = biases[i].cols();
			file.write(reinterpret_cast<const char*>(&cols), sizeof(size_t));
			file.write(reinterpret_cast<const char*>(biases[i].data()), cols * sizeof(Scalar));
		}

		file.close();
	}

	void loadWeights(const std::string& filename)
	{
		std::ifstream file(filename, std::ios::binary);
		if (!file.is_open())
		{
			std::cerr << "Error: Could not open file for loading weights.\n";
			return;
		}

		for (size_t i = 0; i < num_layers - 1; ++i)
		{
			size_t rows, cols;
			file.read(reinterpret_cast<char*>(&rows), sizeof(size_t));
			file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));

			weights[i].resize(rows, cols);
			file.read(reinterpret_cast<char*>(weights[i].data()), rows * cols * sizeof(Scalar));
		}

		for (size_t i = 0; i < num_layers - 1; ++i)
		{
			size_t cols;
			file.read(reinterpret_cast<char*>(&cols), sizeof(size_t));

			biases[i].resize(1, cols);
			file.read(reinterpret_cast<char*>(biases[i].data()), cols * sizeof(Scalar));
		}

		file.close();
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

