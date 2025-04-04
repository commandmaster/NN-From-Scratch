#pragma once
#include <vector>
#include <array>
#include <cmath>       
#include <chrono>
#include <random>
#include <fstream>
#include <iostream>    
#include <algorithm>   
#include <cstddef>     
#include <string>      
#include <numeric>     

#include <raylib.h>
#include "Eigen/Core"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

typedef float Scalar;

inline Scalar sigmoid(Scalar x)
{
	return 1.f / (1.f + std::exp(-x));
}

template<size_t cols>
inline Eigen::Matrix<Scalar, 1, cols> softmax(const Eigen::Matrix<Scalar, 1, cols>& rawOuput)
{
	auto newMatrix = rawOuput;
	float sum = newMatrix.unaryExpr([](Scalar x) { return std::exp(x); }).sum();
	for (int i = 0; i < rawOuput.cols(); ++i)
	{
		newMatrix(0, i) = std::exp(newMatrix(0, i)) / sum;
	}

	return newMatrix;
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

	void forward(const Eigen::Matrix<Scalar, 1, InputLayer>& inputMatrix)
	{
		neuronActivations[0] = inputMatrix;

		for (int i = 1; i < num_layers - 1; ++i)
		{
			neuronActivations[i] = (neuronActivations[i - 1] * weights[i - 1] + biases[i - 1]).unaryExpr([](Scalar x) { return sigmoid(x); });
		}

		neuronActivations[num_layers - 1] = neuronActivations[num_layers - 2] * weights[num_layers - 2] + biases[num_layers - 2];
		softmaxOutput = softmax<OutputLayer>(neuronActivations[num_layers - 1]);
		neuronActivations[num_layers - 1] = (neuronActivations[num_layers - 1]).unaryExpr([](Scalar x) { return sigmoid(x); });
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

	void test(const Eigen::Matrix<Scalar, 1, InputLayer>& image, const Eigen::Matrix<Scalar, 1, OutputLayer>& label)
	{
		forward(image);
		RenderImageMatrix(image);


		std::cout << "Confidence: " << softmaxOutput.row(0) << '\n';
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

		for (uint32_t i = 0; i < num_layers - 1; ++i)
		{
			uint32_t rows = weights[i].rows();
			uint32_t cols = weights[i].cols();
			file.write(reinterpret_cast<const char*>(&rows), sizeof(uint32_t));
			file.write(reinterpret_cast<const char*>(&cols), sizeof(uint32_t));
			file.write(reinterpret_cast<const char*>(weights[i].data()), rows * cols * sizeof(Scalar));
		}

		for (uint32_t i = 0; i < num_layers - 1; ++i)
		{
			uint32_t cols = biases[i].cols();
			file.write(reinterpret_cast<const char*>(&cols), sizeof(uint32_t));
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

		for (uint32_t i = 0; i < num_layers - 1; ++i)
		{
			uint32_t rows, cols;
			file.read(reinterpret_cast<char*>(&rows), sizeof(uint32_t));
			file.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));

			weights[i].resize(rows, cols);
			file.read(reinterpret_cast<char*>(weights[i].data()), rows * cols * sizeof(Scalar));
		}

		for (uint32_t i = 0; i < num_layers - 1; ++i)
		{
			uint32_t cols;
			file.read(reinterpret_cast<char*>(&cols), sizeof(uint32_t));

			biases[i].resize(1, cols);
			file.read(reinterpret_cast<char*>(biases[i].data()), cols * sizeof(Scalar));
		}

		file.close();
	}

	bool loadWeightsFromBuffer(const char* buffer_ptr, size_t buffer_size)
	{
        std::cout << "Attempting to load weights from buffer (" << buffer_size << " bytes)..." << std::endl;
		if (!buffer_ptr || buffer_size == 0)
		{
			std::cerr << "Error: Invalid buffer provided to loadWeightsFromBuffer (null or zero size)." << std::endl;
			return false;
		}

		const char* current_ptr = buffer_ptr;
		const char* end_ptr = buffer_ptr + buffer_size; 

		for (uint32_t i = 0; i < num_layers - 1; ++i)
		{
			uint32_t rows, cols;
			size_t needed_size = sizeof(uint32_t) * 2; 

			if (current_ptr + needed_size > end_ptr) {
				std::cerr << "Error: Buffer too small while trying to read dimensions for weights layer " << i << "." << std::endl;
                std::cerr << "  Remaining buffer size: " << (end_ptr - current_ptr) << " bytes." << std::endl;
                std::cerr << "  Needed: " << needed_size << " bytes." << std::endl;
				return false;
			}

			std::memcpy(&rows, current_ptr, sizeof(uint32_t));
			current_ptr += sizeof(uint32_t);
			std::memcpy(&cols, current_ptr, sizeof(uint32_t));
			current_ptr += sizeof(uint32_t);

            if (rows != topology[i] || cols != topology[i+1]) {
                std::cerr << "Warning: Dimension mismatch in buffer for weights layer " << i
                          << ". Expected " << topology[i] << "x" << topology[i+1]
                          << ", Got " << rows << "x" << cols << ". Resizing." << std::endl;
            }


			weights[i].resize(rows, cols);

			size_t data_size = static_cast<size_t>(rows) * cols * sizeof(Scalar);
			if (current_ptr + data_size > end_ptr) {
				std::cerr << "Error: Buffer too small while trying to read data for weights layer " << i << "." << std::endl;
                std::cerr << "  Remaining buffer size: " << (end_ptr - current_ptr) << " bytes." << std::endl;
                std::cerr << "  Needed: " << data_size << " bytes." << std::endl;
				return false;
			}

			std::memcpy(weights[i].data(), current_ptr, data_size);
			current_ptr += data_size;

		}

		for (uint32_t i = 0; i < num_layers - 1; ++i)
		{
			uint32_t cols;
			size_t needed_size = sizeof(uint32_t); 

			if (current_ptr + needed_size > end_ptr) {
				std::cerr << "Error: Buffer too small while trying to read dimension for bias layer " << i << "." << std::endl;
                std::cerr << "  Remaining buffer size: " << (end_ptr - current_ptr) << " bytes." << std::endl;
                std::cerr << "  Needed: " << needed_size << " bytes." << std::endl;
				return false;
			}

			std::memcpy(&cols, current_ptr, sizeof(uint32_t));
			current_ptr += sizeof(uint32_t);

            if (cols != topology[i+1]) {
                std::cerr << "Warning: Dimension mismatch in buffer for bias layer " << i
                          << ". Expected 1x" << topology[i+1]
                          << ", Got 1x" << cols << ". Resizing." << std::endl;
            }

			biases[i].resize(1, cols);

			size_t data_size = static_cast<size_t>(cols) * sizeof(Scalar);
			if (current_ptr + data_size > end_ptr) {
				std::cerr << "Error: Buffer too small while trying to read data for bias layer " << i << "." << std::endl;
                std::cerr << "  Remaining buffer size: " << (end_ptr - current_ptr) << " bytes." << std::endl;
                std::cerr << "  Needed: " << data_size << " bytes." << std::endl;
				return false;
			}

			std::memcpy(biases[i].data(), current_ptr, data_size);
			current_ptr += data_size;
		}

        if (current_ptr != end_ptr) 
		{
            std::cerr << "Warning: Buffer not fully consumed after loading weights. "
                      << (end_ptr - current_ptr) << " bytes remaining." << std::endl;
        } 
		else 
		{
             std::cout << "Buffer fully consumed." << std::endl;
        }


		std::cout << "Weights loaded successfully from buffer." << std::endl;
		return true; 
	}

	std::array<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>, num_layers - 1> weights;
	std::array<Eigen::Matrix<Scalar, 1, Eigen::Dynamic>, num_layers - 1> biases;

	std::array<Eigen::Matrix<Scalar, 1, Eigen::Dynamic>, num_layers> neuronActivations;
	Eigen::Matrix<Scalar, 1, Eigen::Dynamic> softmaxOutput;
   
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

