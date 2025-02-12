#pragma once

#include "Core"
#include <bits/stdc++.h>
#include <intrin.h>

class DataLoader
{
public:
	static uint8_t read_uint8(std::ifstream &file) 
    {
        uint8_t value = 0;
        file.read(reinterpret_cast<char*>(&value), sizeof(value));
        return value; 
	}
	
	static uint32_t read_uint32(std::ifstream &file) 
    {
		uint32_t value = 0;
		file.read(reinterpret_cast<char*>(&value), sizeof(value));
		return _byteswap_ulong(value); 
	}

	static Eigen::MatrixXf load_mnist_images(const std::string &filename) 
	{
		std::ifstream file(filename, std::ios::binary);
		if (!file.is_open()) 
		{
			throw std::runtime_error("Cannot open file: " + filename);
		}

		uint32_t magic = read_uint32(file);
		uint32_t num_images = read_uint32(file);
		uint32_t rows = read_uint32(file);
		uint32_t cols = read_uint32(file);

		std::cout << "Loading " << num_images << " images of size " << rows << "x" << cols << "...\n";

		Eigen::MatrixXf images(num_images, rows * cols);
		std::vector<uint8_t> buffer(rows * cols);

		for (size_t i = 0; i < num_images; ++i) 
		{
			file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
			for (size_t j = 0; j < buffer.size(); ++j) 
			{
				images(i, j) = buffer[j] / 255.0f;  		
			}
		}

		return images;
	}

	static Eigen::MatrixXf load_mnist_labels(const std::string& filename)
	{
		std::ifstream file(filename, std::ios::binary);
		if (!file.is_open()) 
		{
			throw std::runtime_error("Cannot open file: " + filename);
		}
		
		uint32_t magic = read_uint32(file);
		uint32_t num_labels = read_uint32(file);
		
		std::cout << "Loading " << num_labels << " Labels..." << "\n";


		Eigen::MatrixXf matrix(num_labels, 10);
		for (size_t i = 0; i < num_labels; ++i)
		{
			uint8_t byte = read_uint8(file);
			for (uint8_t j = 0; j < 10; j++)
			{
				matrix(i, j) = (j == byte ? 1 : 0);
				// std::cout << (int)byte << "\n";
			}
		}
		
		return matrix;
	}
};

