#pragma once

#include "NeuralNetwork.h"
#include <raylib.h>


class NeuralRenderer
{
public:
    static void RenderImageMatrix(const Eigen::Matrix<Scalar, 1, 784>& matrix)
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

    template<size_t Input_Layer, size_t... OtherLayers>
    static void RenderNetwork(NeuralNetwork<Input_Layer, OtherLayers...>& nn)
    {
        constexpr size_t layerCount = sizeof...(OtherLayers) + 1;

        int screenWidth = GetScreenWidth();
        int screenHeight = GetScreenHeight();

        float fScreenWidth = static_cast<float>(screenWidth);
        float fScreenHeight = static_cast<float>(screenHeight);

        float startX = (fScreenWidth - (layerCount - 1) * (fScreenWidth / layerCount)) / 2.0f;

        // Draw weights (lines) between layers
        for (int i = 0; i < layerCount - 1; ++i)
        {
            int currentLayer = i;
            int nextLayer = i + 1;

            float xCurrent = startX + currentLayer * (fScreenWidth / layerCount);
            float xNext = startX + nextLayer * (fScreenWidth / layerCount);

            int currentNeurons = nn.topology[currentLayer];
            int nextNeurons = nn.topology[nextLayer];

            auto& weights = nn.weights[i];

            for (int j = 0; j < currentNeurons; ++j)
            {
                float yCurrent = (j + 1) * (fScreenHeight / (currentNeurons + 1));
                for (int k = 0; k < nextNeurons; ++k)
                {
                    float yNext = (k + 1) * (fScreenHeight / (nextNeurons + 1));
                    
                    // Get weight and calculate line color
                    Scalar weight = weights(j, k);
                    float weightAbs = std::abs(weight);
                    unsigned char intensity = static_cast<unsigned char>(std::min(weightAbs * 500.0f, 255.0f)); // Scale for visibility
                    Color lineColor = { intensity, intensity, intensity, 255 };

                    DrawLineEx(
                        Vector2{ xCurrent, yCurrent },
                        Vector2{ xNext, yNext },
                        1.5f, // Line thickness
                        lineColor
                    );
                }
            }
        }

        // Draw neurons on top of connections
        for (int i = 0; i < layerCount; ++i)
        {
            int neuronCount = nn.topology[i];
            float xPos = startX + i * (fScreenWidth / layerCount);

            for (int j = 0; j < neuronCount; ++j)
            {
                float activation = (nn.neuronActivations[i](0, j) + 1.f) / 2.f; // Squish to [0,1]
                float yPos = (j + 1) * (fScreenHeight / (neuronCount + 1));

                // Draw neuron with activation-based transparency
                DrawCircle(xPos, yPos, 4.0f, Color{ 255, 0, 0, static_cast<unsigned char>(255.f * activation) });
                //DrawRing(Vector2{ xPos, yPos }, 20.0f, 23.0f, 0, 360, 100, BLACK);
            }
        }
    }
};