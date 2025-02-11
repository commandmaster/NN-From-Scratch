#include "NeuralNetwork.h"
#include "NeuralRenderer.h"

#include <raylib.h>


int main() {
	const int screenWidth = 1600;
    const int screenHeight = 900;

    SetConfigFlags(FLAG_MSAA_4X_HINT); 
    InitWindow(screenWidth, screenHeight, "NN from scratch");
    SetTargetFPS(60);     


    NeuralNetwork<3, 8, 8, 3> myNN;

    Eigen::Matrix<Scalar, 1, 3> inputMatrix;
    inputMatrix << 0.5f, 1.0f, 0.2f; 

    myNN.forward(inputMatrix);


    while (!WindowShouldClose()) 
    {
        BeginDrawing();

            ClearBackground(RAYWHITE);

            NeuralRenderer::RenderNetwork<3, 8, 8, 3>(myNN);

        EndDrawing();
    }

    CloseWindow();

    return 0;
}