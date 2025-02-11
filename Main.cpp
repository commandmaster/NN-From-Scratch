#include "NeuralNetwork.h"
#include "NeuralRenderer.h"
#include "DataLoader.h"

#include <raylib.h>


int main() {
	const int screenWidth = 1600;
    const int screenHeight = 900;

    SetConfigFlags(FLAG_MSAA_4X_HINT); 
    InitWindow(screenWidth, screenHeight, "NN from scratch");
    SetTargetFPS(60);     


    NeuralNetwork<784, 10, 128, 128> myNN;


    auto images = DataLoader::load_mnist_images("./t10k-images.idx3-ubyte");
       

    Eigen::Matrix<Scalar, 1, 784> inputMatrix;
    inputMatrix = images.row(0);


    myNN.forward(inputMatrix);


    while (!WindowShouldClose()) 
    {
        BeginDrawing();

            ClearBackground(RAYWHITE);

            //NeuralRenderer::RenderNetwork<8, 10, 8, 8>(myNN);

        EndDrawing();
    }

    CloseWindow();

    return 0;
}