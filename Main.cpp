#include "NeuralNetwork.h"
#include "NeuralRenderer.h"
#include "DataLoader.h"

#include <raylib.h>


int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

	const int screenWidth = 1600;
    const int screenHeight = 900;

    SetConfigFlags(FLAG_MSAA_4X_HINT); 
    InitWindow(screenWidth, screenHeight, "NN from scratch");
    SetTargetFPS(60);     


    NeuralNetwork<784, 10, 80, 80> myNN;
    myNN.loadWeights("./weights.data");



    auto images = DataLoader::load_mnist_images("./mnist/train-images.idx3-ubyte");
    auto labels = DataLoader::load_mnist_labels("./mnist/train-labels.idx1-ubyte");

    
    int index = 0;
    while (!WindowShouldClose()) 
    {
        BeginDrawing();


        ClearBackground(RAYWHITE);
        myNN.train(images, labels, 0.001, index);

        NeuralRenderer::RenderNetwork<784, 10, 80, 80>(myNN);

        EndDrawing();

        if (index % 3000 == 0)
        {
            myNN.saveWeights("./weights.data");
            std::cout << "Saving weights and biases... \n";
        }
        index++;
    }

    CloseWindow();

    return 0;
}