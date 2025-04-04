#include "NeuralNetwork.h"
#include "NeuralRenderer.h"
#include "DataLoader.h"

#include <raylib.h>
#include <string>
#include <iostream>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

int main() {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

	const int screenWidth = 1600;
    const int screenHeight = 900;

    SetConfigFlags(FLAG_MSAA_4X_HINT); 
    InitWindow(screenWidth, screenHeight, "NN from scratch");
    SetTargetFPS(60);     

    #ifdef __EMSCRIPTEN__
		const std::string weightsPath = "/weights.data"; // Absolute path in MEMFS
		const std::string imagesPath = "/mnist/train-images.idx3-ubyte"; // Absolute path in MEMFS
		const std::string labelsPath = "/mnist/train-labels.idx1-ubyte"; // Absolute path in MEMFS
		std::cout << "Running in Emscripten environment. Using MEMFS paths.\n";
	#else
		const std::string weightsPath = "./weights.data"; 
		const std::string imagesPath = "./mnist/train-images.idx3-ubyte"; 
		const std::string labelsPath = "./mnist/train-labels.idx1-ubyte"; 
		std::cout << "Running in native environment. Using local paths.\n";
	#endif


    NeuralNetwork<784, 10, 80, 80> myNN;

    #ifndef __EMSCRIPTEN__
		myNN.loadWeights(weightsPath);
    #endif

    auto images = DataLoader::load_mnist_images(imagesPath);
    auto labels = DataLoader::load_mnist_labels(labelsPath);

    if (images.rows() == 0 || labels.rows() == 0) {
         std::cerr << "Error: Failed to load MNIST data. Check paths and preloaded files.\n";
         #ifdef __EMSCRIPTEN__
             while (!WindowShouldClose()) 
             {
                 BeginDrawing();
                 ClearBackground(RAYWHITE);
                 DrawText("Error loading data. Check console.", 10, 10, 20, RED);
                 EndDrawing();
             }
         #endif
         CloseWindow();
         return 1;
    }

    Eigen::Matrix<Scalar, 1, 784> testImage = images.row(30001);
    Eigen::Matrix<Scalar, 1, 10> testLabel = labels.row(30001);
    
    int index = 0;
    while (!WindowShouldClose()) 
    {
        BeginDrawing();

        ClearBackground(RAYWHITE);
        myNN.train(images, labels, 0.001, index);
        //myNN.test(testImage, testLabel);

        NeuralRenderer::RenderNetwork<784, 10, 80, 80>(myNN);

        EndDrawing();

        if (index % 3000 == 0)
        {
            #ifdef __EMSCRIPTEN__

            #else
			myNN.saveWeights(weightsPath);
            std::cout << "Saving weights and biases... \n";
            #endif // __EMSCRIPTEN__

                    }
        index++;
        index = index % images.rows();
    }

    CloseWindow();

    return 0;
}