@echo on
setlocal enabledelayedexpansion
set FILES=
for %%f in (*.cpp) do set FILES=!FILES! %%f

REM Print out the files being passed to em++
echo !FILES!

REM Run the emcc command and output errors to a log file
powershell -Command "em++ -o wasmBuild/NeuralNetwork.html !FILES! -Wall -std=c++20 -D_DEFAULT_SOURCE -Wno-missing-braces -Wunused-result -Os -I. -I C:/raylib/src -I C:/raylib/src/external -I ./Eigen -L. -L C:/raylib/src -s USE_GLFW=3 -s ASYNCIFY --preload-file weights.data@/weights.data --preload-file mnist/train-images.idx3-ubyte@/mnist/train-images.idx3-ubyte --preload-file mnist/train-labels.idx1-ubyte@/mnist/train-labels.idx1-ubyte -s TOTAL_MEMORY=536870912 -s FORCE_FILESYSTEM=1 --shell-file minshell.html C:/raylib/src/web/libraylib.a -DPLATFORM_WEB -s 'EXPORTED_FUNCTIONS=[\"_free\",\"_malloc\",\"_main\"]' -s EXPORTED_RUNTIME_METHODS=ccall"

pause