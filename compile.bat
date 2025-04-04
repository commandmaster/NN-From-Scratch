@echo on
setlocal enabledelayedexpansion
set FILES=
for %%f in (*.cpp) do set FILES=!FILES! %%f

REM Print out the files being passed to em++
echo Compiling: %FILES%

REM Run the emcc command directly (adjust paths with quotes if they have spaces)
em++ %FILES% -o wasmBuild/NeuralNetwork.html ^
    -Wall -std=c++20 -D_DEFAULT_SOURCE -Wno-missing-braces -Wunused-result -Os ^
    -I. -I "C:/raylib/src" -I "C:/raylib/src/external" -I ./Eigen ^
    -L. -L "C:/raylib/src" ^
    -s USE_GLFW=3 ^
    -s ASYNCIFY ^
    --preload-file weights.data@/weights.data ^
    --preload-file mnist/train-images.idx3-ubyte@/mnist/train-images.idx3-ubyte ^
    --preload-file mnist/train-labels.idx1-ubyte@/mnist/train-labels.idx1-ubyte ^
    -s INITIAL_MEMORY=536870912 ^
    --shell-file minshell.html ^
    "C:/raylib/src/web/libraylib.a" ^
    -DPLATFORM_WEB ^
    -s EXPORTED_FUNCTIONS=['_free','_malloc','_main','_loadWeightsFromBuffer'] ^
    -s EXPORTED_RUNTIME_METHODS=['FS','ccall','HEAPU8']

IF %ERRORLEVEL% NEQ 0 (
    echo Build failed! Check errors above.
    pause
    exit /b 1
) ELSE (
    echo Build successful.
)

pause
endlocal