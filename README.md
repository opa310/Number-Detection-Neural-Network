# Neural Network Digit Recognizer

This project implements a simple neural network for recognizing handwritten digits. The project is built in C and uses the SDL2 library for rendering digit images.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Generating Test Output](#generating-test-output)
  - [Prediction](#prediction)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)

## Requirements

- `gcc` (or `x86_64-w64-mingw32-gcc` for Windows)
- `SDL2` library
- `make`
- C Standard Library

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/opa310/Number-Detection-Neural-Network.git
    cd Number-Detection-Neural-Network
    ```

2. Install the SDL2 library:
    - **Ubuntu:**
      ```sh
      sudo apt-get install libsdl2-dev
      ```
    - **MacOS:**
      ```sh
      brew install sdl2
      ```
    - **Windows:**
      Download and install the SDL2 development libraries from [libsdl.org](https://libsdl.org).

3. Unzip the training and test data:
    unzip the `digit-recognizer.zip` file with its contents inside of a folder called `digit-recognizer` 

5. Build the project using `make`:

    ```sh
    make
    ```

## Usage

### Training

To train the neural network on the MNIST dataset, run the executable and follow the prompts:

```sh
./NeuralNetwork
```

1. Enter `train` to start training the model.
2. The training data should be placed in `digit-recognizer/train.csv`.

## Note : Before generating test output or running predictions, the model must have been trained and generated the `Digit-Recogniser.csv` file

### Generating Test Output

To generate predictions on the test set:

1. Enter `gen` at the prompt.
2. The test data should be placed in `digit-recognizer/test.csv`.
3. The output will be written to `digit-recognizer/test_output.csv`.

### Prediction

To predict the digit for a specific test case:

1. Enter `test` at the prompt.
2. Enter the index of the test case.
3. The program will print the predicted digit to the console and render the image.

## Project Structure

- `Makefile` - Build script for compiling the project.
- `DigitRecogniser.c` - Main file handling user interactions, training, and testing.
- `dense.Layer.c` - Implementation of the dense layer.
- `dense.Propagation.c` - Implementation of forward and backward propagation functions.
- `dense.Layer.h` - Header file for the dense layer.
- `digit-recognizer/` - Directory containing training and test data.
  - `train.csv` - Training data file.
  - `test.csv` - Test data file.

## Acknowledgements

- The SDL2 library for rendering images.
- The MNIST dataset for providing training and testing data.


## Notes

- Before generating test output or running predictions, the model must have been trained and generated the `Digit-Recogniser.csv` file
- Ensure that the `train.csv` and `test.csv` files are formatted correctly.
- Adjust `BATCH_SIZE` and `LEARNING_RATE` as needed for optimal performance.
- The project supports Windows, Linux, and macOS. Make sure the correct libraries are installed for your OS.

For any further questions or issues, please open an issue on the project's GitHub repository.

