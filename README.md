# Minimal C++ Neural Network

This project provides a simple neural network implementation in C++, complete with:
- **Matrix** operations (multiply, add, transpose) using parallel STL transforms
- **Activation** functions (Sigmoid, ReLU, Tanh)
- **Loss** functions (MSE, Cross-Entropy)
- **Optimizers** (SGD, Momentum)
- **NeuralNetwork** class that ties everything together

It also includes basic test files to verify functionalities.

## Features

1. **Matrix** (row-major) class with parallel operations using `std::execution::par`
2. **Activation** library providing Sigmoid, ReLU, Tanh
3. **Loss** library supporting MSE and CrossEntropy
4. **Optimizers** like SGD and Momentum
5. **Feed-Forward Neural Network**:
   - Multi-layer
   - Forward pass, backprop, momentum-based weight updates

## Building

You can compile it with C++ 17+, on Windows. a Visual Studio solution file is provided.

## Why Does This Exist?

I was bored, and decided to learn how neural networks worked. This was a culmination of quite some time of knowledge.
