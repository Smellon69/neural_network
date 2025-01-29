/**
 * @file test_neural_network.cpp
 * @brief Tests for the NeuralNetwork class using simple assert-based checks.
 */

#include <cassert>
#include <iostream>
#include "../include/neural_network.h"

namespace test_nn {

    using namespace nn;

    /**
     * @brief Simple test to see if we can partially learn XOR.
     *
     * This only does a smaller training loop than normal
     * to confirm that the network doesn't do anything obviously incorrect.
     */
    static void testXorTrainingBasic() {
        // XOR input/target pairs
        std::vector<Matrix> inputs{
            Matrix(1, 2), // [0,0]
            Matrix(1, 2), // [0,1]
            Matrix(1, 2), // [1,0]
            Matrix(1, 2)  // [1,1]
        };
        inputs[0](0, 0) = 0; inputs[0](0, 1) = 0;
        inputs[1](0, 0) = 0; inputs[1](0, 1) = 1;
        inputs[2](0, 0) = 1; inputs[2](0, 1) = 0;
        inputs[3](0, 0) = 1; inputs[3](0, 1) = 1;

        std::vector<Matrix> targets{
            Matrix(1, 1), // 0
            Matrix(1, 1), // 1
            Matrix(1, 1), // 1
            Matrix(1, 1)  // 0
        };
        targets[0](0, 0) = 0;
        targets[1](0, 0) = 1;
        targets[2](0, 0) = 1;
        targets[3](0, 0) = 0;

        // Create a small net: 2 -> 4 -> 4 -> 1
        std::vector<size_t> layerSizes = { 2, 4, 4, 1 };
        std::vector<ActivationType> activs = {
            ActivationType::ReLU,
            ActivationType::ReLU,
            ActivationType::Sigmoid
        };

        NeuralNetwork net(
            layerSizes,
            activs,
            LossType::CrossEntropy,
            OptimizerType::Momentum,
            0.05,  // learning rate
            0.9    // momentum
        );

        // Train for a small number of epochs
        int epochs = 2000;  // fewer than a real training, just a quick check
        for (int e = 0; e < epochs; ++e) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                net.trainSample(inputs[i], targets[i]);
            }
        }

        // Check if results are at least partially correct:
        for (size_t i = 0; i < inputs.size(); ++i) {
            double outVal = net.forward(inputs[i])(0, 0);
            double targetVal = targets[i](0, 0);

            // Typically XOR output should be near 0 or 1. We'll do a rough check:
            if (targetVal == 0.0) {
                // We expect outVal < 0.5
                assert((outVal < 0.5) && "Expected near 0 but got a higher value.");
            }
            else {
                // We expect outVal > 0.5
                assert((outVal > 0.5) && "Expected near 1 but got a lower value.");
            }
        }
    }

    /**
     * @brief Runs all neural-network-related tests in sequence.
     */
    void runAllNeuralNetworkTests() {
        std::cout << "[test_neural_network] Running tests...\n";
        testXorTrainingBasic();
        std::cout << "[test_neural_network] All tests passed!\n";
    }

}  // namespace test_nn
