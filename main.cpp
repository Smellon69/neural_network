#include <iostream>
#include <vector>
#include <string>
#include "include/matrix.h"
#include "include/activation.h"
#include "include/loss.h"
#include "include/optimizer.h"
#include "include/neural_network.h"

using namespace nn;

/**
 * @brief Trains a small binary classifier network on the given input->target dataset,
 *        then prints final predictions.
 *
 * @param name Name of the task (e.g. "XOR").
 * @param inputs Vector of input matrices (each 1x2 for a 2-bit logic).
 * @param targets Vector of target matrices (each 1x1, 0 or 1).
 * @param layerSizes Network architecture (e.g., {2, 4, 1}).
 * @param activs Activations for each layer except input (e.g., {ReLU, Sigmoid}).
 * @param lossType Loss function to use (e.g., CrossEntropy).
 * @param optType Optimizer type (e.g., Momentum).
 * @param lr Learning rate.
 * @param momentum Momentum factor (if used by the optimizer).
 * @param epochs Number of training epochs.
 * @param logInterval Print loss every this many epochs.
 */
void trainAndTestBinaryFunction(const std::string& name,
    const std::vector<Matrix>& inputs,
    const std::vector<Matrix>& targets,
    const std::vector<size_t>& layerSizes,
    const std::vector<ActivationType>& activs,
    LossType lossType,
    OptimizerType optType,
    double lr,
    double momentum,
    int epochs,
    int logInterval = 1000)
{
    // Construct the network
    NeuralNetwork net(layerSizes, activs, lossType, optType, lr, momentum);

    // Train
    for (int e = 1; e <= epochs; ++e) {
        double totalLoss = 0.0;
        // Here we do single-sample training across all samples
        for (size_t i = 0; i < inputs.size(); ++i) {
            double lossVal = net.trainSample(inputs[i], targets[i]);
            totalLoss += lossVal;
        }
        if (e % logInterval == 0) {
            std::cout << name << " | Epoch " << e
                << " | Loss: " << totalLoss << std::endl;
        }
    }

    // Test / Print results
    std::cout << "\n[" << name << "] Final Predictions:\n";
    for (size_t i = 0; i < inputs.size(); ++i) {
        Matrix out = net.forward(inputs[i]);
        std::cout << "Input: (";
        for (size_t c = 0; c < inputs[i].cols(); ++c) {
            std::cout << inputs[i](0, c) << ((c + 1 < inputs[i].cols()) ? ", " : "");
        }
        std::cout << ") -> " << out(0, 0)
            << " (target: " << targets[i](0, 0) << ")\n";
    }
    std::cout << std::endl;
}

int main() {
    // --------------------------------------------------------------------------
    // 1) Logic: AND
    // 2-bit input => 1-bit output
    // --------------------------------------------------------------------------
    {
        std::vector<Matrix> inputs{
            Matrix(1,2), Matrix(1,2), Matrix(1,2), Matrix(1,2)
        };
        std::vector<Matrix> targets{
            Matrix(1,1), Matrix(1,1), Matrix(1,1), Matrix(1,1)
        };
        // (0,0) => 0
        inputs[0](0, 0) = 0; inputs[0](0, 1) = 0;  targets[0](0, 0) = 0;
        // (0,1) => 0
        inputs[1](0, 0) = 0; inputs[1](0, 1) = 1;  targets[1](0, 0) = 0;
        // (1,0) => 0
        inputs[2](0, 0) = 1; inputs[2](0, 1) = 0;  targets[2](0, 0) = 0;
        // (1,1) => 1
        inputs[3](0, 0) = 1; inputs[3](0, 1) = 1;  targets[3](0, 0) = 1;

        // Architecture: 2 -> 4 -> 1
        std::vector<size_t> layerSizes = { 2, 4, 1 };
        // Activations: hidden => ReLU, output => Sigmoid
        std::vector<ActivationType> activs = {
            ActivationType::ReLU,
            ActivationType::Sigmoid
        };

        trainAndTestBinaryFunction("AND", inputs, targets,
            layerSizes, activs,
            LossType::CrossEntropy,
            OptimizerType::Momentum,
            /*lr=*/0.05, /*momentum=*/0.9,
            /*epochs=*/5000,
            /*logInterval=*/1000);
    }

    // --------------------------------------------------------------------------
    // 2) Logic: OR
    // --------------------------------------------------------------------------
    {
        std::vector<Matrix> inputs{
            Matrix(1,2), Matrix(1,2), Matrix(1,2), Matrix(1,2)
        };
        std::vector<Matrix> targets{
            Matrix(1,1), Matrix(1,1), Matrix(1,1), Matrix(1,1)
        };
        // (0,0) => 0
        inputs[0](0, 0) = 0; inputs[0](0, 1) = 0;  targets[0](0, 0) = 0;
        // (0,1) => 1
        inputs[1](0, 0) = 0; inputs[1](0, 1) = 1;  targets[1](0, 0) = 1;
        // (1,0) => 1
        inputs[2](0, 0) = 1; inputs[2](0, 1) = 0;  targets[2](0, 0) = 1;
        // (1,1) => 1
        inputs[3](0, 0) = 1; inputs[3](0, 1) = 1;  targets[3](0, 0) = 1;

        std::vector<size_t> layerSizes = { 2, 4, 1 };
        std::vector<ActivationType> activs = {
            ActivationType::ReLU,
            ActivationType::Sigmoid
        };

        trainAndTestBinaryFunction("OR", inputs, targets,
            layerSizes, activs,
            LossType::CrossEntropy,
            OptimizerType::Momentum,
            0.05, 0.9,
            5000,
            1000);
    }

    // --------------------------------------------------------------------------
    // 3) Logic: XOR
    // --------------------------------------------------------------------------
    {
        std::vector<Matrix> inputs{
            Matrix(1,2), Matrix(1,2), Matrix(1,2), Matrix(1,2)
        };
        std::vector<Matrix> targets{
            Matrix(1,1), Matrix(1,1), Matrix(1,1), Matrix(1,1)
        };
        // (0,0) => 0
        inputs[0](0, 0) = 0; inputs[0](0, 1) = 0;  targets[0](0, 0) = 0;
        // (0,1) => 1
        inputs[1](0, 0) = 0; inputs[1](0, 1) = 1;  targets[1](0, 0) = 1;
        // (1,0) => 1
        inputs[2](0, 0) = 1; inputs[2](0, 1) = 0;  targets[2](0, 0) = 1;
        // (1,1) => 0
        inputs[3](0, 0) = 1; inputs[3](0, 1) = 1;  targets[3](0, 0) = 0;

        // This time let's do 2 -> 4 -> 4 -> 1
        std::vector<size_t> layerSizes = { 2, 4, 4, 1 };
        std::vector<ActivationType> activs = {
            ActivationType::ReLU,  // hidden 1
            ActivationType::ReLU,  // hidden 2
            ActivationType::Sigmoid // output
        };

        trainAndTestBinaryFunction("XOR", inputs, targets,
            layerSizes, activs,
            LossType::CrossEntropy,
            OptimizerType::Momentum,
            0.05, 0.9,
            10000, // a bit more training for XOR
            2000);
    }

    // --------------------------------------------------------------------------
    // 4) Logic: NAND
    // --------------------------------------------------------------------------
    {
        std::vector<Matrix> inputs{
            Matrix(1,2), Matrix(1,2), Matrix(1,2), Matrix(1,2)
        };
        std::vector<Matrix> targets{
            Matrix(1,1), Matrix(1,1), Matrix(1,1), Matrix(1,1)
        };
        // (0,0) => 1
        inputs[0](0, 0) = 0; inputs[0](0, 1) = 0;  targets[0](0, 0) = 1;
        // (0,1) => 1
        inputs[1](0, 0) = 0; inputs[1](0, 1) = 1;  targets[1](0, 0) = 1;
        // (1,0) => 1
        inputs[2](0, 0) = 1; inputs[2](0, 1) = 0;  targets[2](0, 0) = 1;
        // (1,1) => 0
        inputs[3](0, 0) = 1; inputs[3](0, 1) = 1;  targets[3](0, 0) = 0;

        std::vector<size_t> layerSizes = { 2, 4, 1 };
        std::vector<ActivationType> activs = {
            ActivationType::ReLU,
            ActivationType::Sigmoid
        };

        trainAndTestBinaryFunction("NAND", inputs, targets,
            layerSizes, activs,
            LossType::CrossEntropy,
            OptimizerType::Momentum,
            0.05, 0.9,
            5000,
            1000);
    }

    std::cout << "All tasks completed.\n";
    return 0;
}
