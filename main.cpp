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
 * @param inputs Vector of input matrices (each 1 x N for N-bit logic).
 * @param targets Vector of target matrices (each 1 x 1, 0 or 1).
 * @param layerSizes Network architecture (e.g., {2, 4, 1}).
 * @param activs Activations for each layer except input (e.g., {ReLU, Sigmoid}).
 * @param lossType Loss function (CrossEntropy, etc.).
 * @param optType Optimizer type (Momentum, SGD, etc.).
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
            std::cout << inputs[i](0, c);
            if (c + 1 < inputs[i].cols()) std::cout << ", ";
        }
        std::cout << ") -> " << out(0, 0)
            << " (target: " << targets[i](0, 0) << ")\n";
    }
    std::cout << std::endl;
}

int main() {
    // --------------------------------------------------------------------------
    // 1) Logic: AND
    // --------------------------------------------------------------------------
    {
        std::vector<Matrix> inputs(4, Matrix(1, 2));
        std::vector<Matrix> targets(4, Matrix(1, 1));

        // (0,0) => 0
        inputs[0](0, 0) = 0; inputs[0](0, 1) = 0;  targets[0](0, 0) = 0;
        // (0,1) => 0
        inputs[1](0, 0) = 0; inputs[1](0, 1) = 1;  targets[1](0, 0) = 0;
        // (1,0) => 0
        inputs[2](0, 0) = 1; inputs[2](0, 1) = 0;  targets[2](0, 0) = 0;
        // (1,1) => 1
        inputs[3](0, 0) = 1; inputs[3](0, 1) = 1;  targets[3](0, 0) = 1;

        std::vector<size_t> layerSizes = { 2, 4, 1 };   // 2->4->1
        std::vector<ActivationType> activs = {
            ActivationType::ReLU,
            ActivationType::Sigmoid
        };

        trainAndTestBinaryFunction("AND", inputs, targets,
            layerSizes, activs,
            LossType::CrossEntropy,
            OptimizerType::Momentum,
            0.05, 0.9,
            5000, 1000);
    }

    // --------------------------------------------------------------------------
    // 2) Logic: OR
    // --------------------------------------------------------------------------
    {
        std::vector<Matrix> inputs(4, Matrix(1, 2));
        std::vector<Matrix> targets(4, Matrix(1, 1));

        // (0,0) => 0
        inputs[0](0, 0) = 0; inputs[0](0, 1) = 0;  targets[0](0, 0) = 0;
        // (0,1) => 1
        inputs[1](0, 0) = 0; inputs[1](0, 1) = 1;  targets[1](0, 0) = 1;
        // (1,0) => 1
        inputs[2](0, 0) = 1; inputs[2](0, 1) = 0;  targets[2](0, 0) = 1;
        // (1,1) => 1
        inputs[3](0, 0) = 1; inputs[3](0, 1) = 1;  targets[3](0, 0) = 1;

        std::vector<size_t> layerSizes = { 2, 4, 1 };   // 2->4->1
        std::vector<ActivationType> activs = {
            ActivationType::ReLU,
            ActivationType::Sigmoid
        };

        trainAndTestBinaryFunction("OR", inputs, targets,
            layerSizes, activs,
            LossType::CrossEntropy,
            OptimizerType::Momentum,
            0.05, 0.9,
            5000, 1000);
    }

    // --------------------------------------------------------------------------
    // 3) Logic: XOR
    // --------------------------------------------------------------------------
    {
        std::vector<Matrix> inputs(4, Matrix(1, 2));
        std::vector<Matrix> targets(4, Matrix(1, 1));

        // (0,0) => 0
        inputs[0](0, 0) = 0; inputs[0](0, 1) = 0;  targets[0](0, 0) = 0;
        // (0,1) => 1
        inputs[1](0, 0) = 0; inputs[1](0, 1) = 1;  targets[1](0, 0) = 1;
        // (1,0) => 1
        inputs[2](0, 0) = 1; inputs[2](0, 1) = 0;  targets[2](0, 0) = 1;
        // (1,1) => 0
        inputs[3](0, 0) = 1; inputs[3](0, 1) = 1;  targets[3](0, 0) = 0;

        // 2->4->4->1
        std::vector<size_t> layerSizes = { 2, 4, 4, 1 };
        std::vector<ActivationType> activs = {
            ActivationType::ReLU,
            ActivationType::ReLU,
            ActivationType::Sigmoid
        };

        trainAndTestBinaryFunction("XOR", inputs, targets,
            layerSizes, activs,
            LossType::CrossEntropy,
            OptimizerType::Momentum,
            0.05, 0.9,
            10000,   // more epochs
            2000);
    }

    // --------------------------------------------------------------------------
    // 4) Logic: NAND
    // --------------------------------------------------------------------------
    {
        std::vector<Matrix> inputs(4, Matrix(1, 2));
        std::vector<Matrix> targets(4, Matrix(1, 1));

        // (0,0) => 1
        inputs[0](0, 0) = 0; inputs[0](0, 1) = 0;  targets[0](0, 0) = 1;
        // (0,1) => 1
        inputs[1](0, 0) = 0; inputs[1](0, 1) = 1;  targets[1](0, 0) = 1;
        // (1,0) => 1
        inputs[2](0, 0) = 1; inputs[2](0, 1) = 0;  targets[2](0, 0) = 1;
        // (1,1) => 0
        inputs[3](0, 0) = 1; inputs[3](0, 1) = 1;  targets[3](0, 0) = 0;

        std::vector<size_t> layerSizes = { 2, 4, 1 };   // 2->4->1
        std::vector<ActivationType> activs = {
            ActivationType::ReLU,
            ActivationType::Sigmoid
        };

        trainAndTestBinaryFunction("NAND", inputs, targets,
            layerSizes, activs,
            LossType::CrossEntropy,
            OptimizerType::Momentum,
            0.05, 0.9,
            5000, 1000);
    }

    // --------------------------------------------------------------------------
    // 5) Hard Task: 4-bit Parity (4->16->16->1)
    //     Output is 1 if number of 1-bits is odd, else 0
    // --------------------------------------------------------------------------
    {
        // We have 16 possible inputs for 4 bits (0000..1111)
        // We'll store them, plus the target for parity:
        std::vector<Matrix> inputs;
        std::vector<Matrix> targets;
        inputs.reserve(16);
        targets.reserve(16);

        for (int pattern = 0; pattern < 16; ++pattern) {
            Matrix in(1, 4);
            // Fill the 4 columns with the bits of 'pattern'
            int countOnes = 0;
            for (int bitIndex = 0; bitIndex < 4; ++bitIndex) {
                int bitVal = (pattern >> bitIndex) & 1;
                in(0, bitIndex) = bitVal;
                if (bitVal == 1) ++countOnes;
            }
            inputs.push_back(in);

            Matrix t(1, 1);
            // 1 if odd number of bits set, 0 otherwise
            t(0, 0) = (countOnes % 2 == 1) ? 1.0 : 0.0;
            targets.push_back(t);
        }

        // For 4-bit parity, let's do a bigger/deeper net
        // e.g. 4 -> 16 -> 16 -> 1
        std::vector<size_t> layerSizes = { 4, 16, 16, 1 };
        std::vector<ActivationType> activs = {
            ActivationType::Tanh,
            ActivationType::Tanh,
            ActivationType::Sigmoid
        };

        // We'll train for a large number of epochs since it's tricky
        // and we want it to converge (this might take a while).
        // on my PC this takes like 2 minutes :)
        int epochs = 200000;
        int logInterval = 20000;

        trainAndTestBinaryFunction("4-bit Parity", inputs, targets,
            layerSizes, activs,
            LossType::CrossEntropy,
            OptimizerType::Momentum,
            0.05, 0.9,
            epochs, logInterval);
    }

    std::cout << "All tasks completed.\n";
    return 0;
}
