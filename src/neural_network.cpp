#include "../include/neural_network.h"

#include <cassert>
#include <iostream>

namespace nn {

    NeuralNetwork::NeuralNetwork(const std::vector<size_t>& layerSizes,
        const std::vector<ActivationType>& activations,
        LossType lossType,
        OptimizerType optType,
        double learningRate,
        double momentum) {
        assert(layerSizes.size() >= 2 && "Must have at least input & output layer");
        assert(layerSizes.size() - 1 == activations.size() &&
            "Need one activation for each layer except input");

        size_t numLayers = layerSizes.size() - 1;
        m_activations.reserve(numLayers);
        m_weights.reserve(numLayers);
        m_biases.reserve(numLayers);
        m_optimizersW.reserve(numLayers);
        m_optimizersB.reserve(numLayers);

        // Create each layer
        for (size_t i = 0; i < numLayers; ++i) {
            size_t inDim = layerSizes[i];
            size_t outDim = layerSizes[i + 1];

            m_weights.emplace_back(inDim, outDim, true);
            m_biases.emplace_back(1, outDim, true);

            m_activations.push_back(getActivation(activations[i]));

            m_optimizersW.push_back(createOptimizer(optType, learningRate, momentum));
            m_optimizersB.push_back(createOptimizer(optType, learningRate, momentum));
        }

        // Loss
        m_lossFunc = getLoss(lossType);
    }

    Matrix NeuralNetwork::forward(const Matrix& input) {
        // Clear stored nets/outputs
        m_layerOutputs.clear();
        m_layerNetInputs.clear();

        Matrix current = input;  // shape: (1 x inDim)

        // Forward through each layer
        for (size_t i = 0; i < m_weights.size(); ++i) {
            Matrix net = Matrix::add(Matrix::multiply(current, m_weights[i]), m_biases[i]);
            m_layerNetInputs.push_back(net);

            Matrix out = net;
            out.applyFunction(m_activations[i].forward);

            m_layerOutputs.push_back(out);
            current = out;
        }
        return current;
    }

    double NeuralNetwork::trainSample(const Matrix& input, const Matrix& target) {
        Matrix pred = forward(input);

        // Compute loss
        double lossVal = m_lossFunc.forward(pred, target);

        // Gradient wrt final output
        Matrix gradOut = m_lossFunc.derivative(pred, target);

        // Backprop
        for (int layerIndex = static_cast<int>(m_weights.size()) - 1; layerIndex >= 0; --layerIndex) {
            // dAct = derivative wrt net input
            Matrix dAct = m_layerNetInputs[layerIndex];
            ActivationFunction af = m_activations[layerIndex];
            for (size_t i = 0; i < dAct.data().size(); ++i) {
                double x = dAct.data()[i];
                dAct.data()[i] = af.derivative(x);
            }

            // gradOut *= dAct
            for (size_t i = 0; i < gradOut.data().size(); ++i) {
                gradOut.data()[i] *= dAct.data()[i];
            }

            // layerInput is input to current layer
            Matrix layerInput;
            if (layerIndex == 0) {
                layerInput = input;
            }
            else {
                layerInput = m_layerOutputs[layerIndex - 1];
            }

            // dW = layerInput^T * gradOut
            Matrix layerInputT = Matrix::transpose(layerInput);
            Matrix dW = Matrix::multiply(layerInputT, gradOut);

            // dB = gradOut
            Matrix dB = gradOut;

            // Update
            m_optimizersW[layerIndex]->update(m_weights[layerIndex], dW);
            m_optimizersB[layerIndex]->update(m_biases[layerIndex], dB);

            // Compute gradOut for previous layer
            if (layerIndex > 0) {
                Matrix wT = Matrix::transpose(m_weights[layerIndex]);
                Matrix prevGrad = Matrix::multiply(gradOut, wT);
                gradOut = prevGrad;
            }
        }

        return lossVal;
    }

}  // namespace nn
