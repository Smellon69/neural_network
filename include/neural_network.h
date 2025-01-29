#ifndef MY_NEURAL_NET_NEURAL_NETWORK_H_
#define MY_NEURAL_NET_NEURAL_NETWORK_H_

#include <vector>
#include <memory>
#include "matrix.h"
#include "activation.h"
#include "loss.h"
#include "optimizer.h"

/**
 * @file neural_network.h
 * @brief A simple fully-connected neural network class.
 */

namespace nn {

    /**
     * @class NeuralNetwork
     * @brief Implements a multi-layer feed-forward neural network with
     *        backpropagation training (single-sample version).
     */
    class NeuralNetwork {
    public:
        /**
         * @brief Constructs the network based on layer sizes and other hyperparameters.
         * @param layerSizes e.g. {2, 4, 4, 1}
         * @param activations e.g. {ReLU, ReLU, Sigmoid}
         * @param lossType e.g. CrossEntropy
         * @param optType e.g. Momentum
         * @param learningRate
         * @param momentum
         */
        NeuralNetwork(const std::vector<size_t>& layerSizes,
            const std::vector<ActivationType>& activations,
            LossType lossType,
            OptimizerType optType,
            double learningRate = 0.1,
            double momentum = 0.9);

        /**
         * @brief Forward pass for a single sample.
         * @param input A (1 x input_dim) matrix
         * @return The output matrix (1 x output_dim)
         */
        Matrix forward(const Matrix& input);

        /**
         * @brief Trains on a single sample via backprop.
         * @param input A (1 x input_dim) matrix
         * @param target A (1 x output_dim) matrix
         * @return The scalar loss value for this sample
         */
        double trainSample(const Matrix& input, const Matrix& target);

    private:
        std::vector<Matrix> m_weights;   ///< Weight matrices
        std::vector<Matrix> m_biases;    ///< Bias vectors
        std::vector<ActivationFunction> m_activations;
        std::vector<Matrix> m_layerNetInputs;   ///< Pre-activation net inputs
        std::vector<Matrix> m_layerOutputs;     ///< Post-activation outputs

        LossFunction m_lossFunc;

        // Each layer has its own optimizer for W and B
        std::vector<std::unique_ptr<Optimizer>> m_optimizersW;
        std::vector<std::unique_ptr<Optimizer>> m_optimizersB;
    };

}  // namespace nn

#endif  // MY_NEURAL_NET_NEURAL_NETWORK_H_
