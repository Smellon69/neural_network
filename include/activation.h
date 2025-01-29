#ifndef MY_NEURAL_NET_ACTIVATION_H_
#define MY_NEURAL_NET_ACTIVATION_H_

#include <functional>

/**
 * @file activation.h
 * @brief Activation function types and utilities.
 */

namespace nn {

	/**
	 * @enum ActivationType
	 * @brief Enumerates available activation function types.
	 */
	enum class ActivationType {
		Sigmoid,
		ReLU,
		Tanh
	};

	/**
	 * @struct ActivationFunction
	 * @brief Stores both forward and derivative functions for an activation.
	 */
	struct ActivationFunction {
		std::function<double(double)> forward;    ///< Forward pass
		std::function<double(double)> derivative; ///< Derivative wrt input
	};

	/**
	 * @brief Returns the activation function (forward & derivative) for a given type.
	 * @param type The activation type
	 * @return Corresponding activation functions
	 */
	ActivationFunction getActivation(ActivationType type);

}  // namespace nn

#endif  // MY_NEURAL_NET_ACTIVATION_H_
