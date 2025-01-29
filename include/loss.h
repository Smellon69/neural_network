#ifndef MY_NEURAL_NET_LOSS_H_
#define MY_NEURAL_NET_LOSS_H_

#include <functional>
#include "matrix.h"

/**
 * @file loss.h
 * @brief Defines loss function types and utilities.
 */

namespace nn {

	/**
	 * @enum LossType
	 * @brief Enumerates available loss functions.
	 */
	enum class LossType {
		MSE,
		CrossEntropy
	};

	/**
	 * @struct LossFunction
	 * @brief Holds the forward loss function and its derivative wrt network output.
	 */
	struct LossFunction {
		/**
		 * @brief Computes scalar loss given prediction and target.
		 */
		std::function<double(const Matrix&, const Matrix&)> forward;

		/**
		 * @brief Computes derivative of loss wrt prediction (dL/dY).
		 */
		std::function<Matrix(const Matrix&, const Matrix&)> derivative;
	};

	/**
	 * @brief Factory: returns the loss function for given loss type.
	 * @param type The loss type
	 * @return LossFunction with forward & derivative
	 */
	LossFunction getLoss(LossType type);

}  // namespace nn

#endif  // MY_NEURAL_NET_LOSS_H_
