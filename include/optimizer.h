#ifndef MY_NEURAL_NET_OPTIMIZER_H_
#define MY_NEURAL_NET_OPTIMIZER_H_

#include <memory>
#include "matrix.h"

/**
 * @file optimizer.h
 * @brief Optimizer declarations (SGD, Momentum, etc.).
 */

namespace nn {

	/**
	 * @enum OptimizerType
	 * @brief Enumerates optimizer types.
	 */
	enum class OptimizerType {
		SGD,
		Momentum
	};

	/**
	 * @class Optimizer
	 * @brief Abstract base class for parameter updaters.
	 */
	class Optimizer {
	public:
		virtual ~Optimizer() = default;

		/**
		 * @brief Updates parameter matrix `w` given gradient `grad`.
		 * @param w Weight matrix to update
		 * @param grad Gradient wrt weights
		 */
		virtual void update(Matrix& w, const Matrix& grad) = 0;
	};

	/**
	 * @class SGDOptimizer
	 * @brief Implements vanilla SGD update rule.
	 */
	class SGDOptimizer : public Optimizer {
	public:
		/**
		 * @brief Constructs with a given learning rate.
		 */
		explicit SGDOptimizer(double lr);

		/**
		 * @brief Update rule: w = w - lr * grad
		 */
		void update(Matrix& w, const Matrix& grad) override;

	private:
		double m_lr;
	};

	/**
	 * @class MomentumOptimizer
	 * @brief Implements momentum-based update rule.
	 */
	class MomentumOptimizer : public Optimizer {
	public:
		/**
		 * @brief Constructs with given learning rate and momentum factor.
		 */
		MomentumOptimizer(double lr, double momentum);

		/**
		 * @brief Update rule:
		 *        v = momentum * v - lr * grad
		 *        w = w + v
		 */
		void update(Matrix& w, const Matrix& grad) override;

	private:
		double m_lr;
		double m_momentum;
		Matrix m_velocity;
	};

	/**
	 * @brief Factory function for creating an optimizer.
	 * @param type Which optimizer to create (SGD or Momentum)
	 * @param lr Learning rate
	 * @param momentum Momentum factor (only used for Momentum optimizer)
	 * @return Unique pointer to the optimizer
	 */
	std::unique_ptr<Optimizer> createOptimizer(OptimizerType type, double lr, double momentum = 0.9);

}  // namespace nn

#endif  // MY_NEURAL_NET_OPTIMIZER_H_
