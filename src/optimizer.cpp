#include "../include/optimizer.h"

namespace nn {

    SGDOptimizer::SGDOptimizer(double lr) : m_lr(lr) {}

    void SGDOptimizer::update(Matrix& w, const Matrix& grad) {
        size_t count = w.data().size();
        for (size_t i = 0; i < count; ++i) {
            w.data()[i] -= m_lr * grad.data()[i];
        }
    }

    MomentumOptimizer::MomentumOptimizer(double lr, double momentum)
        : m_lr(lr), m_momentum(momentum) {}

    void MomentumOptimizer::update(Matrix& w, const Matrix& grad) {
        if (m_velocity.rows() == 0) {
            // Initialize velocity with same shape as w
            m_velocity = Matrix(w.rows(), w.cols());
        }

        size_t count = w.data().size();
        for (size_t i = 0; i < count; ++i) {
            double v = m_velocity.data()[i];
            v = m_momentum * v - m_lr * grad.data()[i];
            m_velocity.data()[i] = v;
            w.data()[i] += v;
        }
    }

    std::unique_ptr<Optimizer> createOptimizer(OptimizerType type, double lr, double momentum) {
        if (type == OptimizerType::SGD) {
            return std::make_unique<SGDOptimizer>(lr);
        }
        else {
            return std::make_unique<MomentumOptimizer>(lr, momentum);
        }
    }

}  // namespace nn
