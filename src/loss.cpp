#include "../include/loss.h"

#include <cmath>
#include <cassert>
#include <algorithm>

namespace nn {

    static LossFunction mseLoss = {
        // forward
        [](const Matrix& pred, const Matrix& truth) {
            // mean(0.5*(pred - truth)^2)
            assert(pred.rows() == truth.rows() && pred.cols() == truth.cols());
            double sum = 0.0;
            size_t count = pred.rows() * pred.cols();
            for (size_t i = 0; i < count; ++i) {
              double diff = pred.data()[i] - truth.data()[i];
              sum += 0.5 * diff * diff;
            }
            return sum / static_cast<double>(pred.rows());
          },
        // derivative wrt pred
        [](const Matrix& pred, const Matrix& truth) {
          Matrix grad(pred.rows(), pred.cols());
          size_t count = pred.rows() * pred.cols();
          for (size_t i = 0; i < count; ++i) {
            grad.data()[i] = (pred.data()[i] - truth.data()[i])
                             / static_cast<double>(pred.rows());
          }
          return grad;
        }
    };

    static LossFunction crossEntropyLoss = {
        // forward
        [](const Matrix& pred, const Matrix& truth) {
            // sum( -t*log(p) - (1-t)*log(1-p) ) / batch
            double sum = 0.0;
            size_t count = pred.rows() * pred.cols();
            for (size_t i = 0; i < count; ++i) {
              double p = pred.data()[i];
              double t = truth.data()[i];
              // clamp
              if (p < 1e-12) p = 1e-12;
              if (p > 1.0 - 1e-12) p = 1.0 - 1e-12;
              sum += -(t * std::log(p) + (1.0 - t) * std::log(1.0 - p));
            }
            return sum / static_cast<double>(pred.rows());
          },
        // derivative
        [](const Matrix& pred, const Matrix& truth) {
          Matrix grad(pred.rows(), pred.cols());
          size_t count = pred.rows() * pred.cols();
          for (size_t i = 0; i < count; ++i) {
            double p = pred.data()[i];
            double t = truth.data()[i];
            if (p < 1e-12) p = 1e-12;
            if (p > 1.0 - 1e-12) p = 1.0 - 1e-12;
            grad.data()[i] = (p - t) / (p * (1.0 - p))
                             / static_cast<double>(pred.rows());
          }
          return grad;
        }
    };

    LossFunction getLoss(LossType type) {
        switch (type) {
        case LossType::MSE:
            return mseLoss;
        case LossType::CrossEntropy:
            return crossEntropyLoss;
        }
        // Default
        return mseLoss;
    }

}  // namespace nn
