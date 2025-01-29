#include "../include/activation.h"

#include <cmath>

namespace nn {

    static ActivationFunction sigmoidFunc = {
        /* forward */ [](double x) {
          return 1.0 / (1.0 + std::exp(-x));
        },
        /* derivative */ [](double x) {
          double s = 1.0 / (1.0 + std::exp(-x));
          return s * (1.0 - s);
        }
    };

    static ActivationFunction reluFunc = {
        /* forward */ [](double x) {
          return (x > 0.0) ? x : 0.0;
        },
        /* derivative */ [](double x) {
          return (x > 0.0) ? 1.0 : 0.0;
        }
    };

    static ActivationFunction tanhFunc = {
        /* forward */ [](double x) {
          return std::tanh(x);
        },
        /* derivative */ [](double x) {
          double t = std::tanh(x);
          return 1.0 - t * t;
        }
    };

    ActivationFunction getActivation(ActivationType type) {
        switch (type) {
        case ActivationType::Sigmoid:
            return sigmoidFunc;
        case ActivationType::ReLU:
            return reluFunc;
        case ActivationType::Tanh:
            return tanhFunc;
        }
        // Default
        return sigmoidFunc;
    }

}  // namespace nn
