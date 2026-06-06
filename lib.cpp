#include "libMachineLearning.hpp"
#include <cmath>

namespace MachineLearning {
    double sigmoid(const double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double sigmoidDerivative(const double x) {
        return std::exp(-x) / ((1.0 + std::exp(-x)) * (1.0 + std::exp(-x)));
    }

    std::vector<double> sigmoid(const std::vector<double>& v) {
        std::vector<double> result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = sigmoid(v[i]);
        }
        return result;
    }

}
