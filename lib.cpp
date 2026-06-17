#include "libMachineLearning.hpp"
#include <cmath>

namespace MachineLearning {
    double sigmoid(const double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double sigmoidDerivative(const double x) {
        const double s = std::exp(-x);
        return s / ((1.0 + s) * (1.0 + s));
    }

    std::vector<Neuron> sigmoid(const std::vector<double>& v) {
        std::vector<Neuron> result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i].activation = sigmoid(v[i]);
            result[i].weightedInput = v[i];
        }
        return result;
    }

    std::vector<double> sigmoidDerivative(const std::vector<Neuron>& v) {
        std::vector<double> result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = sigmoidDerivative(v[i].weightedInput);
        }
        return result;
    }
}
