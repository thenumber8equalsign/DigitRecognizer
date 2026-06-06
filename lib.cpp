#include "libMachineLearning.hpp"
#include <cmath>

namespace MachineLearning {
    double sigmoid(const double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double sigmoidDerivative(const double x) {
        return std::exp(-x) / ((1.0 + std::exp(-x)) * (1.0 + std::exp(-x)));
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
