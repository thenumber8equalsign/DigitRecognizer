#include "libMachineLearning.hpp"
#include <cmath>
#include <stdexcept>

namespace MachineLearning {
    double dot(const std::vector<double>& a, const std::vector<double>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("not same dimension");
        }

        double d = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            d += a[i] * b[i];
        }
        return d;
    }

    double sigmoid(const double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    std::vector<double> sigmoid(const std::vector<double>& v) {
        std::vector<double> result(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            result[i] = sigmoid(v[i]);
        }
        return result;
    }

}
