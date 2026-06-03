#include <vector>

namespace MachineLearning {
    struct matrix;

    double sigmoid(const double x);
    // Shorthand vector version of the sigmoid function to avoid having to write out all the for loops
    std::vector<double> sigmoid(const std::vector<double>& v);

    double dot(const std::vector<double>& a, const std::vector<double>& b);
}
