#include <vector>

namespace MachineLearning {
    struct matrix;

    double sigmoid(double x);
    std::vector<double> sigmoid(std::vector<double> v);

    double dot(std::vector<double> a, std::vector<double> b);
}
