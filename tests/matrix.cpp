#include <iostream>
#include <vector>
#include "../libMachineLearning.hpp"

int main() {
    MachineLearning::Matrix mat;
    std::vector<double> vec;
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);

    mat.data = {1, 0, 0,
                0, 1, 0,
                0, 0, 1};
    mat.rows = 3;
    mat.cols = 3;

    vec = mat * vec;

    if (vec[0] == 1 && vec[1] == 2 && vec[2] == 3) {
        std::cout << "Pass identity matrix by [1,2,3]" << std::endl;
    } else {
        std::cout << "Fail identity matrix by [1,2,3]" << std::endl;
    }

    // Test from wikipedia article for matrix multiplication
    mat = MachineLearning::Matrix();
    MachineLearning::Matrix b;

    mat.data = {1, 0, 1,
                2, 1, 1,
                0, 1, 1,
                1, 1, 2};
    mat.rows = 4;
    mat.cols = 3;

    b.data = {1, 2, 1, 2, 3, 1, 4, 2, 2};
    b.rows = 3;
    b.cols = 3;

    MachineLearning::Matrix expected;
    expected.data = {5,4,3,8,9,5,6,5,3,11,9,6};
    expected.rows = 4;
    expected.cols = 3;

    mat * b;
    if (mat * b == expected) {
        std::cout << "Pass example from wikipedia" << std::endl;
    } else {
        std::cout << "Fail example from wikipedia" << std::endl;
    }


    mat = MachineLearning::Matrix();
    mat.data = {0,0,1,0,1,0,1,0,0};
    mat.rows = 3;
    mat.cols = 3;

    vec = mat * vec;
    if (vec[0] == 3 && vec[1] == 2 && vec[2] == 1) {
        std::cout << "Pass reverse vector matrix test" << std::endl;
    } else {
        std::cout << "Fail reverse vector matrix test" << std::endl;
    }

    // test B
    // -2R1 + R2 -> R2
    /* [[1 0 0
     * [-2 1 0]
     * [ 0 0 1]]*/

    mat = MachineLearning::Matrix();
    mat.data = {1,0,0,-2,1,0,0,0,1};
    mat.rows = 3;
    mat.cols = 3;

    expected = MachineLearning::Matrix();
    expected.data = {1,2,1,0,-1,-1,4,2,2};
    expected.rows = 3;
    expected.cols = 3;

    if (mat * b == expected) {
        std::cout << "pass row op" << std::endl;
    } else {
        std::cout << "fail row op" << std::endl;
    }

    mat = MachineLearning::Matrix();

    mat.data = {1,0,1,
                2,1,1,
                0,1,1,
                1,1,2};
    mat.rows = 4;
    mat.cols = 3;

    expected = MachineLearning::Matrix();
    expected.data = {1,2,0,1,
                    0,1,1,1,
                    1,1,1,2};
    expected.rows = 3;
    expected.cols = 4;

    if (mat.transpose() == expected) {
        std::cout << "pass transpose" << std::endl;
    } else {
         std::cout << "fail transpose" << std::endl;
    }

    return 0;
}
