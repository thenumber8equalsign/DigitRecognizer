#include "../libMachineLearning.hpp"
#include <iostream>
#include <vector>

using namespace std;

int main() {
    vector<double> a = {1, 2, 3, 4};
    vector<double> b = {5, 6, 7, 8};

    auto c = a + b;

    if (c[0] == 6 && c[1] == 8 && c[2] == 10 && c[3] == 12) {
        cout << "pass" << endl;
    } else {
        cout << "fail" << endl;
    }

    return 0;
}
