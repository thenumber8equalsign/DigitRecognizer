#pragma once
#include <vector>
#include <stdexcept>

namespace MachineLearning {
    double sigmoid(const double x);
    // Shorthand vector version of the sigmoid function to avoid having to write out all the for loops
    std::vector<double> sigmoid(const std::vector<double>& v);

    double dot(const std::vector<double>& a, const std::vector<double>& b);

    struct matrix {
        std::vector<std::vector<double>> data;

        matrix(const size_t r, const size_t c) {
            data = std::vector<std::vector<double>>(r, std::vector<double>(c, 0));
        }

        matrix() {}

        std::vector<double>& operator [] (const size_t i) {
            return data[i];
        }

        std::vector<double>& at(const size_t i) {
            return data.at(i);
        }

        std::vector<double> colAt(const size_t j) const {
            std::vector<double> col;
            try {
                for (size_t i = 0; i < data.size(); ++i) {
                    col.push_back(data.at(i).at(j));
                }
            } catch (std::out_of_range) {
                throw std::runtime_error("malformed matrix");
            }
            return col;
        }

        std::vector<double> unsafeColAt(const size_t j) const {
            std::vector<double> col;
            for (size_t i = 0; i < data.size(); ++i) {
                col.push_back(data[i][j]);
            }
            return col;
        }


        matrix operator * (const double scalar) const {
            matrix result;

            try {
                result = matrix(data.size(), data.at(0).size());
            } catch (std::out_of_range) {
                throw std::runtime_error("malformed matrix");
            }

            try {
                for (size_t i = 0; i < result.data.size(); ++i) {
                    for (size_t j = 0; j < result.data[0].size(); ++j) {
                        result[i][j] = data.at(i).at(j) * scalar;
                    }
                }
            } catch (std::out_of_range) {
                // not all the rows are the same length
                throw std::runtime_error("malformed matrix");
            }
            return result;
        }


        matrix& operator *= (const double scalar) {
            for (size_t i = 0; i < data.size(); ++i) {
                for (size_t j = 0; j < data[i].size(); ++j) {
                    data.at(i).at(j) *= scalar;
                }
            }
            return *this;
        }

        matrix operator * (const matrix& other) const {
            try {
                // Check restrictions (# col here has to be # row there)
                if (data.at(0).size() != other.data.size()) {
                    throw std::domain_error("undefined"); // Domain error was the closest error type I could find, so I use it cuz i dont wanna make my own
                }
                matrix mat(data.size(), other.data.at(0).size());
                for (size_t i = 0; i < mat.data.size(); ++i) {
                    for (size_t j = 0; j < mat.data[0].size(); ++j) {
                        // dot the LHS matrix's row with the RHS matrix's col
                        mat[i][j] = dot(data.at(i), other.colAt(j));
                    }
                }

                return mat;
            } catch (std::out_of_range) {
                throw std::runtime_error("malformed matrix");
            }
        }

        std::vector<double> operator * (const std::vector<double>& other) const {
            try {
                if (data.at(0).size() != other.size()) {
                    throw std::domain_error("undefined");
                }

                std::vector<double> result(data.size());
                for (size_t i = 0; i < data.size(); ++i) {
                    result[i] = dot(data[i], other);
                }

                return result;
            } catch (std::out_of_range) {
                throw std::runtime_error("malformed matrix");
            }
        }
    };
    typedef struct matrix matrix;
}
