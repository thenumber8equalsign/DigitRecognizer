#pragma once
#include <vector>
#include <stdexcept>
#include <memory>

inline std::vector<double> operator + (const std::vector<double>& lhs, const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size()) throw std::invalid_argument("sizes not equal");

    std::vector<double> res(lhs.size());
    for (size_t i = 0; i < res.size(); ++i) {
        res[i] = lhs[i] + rhs[i];
    }

    return res;
}

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
            std::vector<double> col(data.size());
            try {
                for (size_t i = 0; i < data.size(); ++i) {
                    col.at(i)=(data.at(i).at(j));
                }
            } catch (std::out_of_range) {
                throw std::runtime_error("malformed matrix");
            }
            return col;
        }

        std::vector<double> unsafeColAt(const size_t j) const {
            std::vector<double> col(data.size());
            for (size_t i = 0; i < data.size(); ++i) {
                col[i]=(data[i][j]);
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

        bool operator == (const matrix& other) const {
            try {
                if (data.size() == 0 || data.at(0).size() == 0 || other.data.size() == 0 || other.data.at(0).size() == 0) {
                    throw std::runtime_error("malformed matrix");
                }

                if (data.size() != other.data.size() || data.at(0).size() != other.data.at(0).size()) {
                    return false;
                }

                for (size_t i = 0; i < data.size(); ++i) {
                    for (size_t j = 0; j < data.at(0).size(); ++j) {
                        if (other.data.at(i).at(j) != data.at(i).at(j)) return false;
                    }
                }
                return true;
            } catch (std::out_of_range) {
                throw std::runtime_error("malformed matrix");
            }
        }
    };

    class Layer {
        private:
            std::weak_ptr<Layer> prev;

            std::vector<double> neurons;
            MachineLearning::matrix weights;
            std::vector<double> biases;
        public:
            Layer(std::shared_ptr<Layer> prev, size_t numNeurons) {
                this->prev = prev;
                neurons.resize(numNeurons);
                biases.resize(numNeurons);

                // The number of rows is the number of neurons here,
                // the number of columns is the number of neurons in the previous layer
                weights.data = std::vector<std::vector<double>>(numNeurons, std::vector<double>(prev->neurons.size()));
            }

            const MachineLearning::matrix getWeights() const {
                return weights;
            }

            const std::vector<double>& getNeurons() const {
                return neurons;
            }

            const std::vector<double> getBiases() const {
                return biases;
            }

            double& weightsAt(size_t row, size_t col) {
                return weights.data.at(row).at(col);
            }

            double& neuronsAt(size_t i) {
                // only allow setting a neuron if we are the starting layer
                std::shared_ptr<Layer> tmp = prev.lock();
                if (tmp) {
                    throw std::runtime_error("Can not set activation of a hidden layer");
                }

                return neurons.at(i);
            }

            double& biasesAt(size_t i) {
                return biases.at(i);
            }

            void compute() {
                // neurons = sigmoid(weights*(neuronsPrev) + bias)
                std::shared_ptr<Layer> prev = this->prev.lock();
                std::vector<double> z = weights * prev->neurons + biases;
                neurons = sigmoid(z);
            }
    };

    class Model {
        std::vector<std::shared_ptr<Layer>> layers;
    };
}
