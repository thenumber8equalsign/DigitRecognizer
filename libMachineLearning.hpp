#pragma once
#include <vector>
#include <stdexcept>
#include <memory>
#include <array>
#include <random>

inline std::vector<double> operator + (const std::vector<double>& lhs, const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size()) throw std::invalid_argument("sizes not equal");

    std::vector<double> res(lhs.size());
    for (size_t i = 0; i < res.size(); ++i) {
        res[i] = lhs[i] + rhs[i];
    }

    return res;
}

// Hadamard product
inline std::vector<double> operator * (const std::vector<double>& lhs, const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size()) throw std::invalid_argument("sizes not equal");

    std::vector<double> res(lhs.size());
    for (size_t i = 0; i < res.size(); ++i) {
        res[i] = lhs[i] * rhs[i];
    }

    return res;
}

namespace MachineLearning {
    struct Neuron {
        double activation = 0;
        double weightedInput = 0;
    };
    typedef struct Neuron Neuron;

    double sigmoid(const double x);
    double sigmoidDerivative(const double x);
    std::vector<double> sigmoidDerivative(const std::vector<Neuron>&);
    // Shorthand vector version of the sigmoid function to avoid having to write out all the for loops
    std::vector<Neuron> sigmoid(const std::vector<double>& v);

    inline __attribute__((always_inline)) double dot(const std::vector<double>& a, const std::vector<Neuron>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("not same dimension");
        }

        double d = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            d += a[i] * b[i].activation;
        }
        return d;
    }

    inline __attribute__((always_inline)) double dot(const std::vector<double>& a, const std::vector<double>& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("not same dimension");
        }

        double d = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            d += a[i] * b[i];
        }
        return d;
    }

    struct Matrix {
        size_t rows, cols;
        std::vector<double> data;

        Matrix(const size_t r, const size_t c) : rows(r), cols(c) {
            data = std::vector<double>(r*c);
        }

        Matrix() : rows(0), cols(0) {}

        inline __attribute__((always_inline)) double& at(const size_t i, const size_t j) {
            return data[i * cols + j];
        }

        inline __attribute__((always_inline)) Matrix operator * (const double scalar) const {
            Matrix result;

            // try {
                result = Matrix(rows, cols);
            // } catch (std::out_of_range) {
            //     throw std::runtime_error("malformed matrix");
            // }

            // try {
                for (size_t i = 0; i < rows; ++i) {
                    for (size_t j = 0; j < cols; ++j) {
                        result.data[i * cols + j] = data[i * cols + j] * scalar;
                    }
                }
            // } catch (std::out_of_range) {
            //     // not all the rows are the same length
            //     throw std::runtime_error("malformed matrix");
            // }
            return result;
        }


        inline __attribute__((always_inline)) Matrix& operator *= (const double scalar) {
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    data[i * cols + j] *= scalar;
                }
            }
            return *this;
        }

        inline __attribute__((always_inline)) Matrix operator * (const Matrix& other) const {
            // try {
                // Check restrictions (# col here has to be # row there)
                // if (data.at(0).size() != other.data.size()) {
                //     throw std::domain_error("undefined"); // Domain error was the closest error type I could find, so I use it cuz i dont wanna make my own
                // }
                Matrix mat(rows, other.cols);
                for (size_t i = 0; i < mat.rows; ++i) {
                    for (size_t j = 0; j < mat.cols; ++j) {
                        // dot the LHS matrix's row with the RHS matrix's col
                        // mat[i][j] = dot(data.at(i), other.colAt(j));
                        mat.data[i * mat.cols + j] = 0;

                        // Go through everything in this row
                        for (size_t k = 0; k < cols; ++k) {
                            mat.data[i*mat.cols+j] += data[i*cols+k] * other.data[k*other.cols+j];
                        }

                    }
                }

                return mat;
            // } catch (std::out_of_range) {
            //     throw std::runtime_error("malformed matrix");
            // }
        }

        inline __attribute__((always_inline)) std::vector<double> operator * (const std::vector<Neuron>& other) const {
            // try {
                // if (data.at(0).size() != other.size()) {
                //     throw std::domain_error("undefined");
                // }

                std::vector<double> result(rows);
                for (size_t i = 0; i < rows; ++i) {
                    result[i] = 0;
                    for (size_t j = 0; j < cols; ++j) {
                        result[i] += data[i*cols + j] * other[j].activation;
                    }
                }

                return result;
            // } catch (std::out_of_range) {
            //     throw std::runtime_error("malformed matrix");
            // }
        }

        inline __attribute__((always_inline)) std::vector<double> operator * (const std::vector<double>& other) const {
            // try {
                // if (data.at(0).size() != other.size()) {
                //     throw std::domain_error("undefined");
                // }

                std::vector<double> result(rows);
                for (size_t i = 0; i < rows; ++i) {
                    result[i] = 0;
                    for (size_t j = 0; j < cols; ++j) {
                        result[i] += data[i*cols + j] * other[j];
                    }
                }

                return result;
            // } catch (std::out_of_range) {
            //     throw std::runtime_error("malformed matrix");
            // }
        }

        inline __attribute__((always_inline)) bool operator == (const Matrix& other) const {
            // try {
                // if (data.size() == 0 || data.at(0).size() == 0 || other.data.size() == 0 || other.data.at(0).size() == 0) {
                //     throw std::runtime_error("malformed matrix");
                // }

                if (data.size() != other.data.size() || rows != other.rows || cols != other.cols) {
                    return false;
                }

                for (size_t i = 0; i < rows; ++i) {
                    for (size_t j = 0; j < cols; ++j) {
                        if (other.data[i * cols + j] != data[i * cols + j]) return false;
                    }
                }
                return true;
            // } catch (std::out_of_range) {
            //     throw std::runtime_error("malformed matrix");
            // }
        }

        inline __attribute__((always_inline)) const size_t getRows() const { return rows; }
        inline __attribute__((always_inline)) const size_t getCols() const { return cols; }

        inline __attribute__((always_inline)) Matrix transpose() const {
            // try {
                Matrix res(cols, rows);
                for (size_t i = 0; i < res.rows; ++i) {
                    for (size_t j = 0; j < res.cols; ++j) {
                        res.data[i*res.cols+j] = data[j*cols+i];
                    }
                }
                return res;
            // } catch (std::out_of_range) {
            //     throw std::runtime_error("malformed matrix");
            // }
        }
    };

    // Like EVERY single programmer who hath come before I. I can't name stuff
    struct ParameterStruct {
        Matrix weights;
        std::vector<double> biases;
    };
    typedef struct ParameterStruct ParameterStruct;

    class Layer {
        private:
            std::vector<Neuron> neurons;
            MachineLearning::Matrix weights;
            std::vector<double> biases;
            std::shared_ptr<Layer> prevShared;
        public:
            std::weak_ptr<Layer> prev;

            Layer(std::shared_ptr<Layer> prev, size_t numNeurons) {
                this->prev = prev;
                neurons.resize(numNeurons);
                biases.resize(numNeurons);

                // The number of rows is the number of neurons here,
                // the number of columns is the number of neurons in the previous layer
                // weights.data = std::vector<std::vector<double>>(numNeurons, std::vector<double>(prev->neurons.size()));
                weights.data.resize(numNeurons*prev->neurons.size());
                weights.rows = numNeurons;
                weights.cols = prev->neurons.size();
            }

            Layer(size_t numNeurons) {
                neurons.resize(numNeurons);
                biases.resize(numNeurons);
            }

            Layer() {}

            void updateWeightDimensions() {
                auto prev = this->prev.lock();
                // weights.data = std::vector<std::vector<double>>(neurons.size(), std::vector<double>(prev->neurons.size()));
                weights.data.resize(neurons.size()*prev->neurons.size());
                weights.rows = neurons.size();
                weights.cols = prev->neurons.size();
            }

            const MachineLearning::Matrix getWeights() const {
                return weights;
            }

            const std::vector<Neuron>& getNeurons() const {
                return neurons;
            }

            const std::vector<double> getBiases() const {
                return biases;
            }

            double& weightAt(size_t row, size_t col) {
                return weights.at(row, col);
            }

            Neuron& neuronAt(size_t i) {
                // only allow setting a neuron if we are the starting layer
                // std::shared_ptr<Layer> tmp = prev.lock();
                // if (tmp) {
                //     throw std::runtime_error("Can not set activation of a hidden or output layer");
                // }

                return neurons.at(i);
            }

            double& biasAt(size_t i) {
                return biases.at(i);
            }

            // Use these that way we don't call prev.lock() every single time in training
            void prepareCompute() {
                prevShared = this->prev.lock();
            }

            void endCompute() {
                prevShared = nullptr;
            }

            void compute() {
                // neurons = sigmoid(weights*(neuronsPrev) + bias)
                // std::shared_ptr<Layer> prev = this->prev.lock();
                std::vector<double> z = weights * prevShared->neurons + biases;
                neurons = sigmoid(z);
            }

            void compute(const ParameterStruct& ps) {
                // std::shared_ptr<Layer> prev = this->prev.lock();
                std::vector<double> z = ps.weights * prevShared->neurons + ps.biases;
                neurons = sigmoid(z);
            }
    };

    typedef std::array<std::array<double, 28>, 28> Image;

    inline std::random_device dev;
    inline std::mt19937 rng(dev());

    class Model {
        public:
            std::vector<std::shared_ptr<Layer>> layers;

            // The char is the label
            std::vector<std::pair<Image, char>> trainingData;

            Model() {}
            Model(std::vector<std::shared_ptr<Layer>> layers) {
                this->layers = layers;
            }

            // connect all of the prev pointers in the layer
            // it literally won't do anything if layers is empty
            void linkLayers() {
                for (size_t i = 1; i < layers.size(); ++i) {
                    layers[i]->prev = layers[i-1];
                    layers[i]->updateWeightDimensions();
                }
            }

            double costDerivative(double activation, double expected) {
                return 2 * (activation - expected);
            }

            constexpr double weightedInputWithBias(double b) {
                return 1;
            }


            void prepare() {
                for (size_t i = 0; i < layers.size(); ++i) {
                    layers[i]->prepareCompute();
                }
            }

            void end() {
                for (size_t i = 0; i < layers.size(); ++i) {
                    layers[i]->endCompute();
                }
            }

            std::vector<ParameterStruct> backPropagate(std::vector<ParameterStruct>& derivatives, std::vector<std::vector<double>>& errors, std::vector<double>& expected) {
                // http://neuralnetworksanddeeplearning.com/chap2.html

                for (size_t i = 0; i < trainingData.size(); ++i) {
                    // Load the data
                    const auto& img = trainingData.at(i).first;
                    const auto& lab = trainingData.at(i).second;

                    for (size_t j = 0; j < expected.size(); ++j) {
                        if (j == lab) {
                            expected.at(j) = 1.0;
                        } else {
                            expected.at(j) = 0.0;
                        }
                    }

                    for (size_t j = 0; j < layers.at(0)->getNeurons().size(); ++j) {
                        layers.at(0)->neuronAt(j).activation = img.at(j / 28).at(j % 28);
                    }

                    for (size_t j = 1; j < layers.size(); ++j) {
                        layers.at(j)->compute();
                    }

                    for (size_t j = 0; j < errors.at(errors.size()-1).size(); ++j) {
                        auto& er = errors.at(errors.size()-1);
                        er.at(j) = (layers[layers.size()-1]->getNeurons().at(j).activation - expected.at(j));
                        er.at(j) *= sigmoidDerivative(layers[layers.size()-1]->getNeurons().at(j).weightedInput);
                    }

                    for (size_t j = errors.size()-2; j > 0; --j) {
                        errors.at(j) = ((layers.at(j+2)->getWeights().transpose() * errors.at(j+1)) * sigmoidDerivative(layers.at(j+1)->getNeurons()));
                    }
                    // Since if we change the above loop to be j>=0, we get an error because size_t is unsigned, therefore we must add this hard-coded version for j=0
                    errors.at(0) = ((layers.at(2)->getWeights().transpose() * errors.at(1)) * sigmoidDerivative(layers.at(1)->getNeurons()));


                    for (size_t j = 0; j < derivatives.size(); ++j) {
                        // delC/delB = the error
                        for (size_t k = 0; k < derivatives.at(j).biases.size(); ++k) {
                            derivatives.at(j).biases.at(k) += errors.at(j).at(k);
                        }

                        // delC/delwljk = a(l-1)k*error(l)j
                        // current layer: j+1
                        for (size_t k = 0; k < derivatives.at(j).weights.rows; ++k) {
                            for (size_t l = 0; l < derivatives.at(j).weights.cols; ++l) {
                                derivatives.at(j).weights.at(k, l) += layers.at(j)->getNeurons().at(l).activation * errors.at(j).at(k);
                            }
                        }
                    }
                }

                // Now that we have the sum of all the derivatives, divide it by the number of training samples
                for (size_t i = 0; i < derivatives.size(); ++i) {
                    for (size_t j = 0; j < derivatives.at(i).biases.size(); ++j) {
                        derivatives.at(i).biases.at(j) /= trainingData.size();
                    }

                    for (size_t j = 0; j < derivatives.at(i).weights.getRows(); ++j) {
                        for (size_t k = 0; k < derivatives.at(i).weights.getCols(); ++k) {
                            derivatives.at(i).weights.at(j, k) /= trainingData.size();
                        }
                    }
                }

                return derivatives;
            }

            double computeCost(const std::vector<ParameterStruct>& params) const {
                // This code is identical to computeCost(), except it uses the overloaded compute() in Layer, that takes in a ParameterStruct
                double cost = 0.0;
                for (size_t i = 0; i < trainingData.size(); ++i) {
                    auto inputLayer = layers.at(0);
                    std::array<double, 10> expectedOutput;
                    for (size_t j = 0; j < 10; ++j) {
                        expectedOutput[j] = (j == trainingData.at(i).second) ? 1.0 : 0.0;
                    }

                    for (size_t j = 0; j < trainingData.at(0).first.size() * trainingData.at(0).first.at(0).size(); ++j) {
                        // j / numCols is the row
                        // j % numCols is the col
                        // puttin' this here cuz i always forget how to access a 2d array with a flattened index
                        inputLayer->neuronAt(j).activation = trainingData.at(i).first.at(j/28).at(j%28);
                    }

                    // Now we will compute all the layers
                    for (size_t j = 1; j < layers.size(); ++j) {
                        layers.at(j)->compute(params.at(j-1));
                    }

                    auto lastLayer = layers.at(layers.size() - 1);
                    // C = (expctd - out)^2 + ...
                    for (size_t j = 0; j < expectedOutput.size(); ++j) {
                        cost += (expectedOutput.at(j) - lastLayer->getNeurons().at(j).activation) * (expectedOutput.at(j) - lastLayer->getNeurons().at(j).activation);
                    }
                }
                return cost / trainingData.size(); // average of the cost for all the training data
            }

            double computeCost() const {
                // ASSUME THAT LAYERS COMPLIES (too lazy to check/make a version that is more modular)
                // and we have training data
                // 784 inputs, 10 outputs
                // first, set the input layers

                double cost = 0.0;
                for (size_t i = 0; i < trainingData.size(); ++i) {
                    auto inputLayer = layers.at(0);
                    std::array<double, 10> expectedOutput;
                    for (size_t j = 0; j < 10; ++j) {
                        expectedOutput[j] = (j == trainingData.at(i).second) ? 1.0 : 0.0;
                    }

                    for (size_t j = 0; j < trainingData.at(0).first.size() * trainingData.at(0).first.at(0).size(); ++j) {
                        // j / numCols is the row
                        // j % numCols is the col
                        // puttin' this here cuz i always forget how to access a 2d array with a flattened index
                        inputLayer->neuronAt(j).activation = trainingData.at(i).first.at(j/28).at(j%28);
                    }

                    // Now we will compute all the layers
                    for (size_t j = 1; j < layers.size(); ++j) {
                        layers.at(j)->compute();
                    }

                    auto lastLayer = layers.at(layers.size() - 1);
                    // C = (expctd - out)^2 + ...
                    for (size_t j = 0; j < expectedOutput.size(); ++j) {
                        cost += (expectedOutput.at(j) - lastLayer->getNeurons().at(j).activation) * (expectedOutput.at(j) - lastLayer->getNeurons().at(j).activation);
                    }
                }
                return cost / trainingData.size(); // average of the cost for all the training data
            }
    };
}
