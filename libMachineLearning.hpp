#pragma once
#include <vector>
#include <stdexcept>
#include <memory>
#include <array>

inline std::vector<double> operator + (const std::vector<double>& lhs, const std::vector<double>& rhs) {
    if (lhs.size() != rhs.size()) throw std::invalid_argument("sizes not equal");

    std::vector<double> res(lhs.size());
    for (size_t i = 0; i < res.size(); ++i) {
        res[i] = lhs[i] + rhs[i];
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

    struct Matrix {
        std::vector<std::vector<double>> data;

        Matrix(const size_t r, const size_t c) {
            data = std::vector<std::vector<double>>(r, std::vector<double>(c, 0));
        }

        Matrix() {}

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


        Matrix operator * (const double scalar) const {
            Matrix result;

            try {
                result = Matrix(data.size(), data.at(0).size());
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


        Matrix& operator *= (const double scalar) {
            for (size_t i = 0; i < data.size(); ++i) {
                for (size_t j = 0; j < data[i].size(); ++j) {
                    data.at(i).at(j) *= scalar;
                }
            }
            return *this;
        }

        Matrix operator * (const Matrix& other) const {
            try {
                // Check restrictions (# col here has to be # row there)
                if (data.at(0).size() != other.data.size()) {
                    throw std::domain_error("undefined"); // Domain error was the closest error type I could find, so I use it cuz i dont wanna make my own
                }
                Matrix mat(data.size(), other.data.at(0).size());
                for (size_t i = 0; i < mat.data.size(); ++i) {
                    for (size_t j = 0; j < mat.data[0].size(); ++j) {
                        // dot the LHS matrix's row with the RHS matrix's col
                        // mat[i][j] = dot(data.at(i), other.colAt(j));
                        mat.data[i][j] = 0;

                        // Go through everything in this row
                        for (size_t k = 0; k < data[0].size(); ++k) {
                            mat.data[i][j] += data[i][k] * other.data[k][i];
                        }

                    }
                }

                return mat;
            } catch (std::out_of_range) {
                throw std::runtime_error("malformed matrix");
            }
        }

        inline __attribute__((always_inline)) std::vector<double> operator * (const std::vector<Neuron>& other) const {
            // try {
                // if (data.at(0).size() != other.size()) {
                //     throw std::domain_error("undefined");
                // }

                std::vector<double> result(data.size());
                for (size_t i = 0; i < data.size(); ++i) {
                    result[i] = dot(data[i], other);
                }

                return result;
            // } catch (std::out_of_range) {
            //     throw std::runtime_error("malformed matrix");
            // }
        }

        bool operator == (const Matrix& other) const {
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

        size_t getRows() const { return data.size(); }
        size_t getCols() const { return data.at(0).size(); }
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
        public:
            std::weak_ptr<Layer> prev;

            Layer(std::shared_ptr<Layer> prev, size_t numNeurons) {
                this->prev = prev;
                neurons.resize(numNeurons);
                biases.resize(numNeurons);

                // The number of rows is the number of neurons here,
                // the number of columns is the number of neurons in the previous layer
                weights.data = std::vector<std::vector<double>>(numNeurons, std::vector<double>(prev->neurons.size()));
            }

            Layer(size_t numNeurons) {
                neurons.resize(numNeurons);
                biases.resize(numNeurons);
            }

            Layer() {}

            void updateWeightDimensions() {
                auto prev = this->prev.lock();
                weights.data = std::vector<std::vector<double>>(neurons.size(), std::vector<double>(prev->neurons.size()));
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
                return weights.data.at(row).at(col);
            }

            Neuron& neuronAt(size_t i) {
                // only allow setting a neuron if we are the starting layer
                std::shared_ptr<Layer> tmp = prev.lock();
                if (tmp) {
                    throw std::runtime_error("Can not set activation of a hidden or output layer");
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

            void compute(const ParameterStruct& ps) {
                std::shared_ptr<Layer> prev = this->prev.lock();
                std::vector<double> z = ps.weights * prev->neurons + ps.biases;
                neurons = sigmoid(z);
            }
    };

    typedef std::array<std::array<double, 28>, 28> Image;

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

            std::pair<std::vector<ParameterStruct>, double> gradientDecsent() {
                // size_t numWeights = 0;
                // size_t numBiases = 0;
                // for (size_t i = 0; i < layers.size() - 1; ++i) {
                //     numWeights += layers.at(i)->getNeurons().size() * layers.at(i+1)->getNeurons().size();
                //     numBiases += layers.at(i+1)->getNeurons().size();
                // }

                // std::vector<double> grad(numWeights + numBiases);
                std::vector<ParameterStruct> params(layers.size()-1); // using this struct instead of a raw double vector makes my life SO much easier

                // populate grad with the partial derivatives
                for (size_t i = 1; i < layers.size(); ++i) {
                    params.at(i-1) = {layers[i]->getWeights(),layers[i]->getBiases()};
                }

                std::vector<ParameterStruct> grad(params.size());
                double cost = computeCost();

                for (size_t i = 0; i < params.size(); ++i) {
                    // 1) compute the partial derivative of cost w/ respect to the current parameter
                    // 2) negate it and put it in grad
                    // TODO: actual derivatives

                    Matrix weights(params[i].weights.getRows(), params[i].weights.getCols());
                    std::vector<double> biases(params[i].biases.size());

                    for (size_t j = 0; j < params[i].weights.getRows(); ++j) {
                        for (size_t k = 0; k < params[i].weights.getCols(); ++k) {
                            // constexpr double h = 0.000001;
                            // params[i].weights.data.at(j).at(k) += h;
                            // const double costB = computeCost(params);
                            // params[i].weights.data.at(j).at(k) -= h;

                            // weights.data.at(j).at(k) = (costB - cost) / h;
                            double derivTotal = 0.0;

                            // Load the images, average the derivatives
                            for (size_t m = 0; m < trainingData.size(); ++m) {
                                for (size_t n = 0; n < layers[0]->getNeurons().size(); ++n) {
                                    layers[0]->neuronAt(n).activation = trainingData.at(m).first.at(n/28).at(n%28);
                                }

                                char lab = trainingData.at(m).second;

                                std::vector<double> expected(10);
                                for (size_t n = 0; n < expected.size(); ++n) {
                                    if (n == lab) {
                                        expected.at(n) = 1.0;
                                    } else {
                                        expected.at(n) = 0.0;
                                    }
                                }

                                for (size_t n = 1; n < layers.size(); ++n) {
                                    layers[n]->compute();
                                }


                                // start with 1
                                // multiply by the previous neuron's activation
                                double deriv = 1;
                                // Remember, the row of the weights matrix is for the new neuron
                                // the column is for the old neuron
                                deriv *= layers[i]->getNeurons().at(k).activation;

                                deriv *= sigmoidDerivative(layers[i+1]->getNeurons().at(j).weightedInput);
                                // keep multiplying by all the weights until the top neuron in the last layer
                                // first, we multiply by the next weights [0][row]
                                // then multiply by all the next weights [0][0]
                                if (i < params.size()-1) {
                                    deriv *= params.at(i+1).weights.data.at(0).at(j);
                                    deriv *= costDerivative(layers[layers.size()-1]->getNeurons().at(0).activation, expected.at(0));
                                    for (size_t l = i+2; l < params.size(); ++l) {
                                        deriv *= params.at(l).weights.data.at(0).at(0);
                                        deriv *= sigmoidDerivative(layers.at(l)->getNeurons().at(0).weightedInput);
                                    }
                                    deriv *= sigmoidDerivative(layers.at(layers.size()-1)->getNeurons().at(0).weightedInput);
                                } else {
                                    // If this is the final layer we are not going to the top neuron, we use a different one
                                    // delC/delAj * delAj/delZj * delZj/delwjk
                                    // CostDeriv    sigmoidDeriv   activation prev (already done)
                                    deriv *= costDerivative(layers[i+1]->getNeurons().at(j).activation, expected.at(j));
                                }
                                derivTotal += deriv;
                            }

                            weights.data.at(j).at(k) = derivTotal / trainingData.size();
                        }
                    }

                    for (size_t j = 0; j < biases.size(); ++j) {
                        // constexpr double h = 0.000001;
                        // params[i].biases.at(j) += h;
                        // const double costB = computeCost(params);
                        // params[i].biases.at(j) -= h;

                        // biases.at(j) = (costB - cost) / h;
                        double derivTotal = 0.0;

                        // Load the images, average the derivatives
                        for (size_t m = 0; m < trainingData.size(); ++m) {
                            for (size_t n = 0; n < layers[0]->getNeurons().size(); ++n) {
                                layers[0]->neuronAt(n).activation = trainingData.at(m).first.at(n/28).at(n%28);
                            }

                            char lab = trainingData.at(m).second;

                            std::vector<double> expected(10);
                            for (size_t n = 0; n < expected.size(); ++n) {
                                if (n == lab) {
                                    expected.at(n) = 1.0;
                                } else {
                                    expected.at(n) = 0.0;
                                }
                            }

                            for (size_t n = 1; n < layers.size(); ++n) {
                                layers[n]->compute();
                            }


                            // start with 1
                            double deriv = 1;
                            // Remember, the row of the weights matrix is for the new neuron
                            // the column is for the old neuron

                            deriv *= sigmoidDerivative(layers[i+1]->getNeurons().at(j).weightedInput);
                            // keep multiplying by all the weights until the top neuron in the last layer
                            // first, we multiply by the next weights [0][row]
                            // then multiply by all the next weights [0][0]
                            if (i < params.size()-1) {
                                deriv *= params.at(i+1).weights.data.at(0).at(j);
                                deriv *= costDerivative(layers[layers.size()-1]->getNeurons().at(0).activation, expected.at(0));
                                for (size_t l = i+2; l < params.size(); ++l) {
                                    deriv *= params.at(l).weights.data.at(0).at(0);
                                    deriv *= sigmoidDerivative(layers.at(l)->getNeurons().at(0).weightedInput);
                                }
                                deriv *= sigmoidDerivative(layers.at(layers.size()-1)->getNeurons().at(0).weightedInput);
                            } else {
                                // If this is the final layer we are not going to the top neuron, we use a different one
                                // delC/delAj * delAj/delZj * delZj/delwjk
                                // CostDeriv    sigmoidDeriv   activation prev (already done)
                                deriv *= costDerivative(layers[i+1]->getNeurons().at(j).activation, expected.at(j));
                            }
                            derivTotal += deriv;
                        }

                        biases.at(j) = derivTotal / trainingData.size();
                    }

                    grad.at(i) = {weights, biases};
                }

                return {grad, cost};
            }

            double computeCost(const std::vector<ParameterStruct>& params) {
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
