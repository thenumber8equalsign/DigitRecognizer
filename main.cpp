#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include "libMachineLearning.hpp"

#include <fstream>
#include <filesystem>

#include <unistd.h>
#include <climits>

#include <random>
#define LEARN_RATE 0.5

#ifdef __linux__
std::string getExecutablePath() {
    char path[PATH_MAX+1];

    ssize_t len = readlink("/proc/self/exe", path, PATH_MAX);
    if (len == -1) {
        throw std::runtime_error("readlink fail");
    }
    path[len]= '\0';
    return std::string(path);
}
#else
#error "Linux only, I was too lazy to make it work on windows/mac/bsd/other"
#endif

inline __attribute__((always_inline)) std::vector<std::string> split(std::string s, std::string delim) {
    std::vector<std::string> res;
    auto pos = s.find(delim);
    if (pos == std::string::npos) {
        res.push_back(s);
        return res;
    }
    while (pos != std::string::npos) {
        // after finding the first delim, take the substring
        res.push_back(s.substr(0, pos));

        s.erase(0, pos+1); // erase the substring from the string, including the delim

        pos = s.find(delim); // find the next delim
    }
    res.push_back(s);
    return res;
}

std::vector<std::pair<MachineLearning::Image, char>> readData(std::string dir) {
    const std::filesystem::path exePath(getExecutablePath());
    const std::filesystem::path exeDir(exePath.string().substr(0,exePath.string().length()-exePath.filename().string().length()));

    const std::filesystem::path trainingDir(exeDir.string() + dir + "/");

    std::vector<std::pair<MachineLearning::Image, char>> images;

    size_t i = 0;
    for (auto const& dir_entry : std::filesystem::directory_iterator(trainingDir)) {
        std::ifstream file(dir_entry.path(), std::ifstream::binary);
        char label = '\0';
        char width = 28;
        char height = 28;
        file.read(&label, 1);
        file.read(&width, 1);
        file.read(&height, 1);
        MachineLearning::Image image;
        for (size_t i = 0; i < width*height; ++i) {
            unsigned char val = 0;
            file.read(reinterpret_cast<char*>(&val), 1);

            image.at(i/height).at(i%width) = val / 255.0;
        }

        auto s = split(dir_entry.path().string(), "-");
        // if (s[s.size()-1] == "2.bin") {
            images.push_back({image, label});
            ++i;
        // }
        // if (i >= 100) break;
    }
    return images;
}

void appendToVector(std::vector<double>& v, const double a) {
    for (size_t i = v.size()-1; i > 0; --i) {
        v.at(i) = v.at(i-1);
    }
    v.at(0) = a;
}

int main() {
    MachineLearning::Model model;
    std::vector<std::shared_ptr<MachineLearning::Layer>> layers(1 + 2 + 1); // 2 hidden layers
    layers.at(0) = std::make_shared<MachineLearning::Layer>(28*28);
    layers.at(1) = std::make_shared<MachineLearning::Layer>(16);
    layers.at(2) = std::make_shared<MachineLearning::Layer>(16);
    layers.at(layers.size()-1) = std::make_shared<MachineLearning::Layer>(10);

    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0,2000);

    model.layers = layers;
    model.linkLayers();

    std::cout << "Initializing layers..." << std::endl;
    for (size_t i = 1; i < layers.size(); ++i) {

        auto weights = layers.at(i)->getWeights();
        auto biases = layers.at(i)->getBiases();

        // +- sqrt(6/(neuron_in_prev + neuron_in_cur))
        const double v = std::sqrt(6.0/(layers.at(i-1)->getNeurons().size() + layers.at(i)->getNeurons().size()));

        // Initialize the parameters to be random doubles within [-10, 10]
        for (size_t j = 0; j < weights.getRows(); ++j) {
            for (size_t k = 0; k < weights.getCols(); ++k) {
                layers.at(i)->weightAt(j, k) = ((long)dist(rng)-1000) / 1000.0 * v;
            }
        }

        for (size_t j = 0; j < biases.size(); ++j) {
            layers.at(i)->biasesAt(j) = ((long)dist(rng)-1000) / 1000.0 * v;
        }
    }

    std::cout << "Reading training data..." << std::endl;
    model.trainingData = readData("trainingData");


    std::cout << "Training..." << std::endl;
    model.backPropagate();
    std::vector<double> previousCosts(10, -INFINITY);
    bool brokeAvg = false;
    for (size_t i = 0; i < 99999; ++i) {
        std::vector<MachineLearning::ParameterStruct> s = model.backPropagate();

        // Use the gradient descent
        for (size_t j = 0; j < s.size(); ++j) {
            for (size_t k = 0; k < s.at(j).weights.getRows(); ++k) {
                for (size_t l = 0; l < s.at(j).weights.getCols(); ++l) {
                    model.layers.at(j+1)->weightAt(k, l) -= s.at(j).weights.at(k).at(l) * LEARN_RATE;
                }
            }

            for (size_t k = 0; k < s.at(j).biases.size(); ++k) {
                model.layers.at(j+1)->biasesAt(k) -= s.at(j).biases.at(k) * LEARN_RATE;
            }
        }


        double cost = model.computeCost();
        // Print the cost
        std::cout << "Current cost: " << cost << std::endl;
        appendToVector(previousCosts, cost);

        // Calculate the average delta
        double avg = 0.0;
        for (size_t j = 1; j < previousCosts.size(); ++j) {
            avg += previousCosts.at(j-1) - previousCosts.at(j);
        }
        avg /= previousCosts.size();
        std::cout << "Current average difference " << avg << std::endl;
        if (std::abs(avg) < 1e-7) {
            brokeAvg = true;
            break;
        }
    }

    std::cout << "Done training!" << std::endl;
    if (brokeAvg) {
        std::cout << "Exited because the average difference was very little" << std::endl;
    } else {
        std::cout << "Exited because the number of iterations was very large" << std::endl;
    }


    std::cout << "Testing..." << std::endl;
    std::cout << "Reading testing data..." << std::endl;
    auto testingData = readData("testingData");
    size_t numCorrect = 0;
    for (size_t i = 0; i < testingData.size(); ++i) {
        char lab = testingData.at(i).second;
        for (size_t j = 0; j < model.layers.at(0)->getNeurons().size(); ++j) {
            model.layers.at(0)->neuronAt(j).activation = testingData.at(i).first.at(j / 28).at(j % 28);
        }

        for (size_t j = 1; j < model.layers.size(); ++j) {
            model.layers.at(j)->compute();
        }

        double lastLayerSum = 0;
        for (size_t j = 0; j < model.layers.at(model.layers.size()-1)->getNeurons().size(); ++j) {
            lastLayerSum += model.layers.at(model.layers.size()-1)->getNeurons().at(j).activation;
        }

        std::pair<size_t, double> prediction = {0,-INFINITY};
        for (size_t j = 0; j < model.layers.at(model.layers.size()-1)->getNeurons().size(); ++j) {
            auto& lay = model.layers.at(model.layers.size()-1);

            if (lay->getNeurons().at(j).activation/lastLayerSum > prediction.second) {
                prediction = {j, lay->getNeurons().at(j).activation/lastLayerSum};
            }
        }

        if (prediction.first == lab) {
            ++numCorrect;
        }
    }

    std::cout << "Accuracy: " << numCorrect*1.0/testingData.size() << std::endl;
    return 0;
}
