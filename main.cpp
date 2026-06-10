#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <unistd.h>
#include <limits.h>
#include <random>
#include "libMachineLearning.hpp"
#include <ncurses.h>

#define LEARN_RATE 0.5

#define BATCH_SIZE 500

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
        unsigned char label = '\0';
        unsigned char width = 28;
        unsigned char height = 28;
        file.read(reinterpret_cast<char*>(&label), 1);
        file.read(reinterpret_cast<char*>(&width), 1);
        file.read(reinterpret_cast<char*>(&height), 1);
        file.close();
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

// Assumes little endian (too lazy)
bool writeParamsToFile(const std::string& path, const std::vector<MachineLearning::ParameterStruct>& params, const std::vector<size_t>& neuronsInLayers) {
    std::ofstream file(path, std::ofstream::binary);
    const uint64_t layers = neuronsInLayers.size();
    file.write(reinterpret_cast<const char*>(&layers), sizeof(uint64_t));

    for (size_t i = 0; i < neuronsInLayers.size(); ++i) {
        const uint64_t val = static_cast<const uint64_t>(neuronsInLayers[i]);
        file.write(reinterpret_cast<const char*>(&val), sizeof(uint64_t));
    }

    for (size_t i = 0; i < params.size() ; ++i) {
        for (size_t j = 0; j < params.at(i).biases.size(); ++j) {
            const double val = params.at(i).biases.at(j);
            file.write(reinterpret_cast<const char*>(&val), sizeof(double));
        }

        for (size_t j = 0; j < params.at(i).weights.rows; ++j) {
            for (size_t k = 0; k < params.at(i).weights.cols; ++k) {
                const double val = params.at(i).weights.get(j,k);
                file.write(reinterpret_cast<const char*>(&val), sizeof(double));
            }
        }
    }

    file.close();
    return true;
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
            layers.at(i)->biasAt(j) = ((long)dist(rng)-1000) / 1000.0 * v;
        }
    }

    std::cout << "Reading training data..." << std::endl;
    model.trainingData = readData("trainingData");


    std::cout << "Training..." << std::endl;

    initscr();
    clear();
    timeout(0);

    enum BreakReason {
        Iterations, Magnitude, Exit
    };
    BreakReason reason = Iterations;

    std::vector<double> expected(10);

    std::vector<MachineLearning::ParameterStruct> derivatives(layers.size()-1);
    for (size_t j = 1; j < layers.size(); ++j) {
        derivatives.at(j-1).biases = std::vector<double>(layers.at(j)->getBiases().size());
        derivatives.at(j-1).weights = MachineLearning::Matrix(layers.at(j)->getWeights().getRows(), layers.at(j)->getWeights().getCols());
    }

    std::vector<std::vector<double>> errors(layers.size()-1);
    for (size_t i = 0; i < errors.size(); ++i) {
        errors[i] = std::vector<double>(layers.at(i+1)->getNeurons().size());
    }

    std::vector<size_t> trainingDataIndicies(model.trainingData.size());
    for (size_t i = 0; i < trainingDataIndicies.size(); ++i) {
        trainingDataIndicies[i] = i;
    }

    model.prepare();

    for (size_t i = 0; i < 999999; ++i) {
        std::vector<MachineLearning::ParameterStruct> s = model.backPropagate(derivatives, errors, expected);

        // Reset derivatives, errors, and expected
        for (size_t j = 0; j < errors.size(); ++j) {
            for (size_t k = 0; k < errors[j].size(); ++k) {
                errors[j][k] = 0;
            }
        }

        for (size_t j = 0; j < expected.size(); ++j) {
            expected[j] = 0;
        }

        // Use the gradient descent
        double mag = 0;
        for (size_t j = 0; j < s.size(); ++j) {
            for (size_t k = 0; k < s.at(j).weights.rows; ++k) {
                for (size_t l = 0; l < s.at(j).weights.cols; ++l) {
                    model.layers.at(j+1)->weightAt(k, l) -= s.at(j).weights.at(k, l) * LEARN_RATE;
                    derivatives[j].weights.at(k, l) = 0;
                    mag += s.at(j).weights.at(k,l) * s.at(j).weights.at(k,l);
                }
            }

            for (size_t k = 0; k < s.at(j).biases.size(); ++k) {
                model.layers.at(j+1)->biasAt(k) -= s.at(j).biases.at(k) * LEARN_RATE;
                derivatives[j].biases[k] = 0;
                mag += s.at(j).biases.at(k) * s.at(j).biases.at(k);
            }
        }

        mag = std::sqrt(mag);

        clear();
        move(0,0);
        printw("Number of iterations: %lu\n", i+1);
        printw("Current gradient magnitude: %g\n", mag);
        printw("Current cost: %g\n", model.computeCost(trainingDataIndicies, expected, BATCH_SIZE));
        printw("Press q to quit training\n");
        refresh();
        if (mag < 1e-7) {
            reason = Magnitude;
            break;
        }

        // If we did type anything
        // The < 400 is my cheat of ensuring that we actually typed something printable, cuz it was exitting even if we resized the window
        int ch;
        if ((ch = getch()) != ERR && ch < 400 && ch == 'q') {
            reason = Exit;
            break;
        }
    }

    endwin();
    switch (reason) {
        case Iterations:
            std::cout << "Exited because the number of iterations was very large\n";
            break;
        case Magnitude:
            std::cout << "Exited because the magnitude of the gradient was very little\n";
            break;
        case Exit:
            std::cout << "Exited because you told me to\n";
            break;
    }
    std::cout << "Done training!\n";

    std::cout << "Testing...\n";
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
            const auto& lay = model.layers.at(model.layers.size()-1);

            if (lay->getNeurons().at(j).activation/lastLayerSum > prediction.second) {
                prediction = {j, lay->getNeurons().at(j).activation/lastLayerSum};
            }
        }

        if (prediction.first == lab) {
            ++numCorrect;
        }
    }

    std::cout << "Accuracy: " << numCorrect*1.0/testingData.size() << "\n";

    // Test the stupidity of the model (give it all zeros)
    // TODO: Random noise
    for (size_t i = 0; i < model.layers.at(0)->getNeurons().size(); ++i) {
        model.layers.at(0)->neuronAt(i).activation = 0;
    }
    for (size_t i = 1; i < model.layers.size(); ++i) {
        model.layers.at(i)->compute();
    }

    double lastLayerSum = 0;
    for (size_t j = 0; j < model.layers.at(model.layers.size()-1)->getNeurons().size(); ++j) {
        lastLayerSum += model.layers.at(model.layers.size()-1)->getNeurons().at(j).activation;
    }

    std::vector<std::pair<size_t, double>> predictions;
    for (size_t j = 0; j < model.layers.at(model.layers.size()-1)->getNeurons().size(); ++j) {
        const auto& lay = model.layers.at(model.layers.size()-1);
        predictions.push_back({j, lay->getNeurons().at(j).activation/lastLayerSum});
    }

    // sort predictions by second thing
    struct {
        bool operator()(const std::pair<size_t, double>& a, const std::pair<size_t, double>& b) {
            // return true if a goes BEFORE b
            return a.second > b.second;
        }
    } comp;

    std::sort(predictions.begin(), predictions.end(), comp);
    std::cout << "Here is what the model thinks about the empty image (all dark)\n";
    for (size_t i = 0; i < predictions.size(); ++i) {
        std::cout << predictions[i].first << " " << predictions[i].second*100 << "%\n";
    }


    model.end();

    const std::filesystem::path exePath(getExecutablePath());
    const std::filesystem::path exeDir(exePath.string().substr(0,exePath.string().length()-exePath.filename().string().length()));
    std::cout << "Would you like to save these parameters to " << exeDir.string() + "params.bin" << "? [y\\n] " << std::flush;

    std::string res;
    std::cin >> res;

    if (res == "y") {
        std::cout << "Saving to file: " << exeDir.string() + "params.bin" << std::endl;
        std::vector<MachineLearning::ParameterStruct> params(model.layers.size()-1);
        for (size_t i = 0; i < model.layers.size()-1; ++i) {
            params.at(i).weights = model.layers.at(i+1)->getWeights();
            params.at(i).biases = model.layers.at(i+1)->getBiases();
        }

        std::vector<size_t> neuronsInLayers(model.layers.size());
        for (size_t i = 0; i < neuronsInLayers.size(); ++i) {
            neuronsInLayers[i] = model.layers[i]->getNeurons().size();
        }
        if (writeParamsToFile(exeDir.string() + "params.bin", params, neuronsInLayers)) {
            // Success
        } else {
            // Fail
        }
    } else {
        std::cout << "Not saving" << std::endl;
    }
    return 0;
}
