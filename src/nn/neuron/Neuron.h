//
// Created by oleg on 21.07.24.
//

#ifndef NEURAL_NETWORK_SIMPLE_NEURON_H
#define NEURAL_NETWORK_SIMPLE_NEURON_H

#include "vector"
#include "functional"
#include "iostream"
#include "eigen3/Eigen/Dense"


namespace nn_neuron {
    using namespace Eigen;

    template<typename T, size_t inputs, size_t outputs>
    class Neuron {
    private:
        std::function<T(T)> activation_;
        Matrix<T, 1, inputs> weights_;
    public:
        Neuron() : activation_([](T x) { return x; }), weights_(Matrix<T, 1, inputs>::Random()) {}

        explicit Neuron(std::function<T(T)> activation)
                : activation_(activation), weights_(Matrix<T, 1, inputs>::Random()) {}

        Neuron(Matrix<T, 1, inputs>& weights, std::function<T(T)> activation)
                : weights_(weights), activation_(activation) {}

        void setActivation(std::function<T(T)> activation) {
            activation_ = activation;
        }

        T Transform(const Matrix<T, 1, inputs>& input_matrix) {
            Matrix<T, 1, inputs> weight_normalized_input = input_matrix.cwiseProduct(weights_);
            return activation_(weight_normalized_input.sum());
        };

        Matrix<T, 1, inputs> GetWeights() {
            return weights_;
        }

        void SetWeights(Matrix<T, 1, inputs>& new_weights) {
            weights_ = new_weights;
        }

        static const std::tuple<size_t, size_t> size() {
            return {inputs, outputs};
        }
    };
};


#endif //NEURAL_NETWORK_SIMPLE_NEURON_H
