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

    template<typename WeightType, long long inputs, long long outputs>
    class Neuron {
    private:
        std::function<WeightType (WeightType)> activation_;
        Matrix<WeightType, 1, inputs> weights_;
    public:
        Neuron(
                Matrix<WeightType, 1, inputs>& weights,
                std::function<WeightType (WeightType)> activation
        ): weights_(weights), activation_(activation) {};

        explicit Neuron(std::function<Matrix<WeightType, outputs, inputs> (Matrix<WeightType, 1, inputs>)> activation) {
            Matrix<WeightType, 1, inputs> init_weights;
            Neuron(init_weights, activation);
        };

        WeightType Transform(Matrix<WeightType, 1, inputs> input_matrix) {
            Matrix<WeightType, 1, inputs> weight_normalized_input = input_matrix + weights_;
            return activation_(weight_normalized_input.sum());
        };

        Matrix<WeightType, 1, inputs> GetWeights() {
            return weights_;
        }

        void SetWeights(Matrix<WeightType, 1, inputs>& new_weights) {
            weights_ = new_weights;
        }

        static std::tuple<long, long> size(){
            return {inputs, outputs};
        }
    };
};


#endif //NEURAL_NETWORK_SIMPLE_NEURON_H
