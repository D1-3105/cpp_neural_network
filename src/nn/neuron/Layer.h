//
// Created by oleg on 21.07.24.
//

#ifndef NEURAL_NETWORK_SIMPLE_LAYER_H
#define NEURAL_NETWORK_SIMPLE_LAYER_H

#include "eigen3/Eigen/Dense"
#include "tbb/tbb.h"
#include "mutex"

namespace nn_layer {
    using namespace Eigen;

    template<typename NeuronClass, typename ActivationType>
    class HiddenLayer {
    protected:
        std::vector<NeuronClass> neurons_;
    public:
        explicit HiddenLayer(long long neuron_count, ActivationType layer_activation_func): neurons_(neuron_count) {
            for (size_t i = 0; i < neuron_count; i++)
                neurons_[i] = NeuronClass(layer_activation_func);
        };

        NeuronClass operator[](size_t idx){
            return neurons_[idx];
        }

        template<typename InputT, typename OutputT>
        std::vector<OutputT> CollectNeurons(InputT& layer_input) {
            std::vector<OutputT> on_out(neurons_.size());
            tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, neurons_.size()),
                [layer_input, on_out, this](tbb::blocked_range<size_t> neuron_idx_block){
                    for (size_t i = neuron_idx_block.begin(); i < neuron_idx_block.end(); ++i) {
                        auto res = neurons_[i].Transform(layer_input);
                        on_out[i] = res;
                    }
                }
            );
        };
    };

    template<typename InputValueT>
    class InputLayer {
    protected:
        std::vector<InputValueT> current_input_;
    protected:
        explicit InputLayer(long long inputs);
    };
}


#endif //NEURAL_NETWORK_SIMPLE_LAYER_H
