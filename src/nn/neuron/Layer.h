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

    template<class NeuronClass, typename WeightT>
    class HiddenLayer {
    protected:
        std::vector<NeuronClass> neurons_;
    public:
        HiddenLayer(size_t neuron_count, std::function<WeightT(WeightT)> layer_activation_func)
                : neurons_(neuron_count) {
            for (auto& neuron : neurons_) {
                neuron.setActivation(layer_activation_func);
            }
        }

        NeuronClass operator[](size_t idx) {
            return neurons_[idx];
        }

        template<typename InputT, typename OutputT>
        Matrix<OutputT, Dynamic, Dynamic>
        CollectNeurons(const InputT& layer_input) {
            auto size = NeuronClass::size();
            Matrix<OutputT, Dynamic, Dynamic> on_out = Matrix<OutputT, Dynamic, Dynamic>::Zero(std::get<0>(size), std::get<1>(size));
            Matrix<OutputT, Dynamic, Dynamic> temp = Matrix<OutputT, Dynamic, Dynamic>::Zero(on_out.rows(), on_out.cols());

            tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, neurons_.size()),
                    [layer_input, &temp, this](tbb::blocked_range<size_t> neuron_idx_block) {
                        for (size_t i = neuron_idx_block.begin(); i < neuron_idx_block.end(); ++i) {
                            auto res = neurons_[i].Transform(layer_input);
                            temp(i) = res;
                        }
                    }
            );

            on_out = temp;

            return on_out;
        }
    };

    template<typename InputValueT, size_t Rows, size_t Cols>
    class InputLayer {
    protected:
        Matrix<InputValueT, Dynamic, Dynamic> current_input_;
    public:
        explicit InputLayer() {
            current_input_ = Matrix<InputValueT, Dynamic, Dynamic>::Zero(Rows, Cols);
        };

        template<class HiddenLayerT, typename OutputValueT>
        Matrix<OutputValueT, Dynamic, Dynamic> PassInput(HiddenLayerT& layer) const {
            auto m = layer.template CollectNeurons<const Matrix<InputValueT, Dynamic, Dynamic>&, OutputValueT>(current_input_);
            return m;
        }

        void SetCurrentInput(const Matrix<InputValueT, Dynamic, Dynamic>& current_input) {
            current_input_ = current_input;
        }
    };

    template<typename OutputValueT, int Rows, int Cols>
    class OutputLayer {
    protected:
        std::shared_ptr<Matrix<OutputValueT, Cols, Rows>> layer_output_;
    public:
        explicit OutputLayer(Matrix<OutputValueT, Rows, Cols>& out) {
            layer_output_ = std::make_shared<Matrix<OutputValueT, Cols, Rows>>(out);
        }
    };
}


#endif //NEURAL_NETWORK_SIMPLE_LAYER_H
