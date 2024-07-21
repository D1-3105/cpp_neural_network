//
// Created by oleg on 21.07.24.
//
#include "gtest/gtest.h"
#include "eigen3/Eigen/Dense"
#include "Neuron.h"
#include "Layer.h"

using namespace Eigen;

class LayerTest : public ::testing::Test {
protected:

    using NeuronClass = nn_neuron::Neuron<float, 10, 1>;
    using HiddenLayerClass = nn_layer::HiddenLayer<NeuronClass, float>;

    IOFormat TensorPrintFMT;

    std::shared_ptr<nn_layer::HiddenLayer<NeuronClass, float>> hidden_layer_ptr_;

    void SetUp() override {
        auto activation = std::function<float(float)>(ActivationSigmoid);
        TensorPrintFMT = IOFormat(4, 0, ", ", ", ", "[", "]");

        hidden_layer_ptr_ = std::make_shared<HiddenLayerClass>(10, activation);
    }

    static float ActivationSigmoid(float x) {
        return 1 / (1 + std::exp(-x));
    }

    void TearDown() override {}
};


TEST_F(LayerTest, test_io) {


    nn_layer::InputLayer<float, 10, 1> in;
    // Set current input to the input layer
    Matrix<float, 1, 10> input(1, 10);
    input << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0;
    in.SetCurrentInput(input);
    Matrix<float, 10, 1> output = in.PassInput<HiddenLayerClass, float>(*hidden_layer_ptr_);
    std::cout << output.format(TensorPrintFMT) << std::endl;

    nn_layer::OutputLayer<float, 10, 1> output_layer(output);
}
