//
// Created by oleg on 21.07.24.
//
#include "gtest/gtest.h"
#include "eigen3/Eigen/Dense"
#include "Neuron.h"

using namespace Eigen;

class NeuronTest: public ::testing::Test {
protected:
    void SetUp(){};

    void TearDown(){};
};


float ActivationSigmoid(float input){
    return 1 / (1 + std::exp(-input));
};



TEST_F(NeuronTest, neuron_sigmoid_small) {
    Matrix<float, 1, 1> weights;
    weights << 1.0;
    nn_neuron::Neuron<float, 1, 1> test_neuron(weights, ActivationSigmoid);
    Matrix<float, 1, 1> input_data;
    input_data << 1.0;
    auto output = test_neuron.Transform(input_data);
    ASSERT_NEAR(output, 0.8807, 0.1);
}

TEST_F(NeuronTest, neuron_sigmoid_large) {
    Matrix<float, 1, 100> weights  = Matrix<float, 1, 100>::Random(1, 100);;

    nn_neuron::Neuron<float, 100, 100> test_neuron(weights, ActivationSigmoid);
    Matrix<float, 1, 100>  input_data = Matrix<float, 1, 100>::Random(1, 100);
    auto output = test_neuron.Transform(input_data);
    std::cout << "Output is: " << output << std::endl;
}
