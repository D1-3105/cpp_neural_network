cmake_minimum_required(VERSION 3.28)
project(neural_network_simple)

set(CMAKE_CXX_STANDARD 17)

# tbb
find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIRS})
link_directories(${TBB_LIBRARY_DIRS})

# eigen
find_package(Eigen3 REQUIRED NO_MODULE)

# csv parser
include_directories(include/csv-parser/single_include)
find_package(Threads REQUIRED)

# Option to enable or disable building of tests
set(BUILD_TESTS ON CACHE BOOL "Build the unit tests." FORCE)

# Conditionally compiling the test executable if BUILD_TESTS is set to ON
if(BUILD_TESTS)
    # Finding Google Test framework
    find_package(GTest REQUIRED)
    include_directories(${GTEST_INCLUDE_DIRS})

    # neuron testing
    add_executable(tests_neuron
            qa/tests_neuron/main.cpp
            src/nn/neuron/Neuron.h
            qa/tests_neuron/test_neuron.cpp
            qa/tests_neuron/test_neuron.cpp
            src/nn/neuron/Neuron.h
            qa/tests_neuron/test_layer.cpp
    )
    include_directories(${CMAKE_SOURCE_DIR}/src/nn/neuron)
    target_link_libraries(tests_neuron PRIVATE ${GTEST_LIBRARIES} GTest::GTest GTest::gmock TBB::tbb)
    # Discover and run the Google Tests
    gtest_discover_tests(tests_neuron)
endif()
