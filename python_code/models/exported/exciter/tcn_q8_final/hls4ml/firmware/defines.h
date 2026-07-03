#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef nnet::array<ap_fixed<16,6>, 16384*1> input_t;
typedef nnet::array<ap_fixed<16,6>, 1*1> layer5_t;
typedef ap_fixed<35,15> conv1_accum_t;
typedef nnet::array<ap_fixed<35,15>, 32*1> conv1_result_t;
typedef ap_fixed<16,6> conv1_weight_t;
typedef ap_fixed<16,6> conv1_bias_t;
typedef nnet::array<ap_fixed<16,6>, 32*1> layer3_t;
typedef ap_fixed<18,8> relu_table_t;
typedef ap_fixed<40,20> conv2_accum_t;
typedef nnet::array<ap_fixed<40,20>, 1*1> result_t;
typedef ap_fixed<16,6> conv2_weight_t;
typedef ap_fixed<16,6> conv2_bias_t;

// hls-fpga-machine-learning insert emulator-defines


#endif
