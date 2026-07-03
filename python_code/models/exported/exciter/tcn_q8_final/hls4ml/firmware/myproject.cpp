#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    hls::stream<input_t> &x,
    hls::stream<result_t> &layer4_out
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS INTERFACE axis port=x,layer4_out 
    #pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<conv1_weight_t, 96>(w2, "w2.txt");
        nnet::load_weights_from_txt<conv1_bias_t, 32>(b2, "b2.txt");
        nnet::load_weights_from_txt<conv2_weight_t, 96>(w4, "w4.txt");
        nnet::load_weights_from_txt<conv2_bias_t, 1>(b4, "b4.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    hls::stream<layer5_t> layer5_out("layer5_out");
    #pragma HLS STREAM variable=layer5_out depth=16384

    hls::stream<conv1_result_t> layer2_out("layer2_out");
    #pragma HLS STREAM variable=layer2_out depth=16382

    hls::stream<layer3_t> layer3_out("layer3_out");
    #pragma HLS STREAM variable=layer3_out depth=16382

    nnet::transpose<input_t, layer5_t, config5>(x, layer5_out); // transpose_input_for_x

    nnet::conv_1d_cl<layer5_t, conv1_result_t, config2>(layer5_out, layer2_out, w2, b2); // conv1

    nnet::relu<conv1_result_t, layer3_t, relu_config3>(layer2_out, layer3_out); // relu

    nnet::conv_1d_cl<layer3_t, result_t, config4>(layer3_out, layer4_out, w4, b4); // conv2

}

