`default_nettype none
`timescale 1ns/1ps

module tcn_network_8ch_2layer #(
    parameter integer CHANNELS = 8,
    parameter integer SAMPLE_WIDTH = 8,
    parameter integer WEIGHT_WIDTH = 8,
    parameter integer ACC_WIDTH = 8,
    parameter integer KERNEL_TAPS = 5
) (
    input wire clk,
    input wire rst,
    input wire sample_valid,
    output wire sample_ready,
    input wire signed [CHANNELS*SAMPLE_WIDTH-1:0] sample_in,
    output wire output_valid,
    output wire signed [CHANNELS*SAMPLE_WIDTH-1:0] sample_out
);
    wire layer0_valid;
    wire layer0_ready;
    wire signed [CHANNELS*SAMPLE_WIDTH-1:0] layer0_out;

    assign sample_ready = layer0_ready;

    tcn_8ch_layer_parallel #(
        .CHANNELS(CHANNELS), .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .KERNEL_TAPS(KERNEL_TAPS), .DILATION(1), .USE_RELU(1)
    ) layer0 (
        .clk(clk), .rst(rst), .sample_valid(sample_valid),
        .sample_ready(layer0_ready), .sample_in(sample_in),
        .output_valid(layer0_valid), .sample_out(layer0_out)
    );

    tcn_8ch_layer_parallel #(
        .CHANNELS(CHANNELS), .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH), .ACC_WIDTH(ACC_WIDTH),
        .KERNEL_TAPS(KERNEL_TAPS), .DILATION(1), .USE_RELU(0)
    ) layer1 (
        .clk(clk), .rst(rst), .sample_valid(layer0_valid),
        .sample_ready(), .sample_in(layer0_out),
        .output_valid(output_valid), .sample_out(sample_out)
    );
endmodule
