`default_nettype none
`timescale 1ns/1ps

module tcn_network_basic #(
    parameter integer SAMPLE_WIDTH = 8,
    parameter integer WEIGHT_WIDTH = 8,
    parameter integer ACC_WIDTH = 32,
    parameter integer OUTPUT_SHIFT = 0
) (
    input wire clk,
    input wire rst,
    input wire sample_valid,
    output wire sample_ready,
    input wire signed [SAMPLE_WIDTH-1:0] sample_in,
    output wire output_valid,
    output wire signed [SAMPLE_WIDTH-1:0] sample_out
);

    wire layer0_ready;
    wire layer0_valid;
    wire signed [SAMPLE_WIDTH-1:0] layer0_out;
    wire layer1_ready;
    wire layer1_valid;
    wire signed [SAMPLE_WIDTH-1:0] layer1_out;
    wire output_layer_ready;

    assign sample_ready = layer0_ready;

    tcn_layer_basic #(
        .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .DILATION(1),
        .OUTPUT_SHIFT(OUTPUT_SHIFT),
        .USE_RELU(1),
        .W0(8'sd2),
        .W1(-8'sd1),
        .W2(8'sd1),
        .W3(8'sd0),
        .W4(8'sd0),
        .BIAS(32'sd0)
    ) layer0 (
        .clk(clk),
        .rst(rst),
        .sample_valid(sample_valid),
        .sample_ready(layer0_ready),
        .sample_in(sample_in),
        .output_valid(layer0_valid),
        .sample_out(layer0_out)
    );

    tcn_layer_basic #(
        .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .DILATION(2),
        .OUTPUT_SHIFT(OUTPUT_SHIFT),
        .USE_RELU(1),
        .W0(8'sd1),
        .W1(8'sd1),
        .W2(-8'sd1),
        .W3(8'sd0),
        .W4(8'sd0),
        .BIAS(32'sd0)
    ) layer1 (
        .clk(clk),
        .rst(rst),
        .sample_valid(layer0_valid),
        .sample_ready(layer1_ready),
        .sample_in(layer0_out),
        .output_valid(layer1_valid),
        .sample_out(layer1_out)
    );

    tcn_layer_basic #(
        .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .DILATION(1),
        .OUTPUT_SHIFT(OUTPUT_SHIFT),
        .USE_RELU(0),
        .W0(8'sd1),
        .W1(8'sd0),
        .W2(8'sd0),
        .W3(8'sd0),
        .W4(8'sd0),
        .BIAS(32'sd0)
    ) output_layer (
        .clk(clk),
        .rst(rst),
        .sample_valid(layer1_valid),
        .sample_ready(output_layer_ready),
        .sample_in(layer1_out),
        .output_valid(output_valid),
        .sample_out(sample_out)
    );

endmodule
