
`default_nettype none
`timescale 1ns/1ps

module conv1d_engine #(
    parameter integer SAMPLE_WIDTH = 8,
    parameter integer WEIGHT_WIDTH = 8,
    parameter integer ACC_WIDTH = 32,
    parameter integer DILATION = 1,
    parameter signed [WEIGHT_WIDTH-1:0] W0 = 8'sd2,
    parameter signed [WEIGHT_WIDTH-1:0] W1 = -8'sd1,
    parameter signed [WEIGHT_WIDTH-1:0] W2 = 8'sd3,
    parameter signed [WEIGHT_WIDTH-1:0] W3 = 8'sd0,
    parameter signed [WEIGHT_WIDTH-1:0] W4 = 8'sd1,
    parameter signed [ACC_WIDTH-1:0] BIAS = 32'sd0
) (
    input wire clk,
    input wire rst,
    input wire sample_valid,
    output wire sample_ready,
    input wire signed [SAMPLE_WIDTH-1:0] sample_in,
    output wire output_valid,
    output wire signed [ACC_WIDTH-1:0] sample_out
);

    conv1d_2mult #(
        .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .DILATION(DILATION),
        .W0(W0),
        .W1(W1),
        .W2(W2),
        .W3(W3),
        .W4(W4),
        .BIAS(BIAS)
    ) impl (
        .clk(clk),
        .rst(rst),
        .sample_valid(sample_valid),
        .sample_ready(sample_ready),
        .sample_in(sample_in),
        .output_valid(output_valid),
        .sample_out(sample_out)
    );

endmodule
