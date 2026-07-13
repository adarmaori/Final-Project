`default_nettype none
`timescale 1ns/1ps

module conv1d_large_benchmark_top #(
    parameter integer SAMPLE_WIDTH = 8,
    parameter integer WEIGHT_WIDTH = 8,
    parameter integer ACC_WIDTH = 32,
    parameter integer DILATION = 1,
    parameter integer KERNEL_TAPS = 32
) (
    input wire clk,
    input wire rst,
    input wire sample_valid,
    output wire sample_ready,
    input wire signed [SAMPLE_WIDTH-1:0] sample_in,
    output wire output_valid,
    output wire signed [ACC_WIDTH-1:0] sample_out
);

    conv1d_large_engine #(
        .SAMPLE_WIDTH(SAMPLE_WIDTH),
        .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .DILATION(DILATION),
        .KERNEL_TAPS(KERNEL_TAPS),
        .BIAS({ACC_WIDTH{1'b0}})
    ) conv (
        .clk(clk),
        .rst(rst),
        .sample_valid(sample_valid),
        .sample_ready(sample_ready),
        .sample_in(sample_in),
        .output_valid(output_valid),
        .sample_out(sample_out)
    );

endmodule
