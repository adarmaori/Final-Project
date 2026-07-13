`default_nettype none
`timescale 1ns/1ps

module conv1d_basic #(
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
    output reg output_valid,
    output reg signed [ACC_WIDTH-1:0] sample_out
);

    localparam integer HISTORY_LEN = 4 * DILATION;

    reg signed [SAMPLE_WIDTH-1:0] history [0:HISTORY_LEN-1];
    integer i;

    assign sample_ready = 1'b1;

    wire signed [ACC_WIDTH-1:0] x0 = sample_in;
    wire signed [ACC_WIDTH-1:0] x1 = history[DILATION-1];
    wire signed [ACC_WIDTH-1:0] x2 = history[(2*DILATION)-1];
    wire signed [ACC_WIDTH-1:0] x3 = history[(3*DILATION)-1];
    wire signed [ACC_WIDTH-1:0] x4 = history[(4*DILATION)-1];

    wire signed [ACC_WIDTH-1:0] p0 = x0 * W0;
    wire signed [ACC_WIDTH-1:0] p1 = x1 * W1;
    wire signed [ACC_WIDTH-1:0] p2 = x2 * W2;
    wire signed [ACC_WIDTH-1:0] p3 = x3 * W3;
    wire signed [ACC_WIDTH-1:0] p4 = x4 * W4;
    wire signed [ACC_WIDTH-1:0] conv_sum = BIAS + p0 + p1 + p2 + p3 + p4;

    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < HISTORY_LEN; i = i + 1) begin
                history[i] <= {SAMPLE_WIDTH{1'b0}};
            end
            output_valid <= 1'b0;
            sample_out <= {ACC_WIDTH{1'b0}};
        end else begin
            output_valid <= sample_valid && sample_ready;

            if (sample_valid && sample_ready) begin
                sample_out <= conv_sum;
                for (i = HISTORY_LEN-1; i > 0; i = i - 1) begin
                    history[i] <= history[i-1];
                end
                history[0] <= sample_in;
            end
        end
    end

endmodule
