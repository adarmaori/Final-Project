`default_nettype none
`timescale 1ns/1ps

module conv1d_parallel_pipelined #(
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

    reg stage0_valid;
    reg signed [ACC_WIDTH-1:0] p0_r;
    reg signed [ACC_WIDTH-1:0] p1_r;
    reg signed [ACC_WIDTH-1:0] p2_r;
    reg signed [ACC_WIDTH-1:0] p3_r;
    reg signed [ACC_WIDTH-1:0] p4_r;

    reg stage1_valid;
    reg signed [ACC_WIDTH-1:0] sum01_r;
    reg signed [ACC_WIDTH-1:0] sum23_r;
    reg signed [ACC_WIDTH-1:0] p4_stage1_r;

    assign sample_ready = 1'b1;

    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < HISTORY_LEN; i = i + 1) begin
                history[i] <= {SAMPLE_WIDTH{1'b0}};
            end
            stage0_valid <= 1'b0;
            stage1_valid <= 1'b0;
            output_valid <= 1'b0;
            p0_r <= {ACC_WIDTH{1'b0}};
            p1_r <= {ACC_WIDTH{1'b0}};
            p2_r <= {ACC_WIDTH{1'b0}};
            p3_r <= {ACC_WIDTH{1'b0}};
            p4_r <= {ACC_WIDTH{1'b0}};
            sum01_r <= {ACC_WIDTH{1'b0}};
            sum23_r <= {ACC_WIDTH{1'b0}};
            p4_stage1_r <= {ACC_WIDTH{1'b0}};
            sample_out <= {ACC_WIDTH{1'b0}};
        end else begin
            stage0_valid <= sample_valid && sample_ready;
            stage1_valid <= stage0_valid;
            output_valid <= stage1_valid;

            if (sample_valid && sample_ready) begin
                p0_r <= sample_in * W0;
                p1_r <= history[DILATION-1] * W1;
                p2_r <= history[(2*DILATION)-1] * W2;
                p3_r <= history[(3*DILATION)-1] * W3;
                p4_r <= history[(4*DILATION)-1] * W4;

                for (i = HISTORY_LEN-1; i > 0; i = i - 1) begin
                    history[i] <= history[i-1];
                end
                history[0] <= sample_in;
            end

            if (stage0_valid) begin
                sum01_r <= p0_r + p1_r;
                sum23_r <= p2_r + p3_r;
                p4_stage1_r <= p4_r;
            end

            if (stage1_valid) begin
                sample_out <= BIAS + sum01_r + sum23_r + p4_stage1_r;
            end
        end
    end

endmodule
