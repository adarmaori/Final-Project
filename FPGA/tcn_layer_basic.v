`default_nettype none
`timescale 1ns/1ps

module tcn_layer_basic #(
    parameter integer SAMPLE_WIDTH = 8,
    parameter integer WEIGHT_WIDTH = 8,
    parameter integer ACC_WIDTH = 32,
    parameter integer DILATION = 1,
    parameter integer OUTPUT_SHIFT = 0,
    parameter integer USE_RELU = 1,
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
    output reg signed [SAMPLE_WIDTH-1:0] sample_out
);

    wire conv_ready;
    wire conv_valid;
    wire signed [ACC_WIDTH-1:0] conv_out;

    assign sample_ready = conv_ready;

    // Replaceable convolution primitive. Benchmark scripts generate this
    // adapter around whichever conv1d implementation is under test.
    conv1d_engine #(
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
    ) conv (
        .clk(clk),
        .rst(rst),
        .sample_valid(sample_valid),
        .sample_ready(conv_ready),
        .sample_in(sample_in),
        .output_valid(conv_valid),
        .sample_out(conv_out)
    );

    function signed [SAMPLE_WIDTH-1:0] saturate_to_sample;
        input signed [ACC_WIDTH-1:0] value;
        reg signed [ACC_WIDTH-1:0] max_value;
        reg signed [ACC_WIDTH-1:0] min_value;
        begin
            max_value = ({{(ACC_WIDTH-SAMPLE_WIDTH){1'b0}}, {1'b0, {(SAMPLE_WIDTH-1){1'b1}}}});
            min_value = -({{(ACC_WIDTH-SAMPLE_WIDTH){1'b0}}, {1'b1, {(SAMPLE_WIDTH-1){1'b0}}}});

            if (value > max_value) begin
                saturate_to_sample = {1'b0, {(SAMPLE_WIDTH-1){1'b1}}};
            end else if (value < min_value) begin
                saturate_to_sample = {1'b1, {(SAMPLE_WIDTH-1){1'b0}}};
            end else begin
                saturate_to_sample = value[SAMPLE_WIDTH-1:0];
            end
        end
    endfunction

    wire signed [ACC_WIDTH-1:0] shifted_out = conv_out >>> OUTPUT_SHIFT;
    wire signed [ACC_WIDTH-1:0] relu_out =
        (USE_RELU && shifted_out < 0) ? {ACC_WIDTH{1'b0}} : shifted_out;

    always @(posedge clk) begin
        if (rst) begin
            output_valid <= 1'b0;
            sample_out <= {SAMPLE_WIDTH{1'b0}};
        end else begin
            output_valid <= conv_valid;
            if (conv_valid) begin
                sample_out <= saturate_to_sample(relu_out);
            end
        end
    end

endmodule
