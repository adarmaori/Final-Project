`default_nettype none
`timescale 1ns/1ps

// Fully-connected 8-channel causal convolution.  Every output channel
// consumes every input channel over KERNEL_TAPS samples.
module tcn_8ch_layer_parallel #(
    parameter integer CHANNELS = 8,
    parameter integer SAMPLE_WIDTH = 8,
    parameter integer WEIGHT_WIDTH = 8,
    parameter integer ACC_WIDTH = 16,
    parameter integer KERNEL_TAPS = 5,
    parameter integer DILATION = 1,
    parameter integer USE_RELU = 1
) (
    input wire clk,
    input wire rst,
    input wire sample_valid,
    output wire sample_ready,
    input wire signed [CHANNELS*SAMPLE_WIDTH-1:0] sample_in,
    output reg output_valid,
    output reg signed [CHANNELS*SAMPLE_WIDTH-1:0] sample_out
);
    localparam integer HISTORY_LEN = (KERNEL_TAPS - 1) * DILATION;

    reg signed [SAMPLE_WIDTH-1:0] history [0:CHANNELS-1][0:HISTORY_LEN-1];
    reg signed [ACC_WIDTH-1:0] conv_sum [0:CHANNELS-1];
    integer o;
    integer c;
    integer t;
    integer h;

    assign sample_ready = 1'b1;

    function signed [WEIGHT_WIDTH-1:0] weight_value;
        input integer output_channel;
        input integer input_channel;
        input integer tap_index;
        begin
            // Deterministic nonzero weights for synthesis and simulation.
            case ((output_channel + 3*input_channel + tap_index) % 8)
                0: weight_value = 8'sd2;
                1: weight_value = -8'sd1;
                2: weight_value = 8'sd3;
                3: weight_value = 8'sd0;
                4: weight_value = 8'sd1;
                5: weight_value = -8'sd2;
                6: weight_value = 8'sd1;
                default: weight_value = 8'sd2;
            endcase
        end
    endfunction

    always @* begin
        for (o = 0; o < CHANNELS; o = o + 1) begin
            conv_sum[o] = {ACC_WIDTH{1'b0}};
            for (c = 0; c < CHANNELS; c = c + 1) begin
                conv_sum[o] = conv_sum[o] +
                    ($signed(sample_in[c*SAMPLE_WIDTH +: SAMPLE_WIDTH]) *
                     weight_value(o, c, 0));
                for (t = 1; t < KERNEL_TAPS; t = t + 1) begin
                    conv_sum[o] = conv_sum[o] +
                        (history[c][(t*DILATION)-1] * weight_value(o, c, t));
                end
            end
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            output_valid <= 1'b0;
            sample_out <= {(CHANNELS*SAMPLE_WIDTH){1'b0}};
            for (c = 0; c < CHANNELS; c = c + 1)
                for (h = 0; h < HISTORY_LEN; h = h + 1)
                    history[c][h] <= {SAMPLE_WIDTH{1'b0}};
        end else begin
            output_valid <= sample_valid && sample_ready;
            if (sample_valid && sample_ready) begin
                for (o = 0; o < CHANNELS; o = o + 1) begin
                    if (USE_RELU && conv_sum[o] < 0)
                        sample_out[o*SAMPLE_WIDTH +: SAMPLE_WIDTH] <= {SAMPLE_WIDTH{1'b0}};
                    else
                        sample_out[o*SAMPLE_WIDTH +: SAMPLE_WIDTH] <= conv_sum[o][SAMPLE_WIDTH-1:0];
                end
                for (c = 0; c < CHANNELS; c = c + 1) begin
                    for (h = HISTORY_LEN-1; h > 0; h = h - 1)
                        history[c][h] <= history[c][h-1];
                    history[c][0] <= sample_in[c*SAMPLE_WIDTH +: SAMPLE_WIDTH];
                end
            end
        end
    end
endmodule
