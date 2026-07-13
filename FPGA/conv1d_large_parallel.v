`default_nettype none
`timescale 1ns/1ps

module conv1d_large_parallel #(
    parameter integer SAMPLE_WIDTH = 8,
    parameter integer WEIGHT_WIDTH = 8,
    parameter integer ACC_WIDTH = 32,
    parameter integer DILATION = 1,
    parameter integer KERNEL_TAPS = 32,
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

    localparam integer HISTORY_LEN = (KERNEL_TAPS - 1) * DILATION;

    reg signed [SAMPLE_WIDTH-1:0] history [0:HISTORY_LEN-1];
    reg signed [ACC_WIDTH-1:0] conv_sum;
    integer i;

    assign sample_ready = 1'b1;

    function signed [WEIGHT_WIDTH-1:0] tap_weight;
        input integer tap_index;
        begin
            case (tap_index % 8)
                0: tap_weight = 8'sd2;
                1: tap_weight = -8'sd1;
                2: tap_weight = 8'sd3;
                3: tap_weight = 8'sd0;
                4: tap_weight = 8'sd1;
                5: tap_weight = -8'sd2;
                6: tap_weight = 8'sd1;
                default: tap_weight = 8'sd2;
            endcase
        end
    endfunction

    always @* begin
        conv_sum = BIAS + (sample_in * tap_weight(0));
        for (i = 1; i < KERNEL_TAPS; i = i + 1) begin
            conv_sum = conv_sum + (history[(i * DILATION) - 1] * tap_weight(i));
        end
    end

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
                for (i = HISTORY_LEN - 1; i > 0; i = i - 1) begin
                    history[i] <= history[i - 1];
                end
                history[0] <= sample_in;
            end
        end
    end

endmodule
