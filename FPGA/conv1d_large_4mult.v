`default_nettype none
`timescale 1ns/1ps

module conv1d_large_4mult #(
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
    reg signed [SAMPLE_WIDTH-1:0] tap [0:KERNEL_TAPS-1];
    reg signed [ACC_WIDTH-1:0] acc;
    reg signed [ACC_WIDTH-1:0] group_sum;
    reg [15:0] tap_index;
    reg busy;
    integer i;

    assign sample_ready = !busy;

    function signed [WEIGHT_WIDTH-1:0] tap_weight;
        input integer weight_index;
        begin
            case (weight_index % 8)
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

    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < HISTORY_LEN; i = i + 1) begin
                history[i] <= {SAMPLE_WIDTH{1'b0}};
            end
            for (i = 0; i < KERNEL_TAPS; i = i + 1) begin
                tap[i] <= {SAMPLE_WIDTH{1'b0}};
            end
            acc <= {ACC_WIDTH{1'b0}};
            group_sum <= {ACC_WIDTH{1'b0}};
            tap_index <= 16'd0;
            busy <= 1'b0;
            output_valid <= 1'b0;
            sample_out <= {ACC_WIDTH{1'b0}};
        end else begin
            output_valid <= 1'b0;

            if (!busy && sample_valid) begin
                tap[0] <= sample_in;
                for (i = 1; i < KERNEL_TAPS; i = i + 1) begin
                    tap[i] <= history[(i * DILATION) - 1];
                end

                for (i = HISTORY_LEN - 1; i > 0; i = i - 1) begin
                    history[i] <= history[i - 1];
                end
                history[0] <= sample_in;

                acc <= BIAS;
                tap_index <= 16'd0;
                busy <= 1'b1;
            end else if (busy) begin
                group_sum = {ACC_WIDTH{1'b0}};
                for (i = 0; i < 4; i = i + 1) begin
                    if ((tap_index + i) < KERNEL_TAPS) begin
                        group_sum = group_sum + (tap[tap_index + i] * tap_weight(tap_index + i));
                    end
                end

                if ((tap_index + 4) >= KERNEL_TAPS) begin
                    sample_out <= acc + group_sum;
                    output_valid <= 1'b1;
                    busy <= 1'b0;
                    tap_index <= 16'd0;
                end else begin
                    acc <= acc + group_sum;
                    tap_index <= tap_index + 16'd4;
                end
            end
        end
    end

endmodule
