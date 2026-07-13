`default_nettype none
`timescale 1ns/1ps

module conv1d_serial #(
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
    reg signed [SAMPLE_WIDTH-1:0] tap [0:4];
    reg signed [ACC_WIDTH-1:0] acc;
    reg [2:0] tap_index;
    reg busy;
    integer i;

    assign sample_ready = !busy;

    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < HISTORY_LEN; i = i + 1) begin
                history[i] <= {SAMPLE_WIDTH{1'b0}};
            end
            for (i = 0; i < 5; i = i + 1) begin
                tap[i] <= {SAMPLE_WIDTH{1'b0}};
            end
            acc <= {ACC_WIDTH{1'b0}};
            tap_index <= 3'd0;
            busy <= 1'b0;
            output_valid <= 1'b0;
            sample_out <= {ACC_WIDTH{1'b0}};
        end else begin
            output_valid <= 1'b0;

            if (!busy && sample_valid) begin
                tap[0] <= sample_in;
                tap[1] <= history[DILATION-1];
                tap[2] <= history[(2*DILATION)-1];
                tap[3] <= history[(3*DILATION)-1];
                tap[4] <= history[(4*DILATION)-1];

                acc <= BIAS + (sample_in * W0);
                tap_index <= 3'd1;
                busy <= 1'b1;

                for (i = HISTORY_LEN-1; i > 0; i = i - 1) begin
                    history[i] <= history[i-1];
                end
                history[0] <= sample_in;
            end else if (busy) begin
                case (tap_index)
                    3'd1: begin
                        acc <= acc + (tap[1] * W1);
                        tap_index <= 3'd2;
                    end
                    3'd2: begin
                        acc <= acc + (tap[2] * W2);
                        tap_index <= 3'd3;
                    end
                    3'd3: begin
                        acc <= acc + (tap[3] * W3);
                        tap_index <= 3'd4;
                    end
                    default: begin
                        sample_out <= acc + (tap[4] * W4);
                        output_valid <= 1'b1;
                        busy <= 1'b0;
                        tap_index <= 3'd0;
                    end
                endcase
            end
        end
    end

endmodule
