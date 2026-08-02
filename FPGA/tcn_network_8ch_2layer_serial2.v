`default_nettype none
`timescale 1ns/1ps

// Area-oriented full 8->8, K=5 convolution.
// Two multipliers are reused across all 8 output channels, 8 input channels,
// and 5 taps.  One input vector is accepted only after both layers finish.
module tcn_network_8ch_2layer_serial2 #(
    parameter integer CHANNELS = 8,
    parameter integer SAMPLE_WIDTH = 8,
    parameter integer WEIGHT_WIDTH = 8,
    parameter integer ACC_WIDTH = 8,
    parameter integer KERNEL_TAPS = 5
) (
    input wire clk,
    input wire rst,
    input wire sample_valid,
    output wire sample_ready,
    input wire signed [CHANNELS*SAMPLE_WIDTH-1:0] sample_in,
    output reg output_valid,
    output reg signed [CHANNELS*SAMPLE_WIDTH-1:0] sample_out
);
    localparam integer MACS_PER_OUTPUT = CHANNELS * KERNEL_TAPS;
    localparam integer MAC_INDEX_WIDTH = 8;

    localparam [3:0] IDLE = 4'd0;
    localparam [3:0] CALC0 = 4'd1;
    localparam [3:0] START1 = 4'd2;
    localparam [3:0] CALC1 = 4'd3;

    reg [3:0] state;
    reg [3:0] output_channel;
    reg [MAC_INDEX_WIDTH-1:0] mac_index;
    reg signed [ACC_WIDTH-1:0] acc;
    reg signed [SAMPLE_WIDTH-1:0] current_input [0:CHANNELS-1];
    reg signed [SAMPLE_WIDTH-1:0] layer0_value [0:CHANNELS-1];
    reg signed [SAMPLE_WIDTH-1:0] history0 [0:CHANNELS-1][0:KERNEL_TAPS-2];
    reg signed [SAMPLE_WIDTH-1:0] history1 [0:CHANNELS-1][0:KERNEL_TAPS-2];

    reg signed [ACC_WIDTH-1:0] mac_value0;
    reg signed [ACC_WIDTH-1:0] mac_value1;
    reg signed [ACC_WIDTH-1:0] acc_with_macs;
    integer c;
    integer t;
    integer h;
    integer idx0;
    integer idx1;

    assign sample_ready = (state == IDLE);

    function signed [WEIGHT_WIDTH-1:0] weight_value;
        input integer layer;
        input integer output_channel_arg;
        input integer input_channel_arg;
        input integer tap_arg;
        begin
            case ((layer + output_channel_arg + 3*input_channel_arg + tap_arg) % 8)
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

    function signed [SAMPLE_WIDTH-1:0] quantize;
        input signed [ACC_WIDTH-1:0] value;
        begin
            quantize = value[SAMPLE_WIDTH-1:0];
        end
    endfunction

    // Two MAC operands are selected from the current layer's history.
    always @* begin
        idx0 = mac_index;
        idx1 = mac_index + 1;
        mac_value0 = {ACC_WIDTH{1'b0}};
        mac_value1 = {ACC_WIDTH{1'b0}};

        if (state == CALC0) begin
            if ((idx0 % KERNEL_TAPS) == 0)
                mac_value0 = $signed(current_input[idx0 / KERNEL_TAPS]) * weight_value(0, output_channel, idx0 / KERNEL_TAPS, idx0 % KERNEL_TAPS);
            else if ((idx0 / KERNEL_TAPS) < CHANNELS)
                mac_value0 = $signed(history0[(idx0 / KERNEL_TAPS)][(idx0 % KERNEL_TAPS)-1]) * weight_value(0, output_channel, idx0 / KERNEL_TAPS, idx0 % KERNEL_TAPS);
            if (idx1 < MACS_PER_OUTPUT) begin
                if ((idx1 % KERNEL_TAPS) == 0)
                    mac_value1 = $signed(current_input[idx1 / KERNEL_TAPS]) * weight_value(0, output_channel, idx1 / KERNEL_TAPS, idx1 % KERNEL_TAPS);
                else if ((idx1 / KERNEL_TAPS) < CHANNELS)
                    mac_value1 = $signed(history0[(idx1 / KERNEL_TAPS)][(idx1 % KERNEL_TAPS)-1]) * weight_value(0, output_channel, idx1 / KERNEL_TAPS, idx1 % KERNEL_TAPS);
            end
        end else if (state == CALC1) begin
            if ((idx0 % KERNEL_TAPS) == 0)
                mac_value0 = $signed(layer0_value[idx0 / KERNEL_TAPS]) * weight_value(1, output_channel, idx0 / KERNEL_TAPS, idx0 % KERNEL_TAPS);
            else if ((idx0 / KERNEL_TAPS) < CHANNELS)
                mac_value0 = $signed(history1[(idx0 / KERNEL_TAPS)][(idx0 % KERNEL_TAPS)-1]) * weight_value(1, output_channel, idx0 / KERNEL_TAPS, idx0 % KERNEL_TAPS);
            if (idx1 < MACS_PER_OUTPUT) begin
                if ((idx1 % KERNEL_TAPS) == 0)
                    mac_value1 = $signed(layer0_value[idx1 / KERNEL_TAPS]) * weight_value(1, output_channel, idx1 / KERNEL_TAPS, idx1 % KERNEL_TAPS);
                else if ((idx1 / KERNEL_TAPS) < CHANNELS)
                    mac_value1 = $signed(history1[(idx1 / KERNEL_TAPS)][(idx1 % KERNEL_TAPS)-1]) * weight_value(1, output_channel, idx1 / KERNEL_TAPS, idx1 % KERNEL_TAPS);
            end
        end
        acc_with_macs = acc + mac_value0 + mac_value1;
    end

    always @(posedge clk) begin
        if (rst) begin
            state <= IDLE;
            output_channel <= 0;
            mac_index <= 0;
            acc <= 0;
            output_valid <= 1'b0;
            sample_out <= 0;
            for (c = 0; c < CHANNELS; c = c + 1) begin
                current_input[c] <= 0;
                layer0_value[c] <= 0;
                for (h = 0; h < KERNEL_TAPS-1; h = h + 1) begin
                    history0[c][h] <= 0;
                    history1[c][h] <= 0;
                end
            end
        end else begin
            output_valid <= 1'b0;
            case (state)
                IDLE: begin
                    if (sample_valid) begin
                        for (c = 0; c < CHANNELS; c = c + 1)
                            current_input[c] <= sample_in[c*SAMPLE_WIDTH +: SAMPLE_WIDTH];
                        for (c = 0; c < CHANNELS; c = c + 1) begin
                            for (h = KERNEL_TAPS-2; h > 0; h = h - 1)
                                history0[c][h] <= history0[c][h-1];
                            history0[c][0] <= sample_in[c*SAMPLE_WIDTH +: SAMPLE_WIDTH];
                        end
                        output_channel <= 0;
                        mac_index <= 0;
                        acc <= 0;
                        state <= CALC0;
                    end
                end
                CALC0: begin
                    if (mac_index + 2 >= MACS_PER_OUTPUT) begin
                        layer0_value[output_channel] <= quantize(acc_with_macs);
                        if (output_channel == CHANNELS-1) begin
                            state <= START1;
                            output_channel <= 0;
                            mac_index <= 0;
                            acc <= 0;
                        end else begin
                            output_channel <= output_channel + 1'b1;
                            mac_index <= 0;
                            acc <= 0;
                        end
                    end else begin
                        mac_index <= mac_index + 2;
                        acc <= acc_with_macs;
                    end
                end
                START1: begin
                    for (c = 0; c < CHANNELS; c = c + 1) begin
                        for (h = KERNEL_TAPS-2; h > 0; h = h - 1)
                            history1[c][h] <= history1[c][h-1];
                        history1[c][0] <= layer0_value[c];
                    end
                    state <= CALC1;
                end
                CALC1: begin
                    if (mac_index + 2 >= MACS_PER_OUTPUT) begin
                        sample_out[output_channel*SAMPLE_WIDTH +: SAMPLE_WIDTH] <= quantize(acc_with_macs);
                        if (output_channel == CHANNELS-1) begin
                            state <= IDLE;
                            output_valid <= 1'b1;
                        end else begin
                            output_channel <= output_channel + 1'b1;
                            mac_index <= 0;
                            acc <= 0;
                        end
                    end else begin
                        mac_index <= mac_index + 2;
                        acc <= acc_with_macs;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
