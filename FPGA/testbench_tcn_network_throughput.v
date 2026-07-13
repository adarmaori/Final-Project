`default_nettype none
`timescale 1ns/1ps
module testbench_tcn_network_throughput;
    parameter integer NUM_SAMPLES = 1000;
    parameter real CLOCK_HZ = 25000000.0;
    reg clk = 1'b0;
    reg rst = 1'b1;
    reg sample_valid = 1'b0;
    wire sample_ready;
    reg signed [7:0] sample_in = 8'sd0;
    wire output_valid;
    wire signed [7:0] sample_out;
    integer cycles = 0;
    integer accepted = 0;
    integer produced = 0;
    integer first_accept_cycle = -1;
    integer first_output_cycle = -1;
    integer last_output_cycle = 0;
    integer cycle_at_prev_accept = -1;
    integer max_accept_gap = 0;
    integer accept_gap = 0;
    real finite_cycles_per_sample;
    real steady_cycles_per_sample;
    real finite_samples_per_sec;
    real steady_samples_per_sec;

    tcn_network_basic #(
        .SAMPLE_WIDTH(8),
        .WEIGHT_WIDTH(8),
        .ACC_WIDTH(32),
        .OUTPUT_SHIFT(0)
    ) dut (
        .clk(clk),
        .rst(rst),
        .sample_valid(sample_valid),
        .sample_ready(sample_ready),
        .sample_in(sample_in),
        .output_valid(output_valid),
        .sample_out(sample_out)
    );
    always begin
        #20 clk = ~clk;
    end
    always @(posedge clk) begin
        if (!rst) begin
            cycles <= cycles + 1;
            if (sample_valid && sample_ready) begin
                if (first_accept_cycle < 0) begin
                    first_accept_cycle <= cycles;
                end
                if (cycle_at_prev_accept >= 0) begin
                    accept_gap = cycles - cycle_at_prev_accept;
                    if (accept_gap > max_accept_gap) begin
                        max_accept_gap <= accept_gap;
                    end
                end
                cycle_at_prev_accept <= cycles;
                accepted <= accepted + 1;
            end
            if (output_valid) begin
                if (first_output_cycle < 0) begin
                    first_output_cycle <= cycles;
                end
                last_output_cycle <= cycles;
                produced <= produced + 1;
            end
        end
    end
    always @(negedge clk) begin
        if (rst) begin
            sample_valid <= 1'b0;
            sample_in <= 8'sd0;
        end else if (accepted < NUM_SAMPLES && sample_ready) begin
            sample_valid <= 1'b1;
            sample_in <= accepted[7:0];
        end else begin
            sample_valid <= 1'b0;
            sample_in <= 8'sd0;
        end
    end

    initial begin
        $dumpfile("test_data/tcn_network_throughput.vcd");
        $dumpvars(0, testbench_tcn_network_throughput);
        repeat (5) @(posedge clk);
        @(negedge clk);
        rst = 1'b0;

        while (produced < NUM_SAMPLES) begin
            @(posedge clk);
        end

        finite_cycles_per_sample = (last_output_cycle - first_accept_cycle + 1) * 1.0 / produced;
        finite_samples_per_sec = CLOCK_HZ / finite_cycles_per_sample;

        if (accepted > 1) begin
            steady_cycles_per_sample = (cycle_at_prev_accept - first_accept_cycle) * 1.0 / (accepted - 1);
            steady_samples_per_sec = CLOCK_HZ / steady_cycles_per_sample;
        end else begin
            steady_cycles_per_sample = 0.0;
            steady_samples_per_sec = 0.0;
        end

        $display("THROUGHPUT_REPORT");
        $display("clock_hz=%0f", CLOCK_HZ);
        $display("samples_requested=%0d", NUM_SAMPLES);
        $display("samples_accepted=%0d", accepted);
        $display("samples_produced=%0d", produced);
        $display("first_accept_cycle=%0d", first_accept_cycle);
        $display("first_output_cycle=%0d", first_output_cycle);
        $display("last_output_cycle=%0d", last_output_cycle);
        $display("latency_cycles=%0d", first_output_cycle - first_accept_cycle);
        $display("max_accept_gap_cycles=%0d", max_accept_gap);
        $display("steady_cycles_per_sample=%0f", steady_cycles_per_sample);
        $display("steady_samples_per_sec=%0f", steady_samples_per_sec);
        $display("finite_cycles_per_sample=%0f", finite_cycles_per_sample);
        $display("finite_samples_per_sec=%0f", finite_samples_per_sec);
        $finish;
    end
endmodule
