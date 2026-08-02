`default_nettype none
`timescale 1ns/1ps
module testbench_tcn_8ch_2layer_serial2;
    parameter integer NUM_SAMPLES = 100;
    reg clk = 0, rst = 1, sample_valid = 0;
    reg signed [63:0] sample_in = 0;
    wire sample_ready, output_valid;
    wire signed [63:0] sample_out;
    integer cycles = 0, accepted = 0, produced = 0;
    integer first_accept = -1, first_output = -1, last_output = 0, prev_accept = -1;
    tcn_network_8ch_2layer_serial2 dut (.clk(clk), .rst(rst), .sample_valid(sample_valid),
        .sample_ready(sample_ready), .sample_in(sample_in), .output_valid(output_valid), .sample_out(sample_out));
    always #20 clk = ~clk;
    always @(posedge clk) begin
        if (!rst) begin
            cycles <= cycles + 1;
            if (sample_valid && sample_ready) begin
                if (first_accept < 0) first_accept <= cycles;
                prev_accept <= cycles; accepted <= accepted + 1;
            end
            if (output_valid) begin
                if (first_output < 0) first_output <= cycles;
                last_output <= cycles; produced <= produced + 1;
            end
        end
    end
    always @(negedge clk) begin
        if (rst) begin sample_valid <= 0; sample_in <= 0; end
        else if (accepted < NUM_SAMPLES && sample_ready) begin
            sample_valid <= 1; sample_in <= {8{accepted[7:0]}};
        end else begin sample_valid <= 0; sample_in <= 0; end
    end
    initial begin
        repeat (5) @(posedge clk); @(negedge clk); rst = 0;
        while (produced < NUM_SAMPLES) @(posedge clk);
        $display("THROUGHPUT_REPORT");
        $display("channels=8 sample_width=8 weight_width=8 acc_width=8 kernel_taps=5 layers=2");
        $display("latency_cycles=%0d", first_output-first_accept);
        $display("steady_cycles_per_vector=%0f", (prev_accept-first_accept)*1.0/(accepted-1));
        $display("steady_vectors_per_sec_at_25mhz=%0f", 25000000.0/((prev_accept-first_accept)*1.0/(accepted-1)));
        $display("finite_cycles_per_vector=%0f", (last_output-first_accept+1)*1.0/produced);
        $finish;
    end
endmodule
