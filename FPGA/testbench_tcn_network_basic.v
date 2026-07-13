`default_nettype none
`timescale 1ns/1ps

module testbench_tcn_network_basic;
    reg clk = 1'b0;
    reg rst = 1'b1;
    reg sample_valid = 1'b0;
    wire sample_ready;
    reg signed [7:0] sample_in = 8'sd0;
    wire output_valid;
    wire signed [7:0] sample_out;

    reg signed [7:0] inputs [0:5];
    reg signed [7:0] expected [0:5];
    integer input_index;
    integer output_index;

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
        #5 clk = ~clk;
    end

    always @(posedge clk) begin
        if (!rst && output_valid) begin
            if (output_index >= 6) begin
                $display("FAIL: unexpected extra output %0d", sample_out);
                $finish;
            end

            if (sample_out !== expected[output_index]) begin
                $display(
                    "FAIL: output[%0d] was %0d, expected %0d",
                    output_index,
                    sample_out,
                    expected[output_index]
                );
                $finish;
            end

            output_index <= output_index + 1;
        end
    end

    initial begin
        $dumpfile("test_data/tcn_network_basic.vcd");
        $dumpvars(0, testbench_tcn_network_basic);

        inputs[0] = 8'sd1;
        inputs[1] = 8'sd2;
        inputs[2] = -8'sd1;
        inputs[3] = 8'sd4;
        inputs[4] = 8'sd0;
        inputs[5] = 8'sd3;

        expected[0] = 8'sd2;
        expected[1] = 8'sd3;
        expected[2] = 8'sd2;
        expected[3] = 8'sd14;
        expected[4] = 8'sd0;
        expected[5] = 8'sd18;

        input_index = 0;
        output_index = 0;

        repeat (4) @(posedge clk);
        @(negedge clk);
        rst = 1'b0;

        for (input_index = 0; input_index < 6; input_index = input_index + 1) begin
            @(negedge clk);
            if (sample_ready !== 1'b1) begin
                $display("FAIL: sample_ready unexpectedly low for input[%0d]", input_index);
                $finish;
            end
            sample_in = inputs[input_index];
            sample_valid = 1'b1;
        end

        @(negedge clk);
        sample_valid = 1'b0;
        sample_in = 8'sd0;

        repeat (20) @(posedge clk);

        if (output_index !== 6) begin
            $display("FAIL: saw %0d outputs, expected 6", output_index);
            $finish;
        end

        $display("PASS: tcn_network_basic chained layer outputs verified");
        $finish;
    end
endmodule
