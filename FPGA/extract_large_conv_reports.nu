#!/Users/adarmaori/.cargo/bin/nu

def main [
  conv_file: string,
  conv_module?: string,
  --out-dir (-o): string = "large_conv_reports",
  --freq: int = 25,
  --device: string = "--hx1k",
  --package: string = "vq100",
  --samples: int = 1000,
  --sample-width: int = 8,
  --weight-width: int = 8,
  --acc-width: int = 32,
  --kernel-taps: int = 32,
] {
  let impl_module = if ($conv_module == null) {
    ($conv_file | path parse | get stem)
  } else {
    $conv_module
  }

  let impl_name = $impl_module
  let quant_name = $"s($sample_width)_w($weight_width)_a($acc_width)"
  let build_dir = ($out_dir | path join $"($impl_name)_k($kernel_taps)_($quant_name)")
  let wrapper_file = ($build_dir | path join "conv1d_large_engine.v")
  let sim_vvp = ($build_dir | path join "throughput.vvp")
  let sim_log = ($build_dir | path join "throughput.log")
  let yosys_log = ($build_dir | path join "yosys.log")
  let nextpnr_log = ($build_dir | path join "nextpnr.log")
  let json_file = ($build_dir | path join "conv.json")
  let asc_file = ($build_dir | path join "conv.asc")
  let sdf_file = ($build_dir | path join "conv.sdf")
  let summary_file = ($build_dir | path join "summary.txt")

  mkdir $build_dir

  let wrapper = $"
`default_nettype none
`timescale 1ns/1ps

module conv1d_large_engine #\(
    parameter integer SAMPLE_WIDTH = 8,
    parameter integer WEIGHT_WIDTH = 8,
    parameter integer ACC_WIDTH = 32,
    parameter integer DILATION = 1,
    parameter integer KERNEL_TAPS = 32,
    parameter signed [ACC_WIDTH-1:0] BIAS = 32'sd0
\) \(
    input wire clk,
    input wire rst,
    input wire sample_valid,
    output wire sample_ready,
    input wire signed [SAMPLE_WIDTH-1:0] sample_in,
    output wire output_valid,
    output wire signed [ACC_WIDTH-1:0] sample_out
\);

    ($impl_module) #\(
        .SAMPLE_WIDTH\(SAMPLE_WIDTH\),
        .WEIGHT_WIDTH\(WEIGHT_WIDTH\),
        .ACC_WIDTH\(ACC_WIDTH\),
        .DILATION\(DILATION\),
        .KERNEL_TAPS\(KERNEL_TAPS\),
        .BIAS\(BIAS\)
    \) impl \(
        .clk\(clk\),
        .rst\(rst\),
        .sample_valid\(sample_valid\),
        .sample_ready\(sample_ready\),
        .sample_in\(sample_in\),
        .output_valid\(output_valid\),
        .sample_out\(sample_out\)
    \);

endmodule
"
  $wrapper | save -f $wrapper_file

  let sources = [
    $conv_file,
    $wrapper_file,
    "conv1d_large_benchmark_top.v",
  ]
  let sim_sources = ($sources | append "testbench_conv1d_large_throughput.v")

  print $"== ($impl_name), K=($kernel_taps), ($quant_name): throughput simulation =="
  ^iverilog -Wall -g2012 -P $"testbench_conv1d_large_throughput.NUM_SAMPLES=($samples)" -P $"testbench_conv1d_large_throughput.SAMPLE_WIDTH=($sample_width)" -P $"testbench_conv1d_large_throughput.WEIGHT_WIDTH=($weight_width)" -P $"testbench_conv1d_large_throughput.ACC_WIDTH=($acc_width)" -P $"testbench_conv1d_large_throughput.KERNEL_TAPS=($kernel_taps)" -o $sim_vvp ...$sim_sources
  let sim_output = (^vvp $sim_vvp)
  print $sim_output
  $sim_output | save -f $sim_log

  print $"== ($impl_name), K=($kernel_taps), ($quant_name): synthesis and place/route =="
  let read_verilog = ($sources | each {|src| $"\"($src)\"" } | str join " ")
  ^yosys -q -l $yosys_log -p $"read_verilog ($read_verilog); chparam -set SAMPLE_WIDTH ($sample_width) conv1d_large_benchmark_top; chparam -set WEIGHT_WIDTH ($weight_width) conv1d_large_benchmark_top; chparam -set ACC_WIDTH ($acc_width) conv1d_large_benchmark_top; chparam -set KERNEL_TAPS ($kernel_taps) conv1d_large_benchmark_top; synth_ice40 -top conv1d_large_benchmark_top -json ($json_file)"
  ^nextpnr-ice40 -q -l $nextpnr_log $device --package $package --freq $freq --json $json_file --asc $asc_file --sdf $sdf_file

  let sim_summary = (
    open $sim_log
    | lines
    | where {|line| (($line | str starts-with "THROUGHPUT_REPORT") or ($line | str starts-with "clock_hz=") or ($line | str starts-with "sample_width=") or ($line | str starts-with "weight_width=") or ($line | str starts-with "acc_width=") or ($line | str starts-with "kernel_taps=") or ($line | str starts-with "samples_") or ($line | str starts-with "latency_cycles=") or ($line | str starts-with "max_accept_gap_cycles=") or ($line | str starts-with "steady_") or ($line | str starts-with "finite_")) }
  )

  let area_summary = (
    open $yosys_log
    | lines
    | skip until {|line| $line =~ "=== conv1d_large_benchmark_top ==="}
    | skip 1
    | take until {|line| $line =~ "^==="}
    | where {|line| (($line =~ "Number of cells:") or ($line =~ "SB_LUT4") or ($line =~ "SB_DFF") or ($line =~ "SB_DFFE") or ($line =~ "SB_DFFESR") or ($line =~ "SB_DFFESS") or ($line =~ "SB_DFFSR") or ($line =~ "SB_CARRY") or ($line =~ "SB_MAC16")) }
  )

  let timing_summary = (
    open $nextpnr_log
    | lines
    | where {|line| (($line =~ "Device utilisation:") or (($line | str starts-with "Info: \t") and (($line =~ "ICESTORM_LC:") or ($line =~ "ICESTORM_RAM:") or ($line =~ "ICESTORM_DSP:") or ($line =~ "SB_IO:") or ($line =~ "SB_GB:"))) or ($line =~ "Max frequency for clock") or ($line =~ "Critical path report for clock")) }
  )

  let summary_lines = (
    [
      $"Implementation: ($impl_name)",
      $"Conv file: ($conv_file)",
      $"Kernel taps: ($kernel_taps)",
      $"Quantization: sample_width=($sample_width) weight_width=($weight_width) acc_width=($acc_width)",
      $"Target: ($device) package=($package) freq=($freq)MHz",
      "",
      "Throughput simulation:",
    ]
    | append $sim_summary
    | append ["" "Yosys area summary:"]
    | append $area_summary
    | append ["" "nextpnr timing/utilization summary:"]
    | append $timing_summary
  )

  $summary_lines | str join "\n" | save -f $summary_file

  print ""
  print $"== Summary written to ($summary_file) =="
  open $summary_file
}
