#!/Users/adarmaori/.cargo/bin/nu

def main [
  conv_file: string,             # Verilog file containing the conv1d implementation.
  conv_module?: string,          # Module name. Defaults to the file stem.
  --out-dir (-o): string = "engine_reports",
  --freq: int = 25,
  --device: string = "--hx1k",
  --package: string = "vq100",
  --samples: int = 1000,
] {
  let impl_module = if ($conv_module == null) {
    ($conv_file | path parse | get stem)
  } else {
    $conv_module
  }

  let impl_name = $impl_module
  let build_dir = ($out_dir | path join $impl_name)
  let wrapper_file = ($build_dir | path join "conv1d_engine.v")
  let sim_vvp = ($build_dir | path join "throughput.vvp")
  let sim_log = ($build_dir | path join "throughput.log")
  let yosys_log = ($build_dir | path join "yosys.log")
  let nextpnr_log = ($build_dir | path join "nextpnr.log")
  let json_file = ($build_dir | path join "network.json")
  let asc_file = ($build_dir | path join "network.asc")
  let sdf_file = ($build_dir | path join "network.sdf")
  let summary_file = ($build_dir | path join "summary.txt")

  mkdir $build_dir

  let wrapper = $"
`default_nettype none
`timescale 1ns/1ps

module conv1d_engine #\(
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
        .W0\(W0\),
        .W1\(W1\),
        .W2\(W2\),
        .W3\(W3\),
        .W4\(W4\),
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
    "tcn_layer_basic.v",
    "tcn_network_basic.v",
  ]

  let sim_sources = ($sources | append "testbench_tcn_network_throughput.v")

  print $"== ($impl_name): throughput simulation =="
  ^iverilog -Wall -g2012 -P $"testbench_tcn_network_throughput.NUM_SAMPLES=($samples)" -o $sim_vvp ...$sim_sources
  let sim_output = (^vvp $sim_vvp)
  print $sim_output
  $sim_output | save -f $sim_log

  print $"== ($impl_name): synthesis and place/route =="
  let read_verilog = ($sources | each {|src| $"\"($src)\"" } | str join " ")
  ^yosys -q -l $yosys_log -p $"read_verilog ($read_verilog); synth_ice40 -top tcn_network_basic -json ($json_file)"
  ^nextpnr-ice40 -q -l $nextpnr_log $device --package $package --freq $freq --json $json_file --asc $asc_file --sdf $sdf_file

  let sim_summary = (
    open $sim_log
    | lines
    | where {|line| (($line | str starts-with "THROUGHPUT_REPORT") or ($line | str starts-with "clock_hz=") or ($line | str starts-with "samples_") or ($line | str starts-with "latency_cycles=") or ($line | str starts-with "max_accept_gap_cycles=") or ($line | str starts-with "steady_") or ($line | str starts-with "finite_")) }
  )

  let area_summary = (
    open $yosys_log
    | lines
    | skip until {|line| $line =~ "=== tcn_network_basic ==="}
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
