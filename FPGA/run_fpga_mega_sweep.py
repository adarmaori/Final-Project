#!/usr/bin/env python3
"""Run a resumable FPGA implementation/parameter sweep and write one CSV."""

import argparse
import csv
import itertools
import re
import subprocess
import sys
import tempfile
from pathlib import Path


DESIGNS = [
    ("conv1d_basic", "small"),
    ("conv1d_2mult", "small"),
    ("conv1d_parallel_pipelined", "small"),
    ("conv1d_serial", "small"),
    ("conv1d_large_parallel", "large"),
    ("conv1d_large_4mult", "large"),
    ("conv1d_large_serial", "large"),
]
CSV_FIELDS = [
    "target_device", "target_package", "target_freq_mhz", "status", "design", "architecture", "sample_width", "weight_width",
    "acc_width", "kernel_taps", "dilation", "latency_cycles",
    "steady_cycles_per_sample", "finite_cycles_per_sample",
    "steady_samples_per_sec_at_target_clock", "finite_samples_per_sec_at_target_clock",
    "max_clock_mhz", "max_theoretical_samples_per_sec", "total_cells",
    "icestorm_lc", "sb_lut4", "sb_carry", "sb_dff", "sb_dffe",
    "sb_dffsr", "sb_dffesr", "sb_dffess", "sb_mac16", "error",
]


def run(cmd, cwd, log_path):
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if proc.returncode:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def number(text, key, default=""):
    match = re.search(rf"^{re.escape(key)}=([^\n]+)", text, re.MULTILINE)
    return match.group(1).strip() if match else default


def cell_count(text, cell):
    match = re.search(rf"^\s*(?:Info:\s+)?{re.escape(cell)}\s*:?\s+(\d+)(?:/|\s*$)", text, re.MULTILINE)
    return match.group(1) if match else "0"


def max_clock(text):
    values = re.findall(r"Max frequency for clock.*?:\s*([0-9.]+) MHz", text)
    return values[-1] if values else "0"


def area(yosys_text, key):
    match = re.search(r"Number of cells:\s+(\d+)", yosys_text)
    return match.group(1) if match else "0"


def make_wrapper(design, kind, params):
    kernel_line = f"    .KERNEL_TAPS({params['kernel_taps']}),\n" if kind == "large" else ""
    return f'''`default_nettype none
module mega_top #(parameter integer SAMPLE_WIDTH={params['sample_width']}, parameter integer WEIGHT_WIDTH={params['weight_width']}, parameter integer ACC_WIDTH={params['acc_width']}) (
 input wire clk, input wire rst, input wire sample_valid, output wire sample_ready,
 input wire signed [SAMPLE_WIDTH-1:0] sample_in, output wire output_valid,
 output wire signed [ACC_WIDTH-1:0] sample_out);
 {design} #(.SAMPLE_WIDTH(SAMPLE_WIDTH), .WEIGHT_WIDTH(WEIGHT_WIDTH),
    .ACC_WIDTH(ACC_WIDTH), .DILATION({params['dilation']}),
{kernel_line}    .BIAS({params['acc_width']}'sd0)) impl (
    .clk(clk), .rst(rst), .sample_valid(sample_valid), .sample_ready(sample_ready),
    .sample_in(sample_in), .output_valid(output_valid), .sample_out(sample_out));
endmodule
'''


def make_tb(params, samples):
    return f'''`timescale 1ns/1ps
module mega_tb;
 localparam integer N={samples}; localparam integer SW={params['sample_width']};
 reg clk=0, rst=1, sample_valid=0; reg signed [SW-1:0] sample_in=0;
 wire sample_ready, output_valid; wire signed [{params['acc_width']}-1:0] sample_out;
 integer cycles=0, accepted=0, produced=0, first_accept=-1, first_output=-1;
 integer last_output=0, previous_accept=-1, max_gap=0, gap;
 real finite_cps, steady_cps, finite_sps, steady_sps;
 mega_top #(.SAMPLE_WIDTH({params['sample_width']}), .WEIGHT_WIDTH({params['weight_width']}), .ACC_WIDTH({params['acc_width']})) dut
  (.clk(clk), .rst(rst), .sample_valid(sample_valid), .sample_ready(sample_ready), .sample_in(sample_in), .output_valid(output_valid), .sample_out(sample_out));
 always #20 clk=~clk;
 always @(posedge clk) begin
  if (!rst) begin
   cycles<=cycles+1;
   if (sample_valid && sample_ready) begin
    if (first_accept<0) first_accept<=cycles;
    if (previous_accept>=0) begin gap=cycles-previous_accept; if (gap>max_gap) max_gap<=gap; end
    previous_accept<=cycles; accepted<=accepted+1;
   end
   if (output_valid) begin if (first_output<0) first_output<=cycles; last_output<=cycles; produced<=produced+1; end
  end
 end
 always @(negedge clk) begin
  if (rst) begin sample_valid<=0; sample_in<=0; end
  else if (accepted<N && sample_ready) begin sample_valid<=1; sample_in<=accepted[SW-1:0]; end
  else begin sample_valid<=0; sample_in<=0; end
 end
 initial begin
  repeat(5) @(posedge clk); @(negedge clk); rst=0;
  while(produced<N) @(posedge clk);
  finite_cps=(last_output-first_accept+1)*1.0/produced; finite_sps=25000000.0/finite_cps;
  steady_cps=(previous_accept-first_accept)*1.0/(accepted-1); steady_sps=25000000.0/steady_cps;
  $display("latency_cycles=%0d", first_output-first_accept); $display("steady_cycles_per_sample=%0f", steady_cps);
  $display("finite_cycles_per_sample=%0f", finite_cps); $display("steady_samples_per_sec_at_target_clock=%0f", steady_sps);
  $display("finite_samples_per_sec_at_target_clock=%0f", finite_sps); $finish;
 end
endmodule
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpga-dir", type=Path, default=Path("../../../FPGA"))
    parser.add_argument("--output", type=Path, default=Path("fpga_mega_sweep.csv"))
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--kernel-taps", default="8,16,32,48,64")
    parser.add_argument("--sample-widths", default="4,6,8")
    parser.add_argument("--weight-widths", default="4,6,8")
    parser.add_argument("--acc-widths", default="16,24,32")
    parser.add_argument("--dilations", default="1,2")
    parser.add_argument("--device", default="--hx1k")
    parser.add_argument("--package", default="vq100")
    args = parser.parse_args()
    fpga = args.fpga_dir.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    widths = lambda value: [int(x) for x in value.split(",")]
    kernels, sample_widths, weight_widths, acc_widths, dilations = map(
        widths, (args.kernel_taps, args.sample_widths, args.weight_widths, args.acc_widths, args.dilations)
    )
    existing = {}
    if output.exists():
        with output.open(newline="") as f:
            for row in csv.DictReader(f):
                existing[tuple(row[k] for k in ("design", "sample_width", "weight_width", "acc_width", "kernel_taps", "dilation"))] = row
    if not output.exists():
        with output.open("w", newline="") as f: csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    jobs = []
    for design, kind in DESIGNS:
        for sw, ww, aw, dilation in itertools.product(sample_widths, weight_widths, acc_widths, dilations):
            for kernel in ([5] if kind == "small" else kernels):
                jobs.append((design, kind, {"sample_width": sw, "weight_width": ww, "acc_width": aw, "kernel_taps": kernel, "dilation": dilation}))
    print(f"Planned configurations: {len(jobs)}")
    for index, (design, kind, params) in enumerate(jobs, 1):
        key = tuple(str(params[k]) for k in ("design",)) if False else (design, *(str(params[k]) for k in ("sample_width", "weight_width", "acc_width", "kernel_taps", "dilation")))
        if key in existing and existing[key].get("status") == "ok":
            print(f"[{index}/{len(jobs)}] cached {design} {params}")
            continue
        print(f"[{index}/{len(jobs)}] {design} {params}", flush=True)
        row = {k: "" for k in CSV_FIELDS} | {"target_device": args.device, "target_package": args.package, "target_freq_mhz": "25", "status": "failed", "design": design, "architecture": kind} | {k: str(v) for k, v in params.items()}
        try:
            with tempfile.TemporaryDirectory(prefix="fpga_mega_") as temp:
                temp = Path(temp)
                wrapper, tb = temp / "mega_wrapper.v", temp / "mega_tb.v"
                wrapper.write_text(make_wrapper(design, kind, params))
                tb.write_text(make_tb(params, args.samples))
                sim = temp / "sim.vvp"
                sim_log = temp / "sim.log"
                yosys_log = temp / "yosys.log"
                nextpnr_log = temp / "nextpnr.log"
                json_file, asc_file, sdf_file = (temp / x for x in ("design.json", "design.asc", "design.sdf"))
                run(["iverilog", "-Wall", "-g2012", "-o", str(sim), str(fpga / f"{design}.v"), str(wrapper), str(tb)], fpga, sim_log)
                sim_out = subprocess.check_output(["vvp", str(sim)], cwd=fpga, text=True, stderr=subprocess.STDOUT)
                sim_log.write_text(sim_out)
                run(["yosys", "-q", "-l", str(yosys_log), "-p", f'read_verilog "{fpga / f"{design}.v"}" "{wrapper}"; synth_ice40 -top mega_top -json "{json_file}"'], fpga, temp / "yosys_cmd.log")
                run(["nextpnr-ice40", "-q", "-l", str(nextpnr_log), args.device, "--package", args.package, "--freq", "25", "--json", str(json_file), "--asc", str(asc_file), "--sdf", str(sdf_file)], fpga, temp / "nextpnr_cmd.log")
                sim_text, yosys_text, nextpnr_text = sim_out, yosys_log.read_text(), nextpnr_log.read_text()
                for field in ("latency_cycles", "steady_cycles_per_sample", "finite_cycles_per_sample", "steady_samples_per_sec_at_target_clock", "finite_samples_per_sec_at_target_clock"):
                    row[field] = number(sim_text, field)
                row.update({"max_clock_mhz": max_clock(nextpnr_text), "total_cells": area(yosys_text, "Number of cells:")})
                for cell in ("SB_LUT4", "SB_CARRY", "SB_DFF", "SB_DFFE", "SB_DFFSR", "SB_DFFESR", "SB_DFFESS", "SB_MAC16"):
                    row[cell.lower()] = cell_count(yosys_text, cell)
                row["icestorm_lc"] = cell_count(nextpnr_text, "ICESTORM_LC:")
                row["max_theoretical_samples_per_sec"] = f"{float(row['max_clock_mhz']) * 1e6 / float(row['steady_cycles_per_sample']):.6f}"
                row["status"] = "ok"
        except Exception as exc:
            row["error"] = str(exc)
        with output.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
        existing[key] = row
    print(f"CSV written to {output}")


if __name__ == "__main__":
    main()
