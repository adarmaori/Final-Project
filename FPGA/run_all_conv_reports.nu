#!/Users/adarmaori/.cargo/bin/nu

def get-kv [lines: list<string>, key: string] {
  let prefix = $"($key)="
  let matches = ($lines | where {|line| $line | str starts-with $prefix})
  if (($matches | length) == 0) {
    null
  } else {
    (($matches | first) | split row "=" | get 1)
  }
}

def get-label [lines: list<string>, key: string] {
  let prefix = $"($key):"
  let matches = ($lines | where {|line| $line | str starts-with $prefix})
  if (($matches | length) == 0) {
    null
  } else {
    (($matches | first) | split row ":" | skip 1 | str join ":" | str trim)
  }
}

def get-cell [lines: list<string>, cell_name: string] {
  let matches = (
    $lines
    | where {|line|
        let fields = ($line | str trim | split row -r '\s+')
        (($fields | length) >= 2) and (($fields | get 0) == $cell_name)
      }
  )
  if (($matches | length) == 0) {
    0
  } else {
    ($matches | first | str trim | split row -r '\s+' | get 1 | into int)
  }
}

def get-total-cells [lines: list<string>] {
  let matches = ($lines | where {|line| $line =~ "Number of cells:"})
  if (($matches | length) == 0) {
    0
  } else {
    ($matches | first | parse -r 'Number of cells:\s+(?P<count>\d+)' | get count.0 | into int)
  }
}

def get-lc-used [lines: list<string>] {
  let matches = ($lines | where {|line| $line =~ 'ICESTORM_LC:'})
  if (($matches | length) == 0) {
    0
  } else {
    ($matches | first | parse -r 'ICESTORM_LC:\s+(?P<used>\d+)/' | get used.0 | into int)
  }
}

def get-max-frequency [lines: list<string>] {
  let matches = ($lines | where {|line| $line =~ "Max frequency for clock"})
  if (($matches | length) == 0) {
    0.0
  } else {
    ($matches | last | parse -r 'Max frequency.*: (?P<freq>[0-9.]+) MHz' | get freq.0 | into float)
  }
}

def summarize-report [summary_path: string] {
  let lines = (open $summary_path | lines)
  let max_freq_mhz = (get-max-frequency $lines)
  let steady_cycles = (get-kv $lines "steady_cycles_per_sample" | into float)
  let max_theoretical_samples_per_sec = (($max_freq_mhz * 1000000.0) / $steady_cycles)

  {
    implementation: (get-label $lines "Implementation"),
    latency_cycles: (get-kv $lines "latency_cycles" | into int),
    steady_cycles_per_sample: $steady_cycles,
    finite_cycles_per_sample: (get-kv $lines "finite_cycles_per_sample" | into float),
    steady_samples_per_sec_at_25mhz: (get-kv $lines "steady_samples_per_sec" | into float),
    finite_samples_per_sec_at_25mhz: (get-kv $lines "finite_samples_per_sec" | into float),
    max_clock_mhz: $max_freq_mhz,
    max_theoretical_samples_per_sec: $max_theoretical_samples_per_sec,
    total_cells: (get-total-cells $lines),
    icestorm_lc: (get-lc-used $lines),
    sb_lut4: (get-cell $lines "SB_LUT4"),
    sb_carry: (get-cell $lines "SB_CARRY"),
    sb_dff: (get-cell $lines "SB_DFF"),
    sb_dffe: (get-cell $lines "SB_DFFE"),
    sb_dffsr: (get-cell $lines "SB_DFFSR"),
    sb_dffesr: (get-cell $lines "SB_DFFESR"),
    sb_dffess: (get-cell $lines "SB_DFFESS"),
    sb_mac16: (get-cell $lines "SB_MAC16"),
  }
}

def main [
  --pattern: string = "conv1d_*.v",
  --out-dir (-o): string = "engine_reports",
  --csv: string = "engine_reports/summary.csv",
  --samples: int = 1000,
  --freq: int = 25,
  --device: string = "--hx1k",
  --package: string = "vq100",
] {
  let conv_files = (
    glob $pattern
    | where {|path| ($path | path basename) != "conv1d_engine.v"}
    | sort
  )

  if (($conv_files | length) == 0) {
    error make {msg: $"No convolution files matched pattern: ($pattern)"}
  }

  mut rows = []

  for conv_file in $conv_files {
    let module_name = ($conv_file | path parse | get stem)
    print $"==== Benchmarking ($module_name) from ($conv_file) ===="

    ^nu extract_all_reports.nu $conv_file $module_name --out-dir $out_dir --samples $samples --freq $freq --device $device --package $package

    let summary_path = ($out_dir | path join $module_name | path join "summary.txt")
    let row = (summarize-report $summary_path)
    $rows = ($rows | append $row)
  }

  mkdir ($csv | path dirname)
  $rows | to csv | save -f $csv

  print ""
  print $"CSV written to ($csv)"
  $rows
}
