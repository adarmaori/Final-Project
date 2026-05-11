# NN Presentation Summary

- Total evaluated models: 66
- Real-time viable at 44.1 kHz (avg-sample metric): 66 (100.0%)
- Best NMSE model: tcn_input_channels-1_output_channels-1_hidden_channels-32_kernel_size-7_dilation-2_num_blocks-3 (0.2185%)
- Fastest model: tcn_input_channels-1_output_channels-1_hidden_channels-8_kernel_size-7_dilation-2_num_blocks-1 (0.0062 us/sample)

## Architecture Summary (median-centric)

| model | models | nmse_median | nmse_best | nmse_p90 | params_median | sample_us_median | sample_us_p90 | load_pct_median | load_pct_p90 | viable_models_44k | viable_share_44k_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lstm | 12 | 0.5226 | 0.3684 | 0.5956 | 5049.0000 | 11.3418 | 17.5069 | 50.0173 | 77.2054 | 12 | 100.0000 |
| tcn | 54 | 0.5349 | 0.2185 | 6.2342 | 745.0000 | 0.0103 | 0.0135 | 0.0456 | 0.0597 | 54 | 100.0000 |

## Top Candidates (balanced score)

| experiment | model | nmse_percent | avg_sample_us | avg_batch_ms | num_parameters | model_size_mb_fp32 | realtime_load_pct | headroom_x | is_realtime_viable_44k | viability_tier | presentation_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tcn_input_channels-1_output_channels-1_hidden_channels-32_kernel_size-7_dilation-2_num_blocks-3 | tcn | 0.2185 | 0.0133 | 0.2179 | 14881 | 0.0568 | 0.0587 | 1704.8692 | True | strong_headroom | 0.0031 |
| tcn_input_channels-1_output_channels-1_hidden_channels-16_kernel_size-5_dilation-2_num_blocks-3 | tcn | 0.2542 | 0.0140 | 0.2297 | 2769 | 0.0106 | 0.0618 | 1617.1461 | True | strong_headroom | 0.0036 |
| tcn_input_channels-1_output_channels-1_hidden_channels-16_kernel_size-7_dilation-2_num_blocks-3 | tcn | 0.2667 | 0.0131 | 0.2144 | 3857 | 0.0147 | 0.0577 | 1732.9154 | True | strong_headroom | 0.0037 |
| tcn_input_channels-1_output_channels-1_hidden_channels-32_kernel_size-7_dilation-1_num_blocks-3 | tcn | 0.2656 | 0.0149 | 0.2437 | 14881 | 0.0568 | 0.0656 | 1524.4424 | True | strong_headroom | 0.0038 |
| tcn_input_channels-1_output_channels-1_hidden_channels-32_kernel_size-5_dilation-2_num_blocks-3 | tcn | 0.2685 | 0.0134 | 0.2202 | 10657 | 0.0407 | 0.0593 | 1686.9503 | True | strong_headroom | 0.0038 |
| tcn_input_channels-1_output_channels-1_hidden_channels-32_kernel_size-3_dilation-2_num_blocks-3 | tcn | 0.2734 | 0.0132 | 0.2157 | 6433 | 0.0245 | 0.0581 | 1722.0720 | True | strong_headroom | 0.0038 |
| tcn_input_channels-1_output_channels-1_hidden_channels-32_kernel_size-7_dilation-2_num_blocks-2 | tcn | 0.2802 | 0.0106 | 0.1730 | 7681 | 0.0293 | 0.0466 | 2147.0966 | True | strong_headroom | 0.0039 |
| tcn_input_channels-1_output_channels-1_hidden_channels-32_kernel_size-3_dilation-2_num_blocks-2 | tcn | 0.2882 | 0.0103 | 0.1685 | 3329 | 0.0127 | 0.0454 | 2204.9691 | True | strong_headroom | 0.0040 |
| tcn_input_channels-1_output_channels-1_hidden_channels-32_kernel_size-5_dilation-2_num_blocks-2 | tcn | 0.2925 | 0.0103 | 0.1680 | 5505 | 0.0210 | 0.0452 | 2211.2055 | True | strong_headroom | 0.0040 |
| tcn_input_channels-1_output_channels-1_hidden_channels-16_kernel_size-3_dilation-2_num_blocks-3 | tcn | 0.2938 | 0.0133 | 0.2182 | 1681 | 0.0064 | 0.0587 | 1702.5250 | True | strong_headroom | 0.0041 |
| tcn_input_channels-1_output_channels-1_hidden_channels-16_kernel_size-3_dilation-2_num_blocks-2 | tcn | 0.3029 | 0.0105 | 0.1725 | 897 | 0.0034 | 0.0464 | 2154.2294 | True | strong_headroom | 0.0042 |
| tcn_input_channels-1_output_channels-1_hidden_channels-32_kernel_size-3_dilation-1_num_blocks-3 | tcn | 0.3158 | 0.0131 | 0.2144 | 6433 | 0.0245 | 0.0577 | 1733.1021 | True | strong_headroom | 0.0044 |

## Notes

- Real-time viability threshold uses 44100 Hz sample budget (22.6757 us/sample).
- Chunk-level latency is shown via chunk budget utilization; per-sample load is preferred for cross-architecture viability comparisons.