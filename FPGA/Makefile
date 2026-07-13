DEVICE ?= --hx1k
PACKAGE ?= vq100
FREQ ?= 25
REPORT_DIR ?= reports
TEST_DIR ?= test_data

TOP ?= tcn_network_basic
SRC ?= tcn_network_basic.v tcn_layer_basic.v conv1d_engine.v conv1d_basic.v
EXTRA_SRC ?=

REPORT_BASE = $(REPORT_DIR)/$(TOP)
JSON = $(REPORT_BASE).json
ASC = $(REPORT_BASE).asc
SDF = $(REPORT_BASE).sdf
YOSYS_LOG = $(REPORT_BASE)_yosys.log
NEXTPNR_LOG = $(REPORT_BASE)_nextpnr.log
SUMMARY = $(REPORT_BASE)_summary.txt

test-network:
	mkdir -p "$(TEST_DIR)"
	iverilog -Wall -g2012 -o "$(TEST_DIR)/tcn_network_basic.vvp" conv1d_basic.v conv1d_engine.v tcn_layer_basic.v tcn_network_basic.v testbench_tcn_network_basic.v
	vvp "$(TEST_DIR)/tcn_network_basic.vvp"

throughput-network:
	mkdir -p "$(TEST_DIR)"
	iverilog -Wall -g2012 -o "$(TEST_DIR)/tcn_network_throughput.vvp" conv1d_basic.v conv1d_engine.v tcn_layer_basic.v tcn_network_basic.v testbench_tcn_network_throughput.v
	vvp "$(TEST_DIR)/tcn_network_throughput.vvp"

report:
	mkdir -p "$(REPORT_DIR)"
	yosys -q -l "$(YOSYS_LOG)" -p "read_verilog $(SRC) $(EXTRA_SRC); synth_ice40 -top $(TOP) -json $(JSON)"
	nextpnr-ice40 -q -l "$(NEXTPNR_LOG)" $(DEVICE) --package $(PACKAGE) --freq $(FREQ) --json "$(JSON)" --asc "$(ASC)" --sdf "$(SDF)"
	@{ \
		echo "Report: $(TOP)"; \
		echo "Source: $(SRC) $(EXTRA_SRC)"; \
		echo "Target: $(DEVICE) package=$(PACKAGE) freq=$(FREQ)MHz"; \
		echo ""; \
		echo "Yosys area summary:"; \
		awk '/=== $(TOP) ===/{in_stats=1; next} in_stats && /^===/{exit} in_stats && /Number of cells:|SB_LUT4|SB_DFF|SB_DFFE|SB_DFFESR|SB_DFFESS|SB_DFFSR|SB_CARRY|SB_MAC16/{print}' "$(YOSYS_LOG)" || true; \
		echo ""; \
		echo "nextpnr timing/utilization summary:"; \
		awk '/Device utilisation:/{print; in_util=1; next} in_util && /^Info: \t/{print; next} in_util && !/^Info: \t/{in_util=0} /Max frequency for clock/{print} /Critical path report for clock/{print}' "$(NEXTPNR_LOG)" || true; \
	} | tee "$(SUMMARY)"

clean:
	rm -rf "$(TEST_DIR)" "$(REPORT_DIR)"

.PHONY: test-network throughput-network report clean
