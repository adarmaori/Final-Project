#include <Bela.h>
#include <libraries/AudioFile/AudioFile.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace {

struct AppConfig {
	std::string configFile = "experiment_config.cfg";
	bool loadConfigFile = true;
	std::string inputFile = "funk-soul-guitar-clean-4_90bpm_G.wav";
	std::string outputFile = "distorted4.wav";
	bool writeOutputFile = true;
	bool writeAudioToOutputs = false;
	bool clipOutput = false;
	float inputGain = 1.0f;
	float outputGain = 1.0f;
};

struct BenchmarkConfig {
	bool enabled = true;
	bool writeCsv = true;
	bool sampleSystemCpu = false;
	bool recordPerBlockRows = true;
	std::string csvFile = "bench.csv";
	uint64_t warmupBlocks = 0;
	uint64_t cpuSampleEveryBlocks = 100;
};

struct RuntimeConfig {
	bool stopWhenInputEnds = true;
	bool zeroUnusedOutputChannels = true;
};

enum class ExecutionMode {
	FusedStreaming,
	LayeredBlock
};

enum class HistoryMode {
	Shift,
	Circular
};

enum class ScratchMode {
	Persistent,
	PerChunk
};

struct ExperimentConfig {
	ExecutionMode executionMode = ExecutionMode::FusedStreaming;
	HistoryMode historyMode = HistoryMode::Shift;
	ScratchMode scratchMode = ScratchMode::Persistent;
	unsigned int inferenceChunkFrames = 0;
};

struct SweepConfig {
	bool enabled = false;
	std::string csvPrefix = "results/sweep";
};

struct SweepCase {
	const char* name;
	ExecutionMode executionMode;
	HistoryMode historyMode;
	ScratchMode scratchMode;
	unsigned int inferenceChunkFrames;
};

static const SweepCase kSweepCases[] = {
	{ "naive_chunk1", ExecutionMode::LayeredBlock, HistoryMode::Shift, ScratchMode::PerChunk, 1 },
	{ "naive_chunk2", ExecutionMode::LayeredBlock, HistoryMode::Shift, ScratchMode::PerChunk, 2 },
	{ "naive_chunk4", ExecutionMode::LayeredBlock, HistoryMode::Shift, ScratchMode::PerChunk, 4 },
	{ "naive_chunk8", ExecutionMode::LayeredBlock, HistoryMode::Shift, ScratchMode::PerChunk, 8 },
	{ "naive_chunk16", ExecutionMode::LayeredBlock, HistoryMode::Shift, ScratchMode::PerChunk, 16 },
	{ "fused_chunk1", ExecutionMode::FusedStreaming, HistoryMode::Shift, ScratchMode::PerChunk, 1 },
	{ "fused_chunk2", ExecutionMode::FusedStreaming, HistoryMode::Shift, ScratchMode::PerChunk, 2 },
	{ "fused_chunk4", ExecutionMode::FusedStreaming, HistoryMode::Shift, ScratchMode::PerChunk, 4 },
	{ "fused_chunk8", ExecutionMode::FusedStreaming, HistoryMode::Shift, ScratchMode::PerChunk, 8 },
	{ "fused_chunk16", ExecutionMode::FusedStreaming, HistoryMode::Shift, ScratchMode::PerChunk, 16 },
	{ "circular_chunk1", ExecutionMode::LayeredBlock, HistoryMode::Circular, ScratchMode::PerChunk, 1 },
	{ "circular_chunk2", ExecutionMode::LayeredBlock, HistoryMode::Circular, ScratchMode::PerChunk, 2 },
	{ "circular_chunk4", ExecutionMode::LayeredBlock, HistoryMode::Circular, ScratchMode::PerChunk, 4 },
	{ "circular_chunk8", ExecutionMode::LayeredBlock, HistoryMode::Circular, ScratchMode::PerChunk, 8 },
	{ "circular_chunk16", ExecutionMode::LayeredBlock, HistoryMode::Circular, ScratchMode::PerChunk, 16 },
	{ "persistent_chunk1", ExecutionMode::LayeredBlock, HistoryMode::Shift, ScratchMode::Persistent, 1 },
	{ "persistent_chunk2", ExecutionMode::LayeredBlock, HistoryMode::Shift, ScratchMode::Persistent, 2 },
	{ "persistent_chunk4", ExecutionMode::LayeredBlock, HistoryMode::Shift, ScratchMode::Persistent, 4 },
	{ "persistent_chunk8", ExecutionMode::LayeredBlock, HistoryMode::Shift, ScratchMode::Persistent, 8 },
	{ "persistent_chunk16", ExecutionMode::LayeredBlock, HistoryMode::Shift, ScratchMode::Persistent, 16 },
	{ "combined_chunk1", ExecutionMode::FusedStreaming, HistoryMode::Circular, ScratchMode::Persistent, 1 },
	{ "combined_chunk2", ExecutionMode::FusedStreaming, HistoryMode::Circular, ScratchMode::Persistent, 2 },
	{ "combined_chunk4", ExecutionMode::FusedStreaming, HistoryMode::Circular, ScratchMode::Persistent, 4 },
	{ "combined_chunk8", ExecutionMode::FusedStreaming, HistoryMode::Circular, ScratchMode::Persistent, 8 },
	{ "combined_chunk16", ExecutionMode::FusedStreaming, HistoryMode::Circular, ScratchMode::Persistent, 16 }
};

constexpr size_t kSweepCaseCount = sizeof(kSweepCases) / sizeof(kSweepCases[0]);

constexpr int kNumConvLayers = 2;
constexpr int kHiddenChannels = 16;
constexpr int kKernelSize = 3;
constexpr int kMaxHistory = 2;
static const int kDilations[kNumConvLayers] = { 1, 1 };

static const float conv0_w[kHiddenChannels][1][kKernelSize] = {
    { { 0.644592285f, 0.533045709f, 1.05618882f } },
    { { 0.841638207f, 1.20650136f, 0.484849036f } },
    { { 0.47314477f, -0.162172526f, -0.00907641649f } },
    { { 0.0513471365f, -0.554325521f, 0.172708333f } },
    { { 0.112497091f, 0.0691599846f, -0.559089601f } },
    { { -0.153777197f, -0.585065305f, -0.376663148f } },
    { { 0.371859968f, 0.140536308f, -0.559399128f } },
    { { 0.340878069f, -0.426355064f, 0.338769138f } },
    { { 0.629515827f, 0.995258927f, 1.52625501f } },
    { { 1.04953897f, 1.29734933f, 1.05061269f } },
    { { 1.60446501f, 0.525971055f, 0.949755192f } },
    { { -0.530482829f, -1.32829392f, -1.09715736f } },
    { { 0.0214586854f, -0.343939543f, 0.280881107f } },
    { { -0.105528556f, -1.26699495f, -1.27791333f } },
    { { -1.35419369f, -1.12423193f, -0.694253623f } },
    { { -0.699474514f, -1.21688926f, -1.30189013f } },
};
static const float conv_b[kNumConvLayers][kHiddenChannels] = {
    { 0.0652296096f, 0.0847290233f, -0.346418917f, -0.518137693f, -0.474925846f, -0.290601343f, -0.48009038f, -0.279593319f, -0.0470745601f, -0.0541700311f, 0.118892558f, -0.0853675157f, -0.405216485f, -0.0708813891f, 0.116002999f, 0.0637843609f },
    { 0.0f },
};
static const float final_w[kHiddenChannels][kKernelSize] = {
    { 0.600854933f, 0.223445401f, 0.392813891f },
    { 0.921911418f, 0.644353986f, 0.628558874f },
    { 0.0919911116f, 0.00436902046f, 0.0830723196f },
    { 0.0454472303f, -0.123378508f, 0.133147031f },
    { -0.0539488941f, -0.0504148304f, -0.105675921f },
    { 0.201495975f, -0.128252253f, -0.19852151f },
    { 0.0683611929f, -0.050033398f, -0.126710922f },
    { -0.0552411675f, 0.10406217f, 0.00633007288f },
    { -1.00763011f, -0.967031956f, -0.621490121f },
    { -0.553512871f, -0.690842211f, -0.677569807f },
    { 0.920573592f, 0.632871032f, 0.587244153f },
    { 0.872034132f, 0.851120293f, 0.823450565f },
    { -0.0996923298f, 0.0607897639f, 0.109512061f },
    { 0.771921873f, 0.598700762f, 0.605151772f },
    { -0.751569152f, -0.418815106f, -0.254063308f },
    { -1.11805892f, -0.670774639f, -0.783497453f },
};
static const float final_b = -0.0959965885f;


struct BenchmarkRow {
	uint32_t modelNs = 0;
	uint32_t blockNs = 0;
	uint32_t framesProcessed = 0;
	uint16_t sysCpu_x100 = 0xFFFF;
	uint32_t memRssKb = 0xFFFFFFFFu;
	uint8_t overrun = 0;
	uint8_t warmup = 0;
};

struct RunStats {
	uint64_t count = 0;
	uint64_t totalNs = 0;
	uint64_t minNs = std::numeric_limits<uint64_t>::max();
	uint64_t maxNs = 0;
	uint64_t overruns = 0;
};

struct CompletedStats {
	double avgNs = 0.0;
	double minNs = 0.0;
	double maxNs = 0.0;
	double stddevNs = 0.0;
	double p50Ns = 0.0;
	double p90Ns = 0.0;
	double p95Ns = 0.0;
	double p99Ns = 0.0;
	double avgBudgetPct = 0.0;
	double maxBudgetPct = 0.0;
	double avgAbsJitterNs = 0.0;
	double maxAbsJitterNs = 0.0;
	uint64_t overruns = 0;
};

struct MonitorStats {
	double avg = 0.0;
	double p95 = 0.0;
	double max = 0.0;
};

struct BenchmarkState {
	std::vector<BenchmarkRow> rows;
	std::vector<uint64_t> modelNs;
	std::vector<uint64_t> blockNs;
	std::vector<double> systemCpuPct;
	std::vector<double> processMemoryPct;
	std::vector<double> processRssMb;
	RunStats model;
	RunStats block;
	uint64_t blocksSeen = 0;
	double blockBudgetNs = 0.0;
	uint64_t totalMemoryKb = 0;
};

struct ShiftChannelState {
	float history[kNumConvLayers][kMaxHistory][kHiddenChannels];

	ShiftChannelState()
	{
		reset();
	}

	void reset()
	{
		for(int layer = 0; layer < kNumConvLayers; ++layer) {
			for(int delay = 0; delay < kMaxHistory; ++delay) {
				for(int channel = 0; channel < kHiddenChannels; ++channel) {
					history[layer][delay][channel] = 0.0f;
	}
		}
	}
	}

	float previous(int layer, int channel, int delay) const
	{
		return history[layer][delay - 1][channel];
	}

	void push(int layer, const float* values, int channels)
	{
		const int length = (kKernelSize - 1) * kDilations[layer];
		for(int delay = length - 1; delay > 0; --delay) {
			for(int channel = 0; channel < channels; ++channel) {
				history[layer][delay][channel] = history[layer][delay - 1][channel];
			}
		}
		for(int channel = 0; channel < channels; ++channel) {
			history[layer][0][channel] = values[channel];
		}
	}
};

struct CircularChannelState {
	float history[kNumConvLayers][kMaxHistory][kHiddenChannels];
	unsigned int heads[kNumConvLayers] = {};

	CircularChannelState()
	{
		reset();
	}

	void reset()
	{
		for(int layer = 0; layer < kNumConvLayers; ++layer) {
			heads[layer] = 0;
			for(int delay = 0; delay < kMaxHistory; ++delay) {
				for(int channel = 0; channel < kHiddenChannels; ++channel) {
					history[layer][delay][channel] = 0.0f;
				}
			}
		}
	}

	float previous(int layer, int channel, int delay) const
	{
		const int length = (kKernelSize - 1) * kDilations[layer];
		const int index = (int(heads[layer]) + length - (delay - 1)) % length;
		return history[layer][index][channel];
	}

	void push(int layer, const float* values, int channels)
	{
		const int length = (kKernelSize - 1) * kDilations[layer];
		heads[layer] = (heads[layer] + 1) % length;
		for(int channel = 0; channel < channels; ++channel) {
			history[layer][heads[layer]][channel] = values[channel];
		}
	}
};

class TCNProcessor {
public:
	virtual ~TCNProcessor() {}
	virtual const char* name() const = 0;
	virtual void reset(size_t channels, unsigned int maxChunkFrames) = 0;
	virtual void processChunk(
		const std::vector<std::vector<float>>& input,
		std::vector<std::vector<float>>& output,
		unsigned int startFrame,
		unsigned int frames,
		float inputGain,
		float outputGain,
		bool clipOutput
	) = 0;
};

static inline uint64_t nsNow()
{
	timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static inline float relu(float x)
{
	return x > 0.0f ? x : 0.0f;
}

static inline float applyOutputPostprocess(float x, float outputGain, bool clipOutput)
{
	float y = x * outputGain;
	if(clipOutput) {
		y = std::max(-1.0f, std::min(1.0f, y));
	}
	return y;
}

static std::string trim(const std::string& input)
{
	const std::string whitespace = " \t\r\n";
	const size_t begin = input.find_first_not_of(whitespace);
	if(begin == std::string::npos) {
		return "";
	}
	const size_t end = input.find_last_not_of(whitespace);
	return input.substr(begin, end - begin + 1);
}

static std::string lowerCopy(std::string value)
{
	std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
		return (char)std::tolower(c);
	});
	return value;
}

static bool parseBool(const std::string& value, bool& out)
{
	const std::string lowered = lowerCopy(trim(value));
	if(lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "on") {
		out = true;
		return true;
	}
	if(lowered == "0" || lowered == "false" || lowered == "no" || lowered == "off") {
		out = false;
		return true;
	}
	return false;
}

static bool parseUInt64(const std::string& value, uint64_t& out)
{
	try {
		out = std::stoull(trim(value));
		return true;
	} catch(...) {
		return false;
	}
}

static bool parseUInt(const std::string& value, unsigned int& out)
{
	uint64_t parsed = 0;
	if(!parseUInt64(value, parsed)) {
		return false;
	}
	out = (unsigned int)parsed;
	return true;
}

static bool parseFloat(const std::string& value, float& out)
{
	try {
		out = std::stof(trim(value));
		return true;
	} catch(...) {
		return false;
	}
}

static const char* executionModeName(ExecutionMode mode)
{
	switch(mode) {
	case ExecutionMode::FusedStreaming:
		return "fused_streaming";
	case ExecutionMode::LayeredBlock:
		return "layered_block";
	}
	return "unknown";
}

static const char* historyModeName(HistoryMode mode)
{
	switch(mode) {
	case HistoryMode::Shift:
		return "shift";
	case HistoryMode::Circular:
		return "circular";
	}
	return "unknown";
}

static const char* scratchModeName(ScratchMode mode)
{
	switch(mode) {
	case ScratchMode::Persistent:
		return "persistent";
	case ScratchMode::PerChunk:
		return "per_chunk";
	}
	return "unknown";
}

static bool parseExecutionMode(const std::string& value, ExecutionMode& out)
{
	const std::string lowered = lowerCopy(trim(value));
	if(lowered == "fused_streaming" || lowered == "fused") {
		out = ExecutionMode::FusedStreaming;
		return true;
	}
	if(lowered == "layered_block" || lowered == "layered") {
		out = ExecutionMode::LayeredBlock;
		return true;
	}
	return false;
}

static bool parseHistoryMode(const std::string& value, HistoryMode& out)
{
	const std::string lowered = lowerCopy(trim(value));
	if(lowered == "shift") {
		out = HistoryMode::Shift;
		return true;
	}
	if(lowered == "circular") {
		out = HistoryMode::Circular;
		return true;
	}
	return false;
}

static bool parseScratchMode(const std::string& value, ScratchMode& out)
{
	const std::string lowered = lowerCopy(trim(value));
	if(lowered == "persistent") {
		out = ScratchMode::Persistent;
		return true;
	}
	if(lowered == "per_chunk" || lowered == "alloc_per_chunk") {
		out = ScratchMode::PerChunk;
		return true;
	}
	return false;
}

template <typename StateT>
static inline void computeHiddenLayer(
	const float* input,
	StateT& state,
	float* output
)
{
	for(int out = 0; out < kHiddenChannels; ++out) {
		float z = conv_b[0][out] + conv0_w[out][0][0] * input[0];
		for(int tap = 1; tap < kKernelSize; ++tap) {
			z += conv0_w[out][0][tap] * state.previous(0, 0, tap);
		}
		output[out] = relu(z);
	}
}

template <typename StateT>
static inline float computeOutput(const float* hidden, StateT& state)
{
	float y = final_b;
	for(int channel = 0; channel < kHiddenChannels; ++channel) {
		y += final_w[channel][0] * hidden[channel];
		for(int tap = 1; tap < kKernelSize; ++tap) {
			y += final_w[channel][tap] * state.previous(1, channel, tap);
		}
	}
	return y;
}

template <typename StateT>
class FusedStreamingProcessor : public TCNProcessor {
public:
	const char* name() const override
	{
		return "fused_streaming";
	}

	void reset(size_t channels, unsigned int) override
	{
		states_.assign(channels, StateT());
	}

	void processChunk(
		const std::vector<std::vector<float>>& input,
		std::vector<std::vector<float>>& output,
		unsigned int startFrame,
		unsigned int frames,
		float inputGain,
		float outputGain,
		bool clipOutput
	) override
	{
		for(size_t ch = 0; ch < states_.size(); ++ch) {
			StateT& state = states_[ch];
			for(unsigned int frame = 0; frame < frames; ++frame) {
				const float x = input[ch][startFrame + frame] * inputGain;
				float hidden[kHiddenChannels] = {};
				computeHiddenLayer(&x, state, hidden);
				state.push(0, &x, 1);
				const float y = computeOutput(hidden, state);
				state.push(1, hidden, kHiddenChannels);
				output[ch][startFrame + frame] = applyOutputPostprocess(y, outputGain, clipOutput);
			}
		}
	}

private:
	std::vector<StateT> states_;
};

template <typename StateT>
class LayeredBlockProcessor : public TCNProcessor {
public:
	explicit LayeredBlockProcessor(ScratchMode scratchMode)
		: scratchMode_(scratchMode)
	{
	}

	const char* name() const override
	{
		return "layered_block";
	}

	void reset(size_t channels, unsigned int maxChunkFrames) override
	{
		states_.assign(channels, StateT());
		if(scratchMode_ == ScratchMode::Persistent) {
			scratch_.assign((size_t)maxChunkFrames * (size_t)kHiddenChannels, 0.0f);
		} else {
			scratch_.clear();
		}
	}

	void processChunk(
		const std::vector<std::vector<float>>& input,
		std::vector<std::vector<float>>& output,
		unsigned int startFrame,
		unsigned int frames,
		float inputGain,
		float outputGain,
		bool clipOutput
	) override
	{
		for(size_t ch = 0; ch < states_.size(); ++ch) {
			std::vector<float> localScratch;
			std::vector<float>& scratch = prepareScratch(localScratch, frames);
			StateT& state = states_[ch];

			for(unsigned int frame = 0; frame < frames; ++frame) {
				const float x = input[ch][startFrame + frame] * inputGain;
				float* hidden = &scratch[(size_t)frame * (size_t)kHiddenChannels];
				computeHiddenLayer(&x, state, hidden);
				state.push(0, &x, 1);
			}

			for(unsigned int frame = 0; frame < frames; ++frame) {
				const float* hidden = &scratch[(size_t)frame * (size_t)kHiddenChannels];
				const float y = computeOutput(hidden, state);
				state.push(1, hidden, kHiddenChannels);
				output[ch][startFrame + frame] = applyOutputPostprocess(y, outputGain, clipOutput);
			}
		}
	}

private:
	std::vector<float>& prepareScratch(std::vector<float>& localScratch, unsigned int frames)
	{
		const size_t required = (size_t)frames * (size_t)kHiddenChannels;
		if(scratchMode_ == ScratchMode::Persistent) {
			if(scratch_.size() < required) {
				scratch_.resize(required, 0.0f);
			}
			return scratch_;
		}

		localScratch.assign(required, 0.0f);
		return localScratch;
	}

	ScratchMode scratchMode_;
	std::vector<StateT> states_;
	std::vector<float> scratch_;
};



AppConfig gAppConfig;
BenchmarkConfig gBenchConfig;
RuntimeConfig gRuntimeConfig;
ExperimentConfig gExperimentConfig;
SweepConfig gSweepConfig;
BenchmarkState gBenchState;
std::unique_ptr<TCNProcessor> gProcessor;
std::vector<std::unique_ptr<TCNProcessor>> gSweepProcessors;
std::vector<BenchmarkState> gSweepStates;
std::vector<std::vector<float>> gSamples;
std::vector<std::vector<float>> gOutput;
unsigned int gReadPtr = 0;

static double readSystemCpuUsagePct()
{
	static uint64_t prevIdle = 0;
	static uint64_t prevTotal = 0;

	FILE* f = fopen("/proc/stat", "r");
	if(!f) {
		return -1.0;
	}

	uint64_t user = 0;
	uint64_t nice = 0;
	uint64_t system = 0;
	uint64_t idle = 0;
	uint64_t iowait = 0;
	uint64_t irq = 0;
	uint64_t softirq = 0;
	uint64_t steal = 0;
	int n = fscanf(
		f,
		"cpu %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64 " %" SCNu64,
		&user, &nice, &system, &idle, &iowait, &irq, &softirq, &steal
	);
	fclose(f);
	if(n < 4) {
		return -1.0;
	}

	const uint64_t idleNow = idle + iowait;
	const uint64_t totalNow = user + nice + system + idle + iowait + irq + softirq + steal;
	const uint64_t diffIdle = idleNow - prevIdle;
	const uint64_t diffTotal = totalNow - prevTotal;

	prevIdle = idleNow;
	prevTotal = totalNow;

	if(diffTotal == 0) {
		return -1.0;
	}

	return 100.0 * (1.0 - (double)diffIdle / (double)diffTotal);
}

static uint64_t readTotalMemoryKb()
{
	FILE* f = fopen("/proc/meminfo", "r");
	if(!f) {
		return 0;
	}

	char key[64];
	uint64_t value = 0;
	char unit[32];
	while(fscanf(f, "%63s %" SCNu64 " %31s", key, &value, unit) == 3) {
		if(std::string(key) == "MemTotal:") {
			fclose(f);
			return value;
		}
	}

	fclose(f);
	return 0;
}

static uint64_t readProcessRssKb()
{
	FILE* f = fopen("/proc/self/status", "r");
	if(!f) {
		return 0;
	}

	char line[256];
	while(fgets(line, sizeof(line), f)) {
		uint64_t value = 0;
		if(sscanf(line, "VmRSS: %" SCNu64 " kB", &value) == 1) {
			fclose(f);
			return value;
		}
	}

	fclose(f);
	return 0;
}

static void resetBenchmarkState(BenchmarkState& state, uint64_t estimatedBlocks, double blockBudgetNs)
{
	state = BenchmarkState();
	state.blockBudgetNs = blockBudgetNs;

	if(!gBenchConfig.enabled) {
		return;
	}

	if(gBenchConfig.recordPerBlockRows) {
		state.rows.reserve(estimatedBlocks);
	}
	state.modelNs.reserve(estimatedBlocks);
	state.blockNs.reserve(estimatedBlocks);

	if(gBenchConfig.sampleSystemCpu) {
		state.totalMemoryKb = readTotalMemoryKb();
		const uint64_t estimatedSamples = (gBenchConfig.cpuSampleEveryBlocks > 0)
			? ((estimatedBlocks + gBenchConfig.cpuSampleEveryBlocks - 1) / gBenchConfig.cpuSampleEveryBlocks)
			: 0;
		state.systemCpuPct.reserve(estimatedSamples);
		state.processMemoryPct.reserve(estimatedSamples);
		state.processRssMb.reserve(estimatedSamples);
		(void)readSystemCpuUsagePct();
	}
}

static void updateRunStats(RunStats& stats, uint64_t nsValue, bool overrun)
{
	stats.count++;
	stats.totalNs += nsValue;
	stats.minNs = std::min(stats.minNs, nsValue);
	stats.maxNs = std::max(stats.maxNs, nsValue);
	if(overrun) {
		stats.overruns++;
	}
}

static uint32_t clampToU32(uint64_t value)
{
	return value > 0xFFFFFFFFull ? 0xFFFFFFFFu : (uint32_t)value;
}

static bool isWarmupBlock(uint64_t blockIndex)
{
	return blockIndex < gBenchConfig.warmupBlocks;
}

static void recordBenchmarkBlock(BenchmarkState& state, uint64_t modelNs, uint64_t blockNs, unsigned int framesProcessed)
{
	if(!gBenchConfig.enabled) {
		return;
	}

	const uint64_t blockIndex = state.blocksSeen++;
	const bool warmup = isWarmupBlock(blockIndex);
	const bool overrun = (state.blockBudgetNs > 0.0) && ((double)blockNs > state.blockBudgetNs);
	if(gBenchConfig.recordPerBlockRows) {
		BenchmarkRow row;
		row.modelNs = clampToU32(modelNs);
		row.blockNs = clampToU32(blockNs);
		row.framesProcessed = framesProcessed;
		row.overrun = overrun ? 1 : 0;
		row.warmup = warmup ? 1 : 0;

		state.rows.push_back(row);
	}

	if(warmup) {
		return;
	}

	state.modelNs.push_back(modelNs);
	state.blockNs.push_back(blockNs);
	updateRunStats(state.model, modelNs, (state.blockBudgetNs > 0.0) && ((double)modelNs > state.blockBudgetNs));
	updateRunStats(state.block, blockNs, overrun);
}

static double percentileFromSorted(const std::vector<uint64_t>& sortedValues, double fraction)
{
	if(sortedValues.empty()) {
		return 0.0;
	}

	const double clamped = std::max(0.0, std::min(1.0, fraction));
	const size_t idx = (size_t)llround(clamped * (double)(sortedValues.size() - 1));
	return (double)sortedValues[idx];
}

static double stddevFromValues(const std::vector<uint64_t>& values, double mean)
{
	if(values.empty()) {
		return 0.0;
	}

	double accum = 0.0;
	for(uint64_t value : values) {
		const double delta = (double)value - mean;
		accum += delta * delta;
	}
	return std::sqrt(accum / (double)values.size());
}

static void computeJitterStats(
	const std::vector<uint64_t>& values,
	double& avgAbsJitterNs,
	double& maxAbsJitterNs
)
{
	avgAbsJitterNs = 0.0;
	maxAbsJitterNs = 0.0;

	if(values.size() < 2) {
		return;
	}

	double totalAbsDelta = 0.0;
	for(size_t i = 1; i < values.size(); ++i) {
		const double delta = std::fabs((double)values[i] - (double)values[i - 1]);
		totalAbsDelta += delta;
		maxAbsJitterNs = std::max(maxAbsJitterNs, delta);
	}

	avgAbsJitterNs = totalAbsDelta / (double)(values.size() - 1);
}

static CompletedStats computeCompletedStats(const std::vector<uint64_t>& values, const RunStats& runStats, double blockBudgetNs)
{
	CompletedStats out;
	if(values.empty() || runStats.count == 0) {
		return out;
	}

	std::vector<uint64_t> sortedValues = values;
	std::sort(sortedValues.begin(), sortedValues.end());

	out.avgNs = (double)runStats.totalNs / (double)runStats.count;
	out.minNs = (double)runStats.minNs;
	out.maxNs = (double)runStats.maxNs;
	out.stddevNs = stddevFromValues(values, out.avgNs);
	out.p50Ns = percentileFromSorted(sortedValues, 0.50);
	out.p90Ns = percentileFromSorted(sortedValues, 0.90);
	out.p95Ns = percentileFromSorted(sortedValues, 0.95);
	out.p99Ns = percentileFromSorted(sortedValues, 0.99);
	out.overruns = runStats.overruns;

	computeJitterStats(values, out.avgAbsJitterNs, out.maxAbsJitterNs);

	if(blockBudgetNs > 0.0) {
		out.avgBudgetPct = 100.0 * out.avgNs / blockBudgetNs;
		out.maxBudgetPct = 100.0 * out.maxNs / blockBudgetNs;
	}

	return out;
}

static MonitorStats computeMonitorStats(const std::vector<double>& values)
{
	MonitorStats out;
	if(values.empty()) {
		return out;
	}

	std::vector<double> sortedValues = values;
	std::sort(sortedValues.begin(), sortedValues.end());

	double total = 0.0;
	for(double value : values) {
		total += value;
	}

	out.avg = total / (double)values.size();
	out.p95 = sortedValues[(size_t)llround(0.95 * (double)(sortedValues.size() - 1))];
	out.max = sortedValues.back();
	return out;
}

static void printCompletedStats(const char* label, const CompletedStats& stats)
{
	rt_printf(
		"%s avg %.0f ns (%.3f ms), min %.0f ns, p50 %.0f ns, p90 %.0f ns, p95 %.0f ns, p99 %.0f ns, max %.0f ns\n",
		label,
		stats.avgNs, stats.avgNs / 1e6,
		stats.minNs, stats.p50Ns, stats.p90Ns, stats.p95Ns, stats.p99Ns, stats.maxNs
	);
	rt_printf(
		"%s stdev %.0f ns, avg abs jitter %.0f ns, max abs jitter %.0f ns, avg budget %.2f%%, max budget %.2f%%, overruns %" PRIu64 "\n",
		label,
		stats.stddevNs,
		stats.avgAbsJitterNs,
		stats.maxAbsJitterNs,
		stats.avgBudgetPct,
		stats.maxBudgetPct,
		stats.overruns
	);
}

static void printMonitorSummary(const BenchmarkState& state)
{
	if(!gBenchConfig.sampleSystemCpu) {
		return;
	}

	const MonitorStats cpuStats = computeMonitorStats(state.systemCpuPct);
	const MonitorStats memPctStats = computeMonitorStats(state.processMemoryPct);
	const MonitorStats rssStats = computeMonitorStats(state.processRssMb);

	rt_printf("\n--- Processor monitor summary ---\n");
	rt_printf("Samples: %zu, every %" PRIu64 " block(s)\n", state.systemCpuPct.size(), gBenchConfig.cpuSampleEveryBlocks);

	if(state.systemCpuPct.empty()) {
		rt_printf("No processor monitor samples collected\n");
		return;
	}

	rt_printf(
		"System CPU avg %.2f%%, p95 %.2f%%, max %.2f%%\n",
		cpuStats.avg,
		cpuStats.p95,
		cpuStats.max
	);

	rt_printf(
		"Process RSS avg %.2f MiB, p95 %.2f MiB, max %.2f MiB\n",
		rssStats.avg,
		rssStats.p95,
		rssStats.max
	);

	if(!state.processMemoryPct.empty()) {
		rt_printf(
			"Process memory avg %.2f%%, p95 %.2f%%, max %.2f%% of RAM\n",
			memPctStats.avg,
			memPctStats.p95,
			memPctStats.max
		);
	}
}

static void zeroHardwareOutputs(BelaContext* context, unsigned int startFrame)
{
	if(!gRuntimeConfig.zeroUnusedOutputChannels) {
		return;
	}

	for(unsigned int n = startFrame; n < context->audioFrames; ++n) {
		for(unsigned int ch = 0; ch < context->audioOutChannels; ++ch) {
			audioWrite(context, n, ch, 0.0f);
		}
	}
}

static bool loadConfigFile()
{
	if(!gAppConfig.loadConfigFile) {
		return true;
	}

	std::ifstream config(gAppConfig.configFile.c_str());
	if(!config.good()) {
		rt_printf("Config file %s not found, using in-source defaults\n", gAppConfig.configFile.c_str());
		return true;
	}

	std::string line;
	unsigned int lineNo = 0;
	while(std::getline(config, line)) {
		++lineNo;
		const size_t commentPos = line.find('#');
		if(commentPos != std::string::npos) {
			line = line.substr(0, commentPos);
		}
		line = trim(line);
		if(line.empty()) {
			continue;
		}

		const size_t eqPos = line.find('=');
		if(eqPos == std::string::npos) {
			rt_printf("Ignoring malformed config line %u: %s\n", lineNo, line.c_str());
			continue;
		}

		const std::string key = lowerCopy(trim(line.substr(0, eqPos)));
		const std::string value = trim(line.substr(eqPos + 1));

		bool handled = false;
		bool boolValue = false;
		uint64_t u64Value = 0;
		unsigned int uintValue = 0;
		float floatValue = 0.0f;

		if(key == "app.input_file") {
			gAppConfig.inputFile = value;
			handled = true;
		} else if(key == "app.output_file") {
			gAppConfig.outputFile = value;
			handled = true;
		} else if(key == "app.write_output_file" && parseBool(value, boolValue)) {
			gAppConfig.writeOutputFile = boolValue;
			handled = true;
		} else if(key == "app.write_audio_to_outputs" && parseBool(value, boolValue)) {
			gAppConfig.writeAudioToOutputs = boolValue;
			handled = true;
		} else if(key == "app.clip_output" && parseBool(value, boolValue)) {
			gAppConfig.clipOutput = boolValue;
			handled = true;
		} else if(key == "app.input_gain" && parseFloat(value, floatValue)) {
			gAppConfig.inputGain = floatValue;
			handled = true;
		} else if(key == "app.output_gain" && parseFloat(value, floatValue)) {
			gAppConfig.outputGain = floatValue;
			handled = true;
		} else if(key == "bench.enabled" && parseBool(value, boolValue)) {
			gBenchConfig.enabled = boolValue;
			handled = true;
		} else if(key == "bench.write_csv" && parseBool(value, boolValue)) {
			gBenchConfig.writeCsv = boolValue;
			handled = true;
		} else if(key == "bench.sample_system_cpu" && parseBool(value, boolValue)) {
			gBenchConfig.sampleSystemCpu = boolValue;
			handled = true;
		} else if(key == "bench.record_per_block_rows" && parseBool(value, boolValue)) {
			gBenchConfig.recordPerBlockRows = boolValue;
			handled = true;
		} else if(key == "bench.csv_file") {
			gBenchConfig.csvFile = value;
			handled = true;
		} else if(key == "bench.warmup_blocks" && parseUInt64(value, u64Value)) {
			gBenchConfig.warmupBlocks = u64Value;
			handled = true;
		} else if(key == "bench.cpu_sample_every_blocks" && parseUInt64(value, u64Value)) {
			gBenchConfig.cpuSampleEveryBlocks = u64Value;
			handled = true;
		} else if(key == "runtime.stop_when_input_ends" && parseBool(value, boolValue)) {
			gRuntimeConfig.stopWhenInputEnds = boolValue;
			handled = true;
		} else if(key == "runtime.zero_unused_output_channels" && parseBool(value, boolValue)) {
			gRuntimeConfig.zeroUnusedOutputChannels = boolValue;
			handled = true;
		} else if(key == "experiment.execution_mode" && parseExecutionMode(value, gExperimentConfig.executionMode)) {
			handled = true;
		} else if(key == "experiment.history_mode" && parseHistoryMode(value, gExperimentConfig.historyMode)) {
			handled = true;
		} else if(key == "experiment.scratch_mode" && parseScratchMode(value, gExperimentConfig.scratchMode)) {
			handled = true;
		} else if(key == "experiment.inference_chunk_frames" && parseUInt(value, uintValue)) {
			gExperimentConfig.inferenceChunkFrames = uintValue;
			handled = true;
		} else if(key == "sweep.enabled" && parseBool(value, boolValue)) {
			gSweepConfig.enabled = boolValue;
			handled = true;
		} else if(key == "sweep.csv_prefix") {
			gSweepConfig.csvPrefix = value;
			handled = true;
		}

		if(!handled) {
			rt_printf("Ignoring unknown or invalid config entry on line %u: %s\n", lineNo, line.c_str());
		}
	}

	return true;
}

static std::unique_ptr<TCNProcessor> makeProcessor(const ExperimentConfig& config)
{
	if(config.executionMode == ExecutionMode::FusedStreaming) {
		if(config.historyMode == HistoryMode::Shift) {
			return std::unique_ptr<TCNProcessor>(new FusedStreamingProcessor<ShiftChannelState>());
		}
		return std::unique_ptr<TCNProcessor>(new FusedStreamingProcessor<CircularChannelState>());
	}

	if(config.historyMode == HistoryMode::Shift) {
		return std::unique_ptr<TCNProcessor>(new LayeredBlockProcessor<ShiftChannelState>(config.scratchMode));
	}
	return std::unique_ptr<TCNProcessor>(new LayeredBlockProcessor<CircularChannelState>(config.scratchMode));
}

static std::unique_ptr<TCNProcessor> makeProcessor()
{
	return makeProcessor(gExperimentConfig);
}

static void printExperimentSummary(unsigned int callbackFrames)
{
	rt_printf("\n--- Experiment config ---\n");
	rt_printf("Input file: %s\n", gAppConfig.inputFile.c_str());
	rt_printf("Execution mode: %s\n", executionModeName(gExperimentConfig.executionMode));
	rt_printf("History mode: %s\n", historyModeName(gExperimentConfig.historyMode));
	rt_printf("Scratch mode: %s\n", scratchModeName(gExperimentConfig.scratchMode));
	rt_printf(
		"Inference chunk frames: %u (%s)\n",
		gExperimentConfig.inferenceChunkFrames == 0 ? callbackFrames : gExperimentConfig.inferenceChunkFrames,
		gExperimentConfig.inferenceChunkFrames == 0 ? "callback-sized" : "explicit"
	);
}

} // namespace

bool setup(BelaContext *context, void *userData)
{
	if(!loadConfigFile()) {
		return false;
	}

	gSamples = AudioFileUtilities::load(gAppConfig.inputFile);
	if(gSamples.empty()) {
		rt_printf("Failed to load input file: %s\n", gAppConfig.inputFile.c_str());
		return false;
	}

	gOutput.resize(gSamples.size());
	for(size_t ch = 0; ch < gSamples.size(); ++ch) {
		gOutput[ch].resize(gSamples[ch].size(), 0.0f);
	}

	const uint64_t totalFrames = gSamples.empty() ? 0 : (uint64_t)gSamples[0].size();
	const uint64_t framesPerBlock = context->audioFrames ? (uint64_t)context->audioFrames : 0;
	const uint64_t estimatedBlocks = framesPerBlock ? (totalFrames + framesPerBlock - 1) / framesPerBlock : 0;
	const double blockBudgetNs = (context->audioSampleRate > 0.0 && context->audioFrames > 0)
		? ((double)context->audioFrames / context->audioSampleRate) * 1e9
		: 0.0;
	gReadPtr = 0;

	if(gSweepConfig.enabled) {
		// All processors are created and reset during setup. This keeps the sweep
		// from allocating or changing processor state on Bela's real-time thread.
		gBenchConfig.enabled = true;
		gBenchConfig.writeCsv = true;
		gAppConfig.writeOutputFile = false;
		gAppConfig.writeAudioToOutputs = false;
		gSweepProcessors.clear();
		gSweepProcessors.reserve(kSweepCaseCount);
		gSweepStates.resize(kSweepCaseCount);

		for(size_t i = 0; i < kSweepCaseCount; ++i) {
			ExperimentConfig config;
			config.executionMode = kSweepCases[i].executionMode;
			config.historyMode = kSweepCases[i].historyMode;
			config.scratchMode = kSweepCases[i].scratchMode;
			config.inferenceChunkFrames = kSweepCases[i].inferenceChunkFrames;
			std::unique_ptr<TCNProcessor> processor = makeProcessor(config);
			if(!processor) {
				rt_printf("Failed to create sweep processor %s\n", kSweepCases[i].name);
				return false;
			}
			const unsigned int chunkFrames = config.inferenceChunkFrames == 0
				? context->audioFrames
				: config.inferenceChunkFrames;
			processor->reset(gSamples.size(), chunkFrames);
			gSweepProcessors.push_back(std::move(processor));
			resetBenchmarkState(gSweepStates[i], estimatedBlocks, blockBudgetNs);
		}
		rt_printf("\n--- Sweep enabled: %zu cases ---\n", kSweepCaseCount);
		rt_printf("Sweep CSV prefix: %s\n", gSweepConfig.csvPrefix.c_str());
	} else {
		gProcessor = makeProcessor();
		if(!gProcessor) {
			rt_printf("Failed to create processor\n");
			return false;
		}
		const unsigned int chunkFrames = (gExperimentConfig.inferenceChunkFrames == 0)
			? context->audioFrames
			: gExperimentConfig.inferenceChunkFrames;
		gProcessor->reset(gSamples.size(), chunkFrames);
		resetBenchmarkState(gBenchState, estimatedBlocks, blockBudgetNs);
		printExperimentSummary(context->audioFrames);
	}

	return true;
}

void render(BelaContext *context, void *userData)
{
	if(gSamples.empty() || gSamples[0].empty()) {
		if(gRuntimeConfig.stopWhenInputEnds) {
			Bela_requestStop();
		}
		return;
	}

	if(gSweepConfig.enabled) {
		const unsigned int nFramesTotal = (unsigned int)gSamples[0].size();
		const unsigned int startFrame = gReadPtr;
		unsigned int framesProcessed = 0;

		for(size_t i = 0; i < kSweepCaseCount; ++i) {
			const unsigned int chunkFrames = kSweepCases[i].inferenceChunkFrames == 0
				? context->audioFrames
				: kSweepCases[i].inferenceChunkFrames;
			unsigned int caseFramesProcessed = 0;
			const uint64_t blockStart = nsNow();
			const uint64_t modelStart = nsNow();
			while(caseFramesProcessed < context->audioFrames && startFrame + caseFramesProcessed < nFramesTotal) {
				const unsigned int framesLeftInCallback = context->audioFrames - caseFramesProcessed;
				const unsigned int framesLeftInInput = nFramesTotal - startFrame - caseFramesProcessed;
				const unsigned int currentChunk = std::min(chunkFrames, std::min(framesLeftInCallback, framesLeftInInput));
				gSweepProcessors[i]->processChunk(
					gSamples,
					gOutput,
					startFrame + caseFramesProcessed,
					currentChunk,
					gAppConfig.inputGain,
					gAppConfig.outputGain,
					gAppConfig.clipOutput
				);
				caseFramesProcessed += currentChunk;
			}
			const uint64_t modelEnd = nsNow();
			const uint64_t blockEnd = nsNow();
			recordBenchmarkBlock(gSweepStates[i], modelEnd - modelStart, blockEnd - blockStart, caseFramesProcessed);
			framesProcessed = caseFramesProcessed;
		}

		gReadPtr += framesProcessed;
		if(gReadPtr >= nFramesTotal && gRuntimeConfig.stopWhenInputEnds) {
			Bela_requestStop();
		}
		return;
	}

	const unsigned int nFramesTotal = (unsigned int)gSamples[0].size();
	unsigned int framesProcessed = 0;
	const unsigned int chunkFrames = (gExperimentConfig.inferenceChunkFrames == 0)
		? context->audioFrames
		: gExperimentConfig.inferenceChunkFrames;

	const uint64_t blockStart = gBenchConfig.enabled ? nsNow() : 0;
	const uint64_t modelStart = gBenchConfig.enabled ? nsNow() : 0;

	while(framesProcessed < context->audioFrames && gReadPtr < nFramesTotal) {
		const unsigned int framesLeftInCallback = context->audioFrames - framesProcessed;
		const unsigned int framesLeftInInput = nFramesTotal - gReadPtr;
		const unsigned int currentChunk = std::min(chunkFrames, std::min(framesLeftInCallback, framesLeftInInput));

		gProcessor->processChunk(
			gSamples,
			gOutput,
			gReadPtr,
			currentChunk,
			gAppConfig.inputGain,
			gAppConfig.outputGain,
			gAppConfig.clipOutput
		);

		if(gAppConfig.writeAudioToOutputs) {
			for(unsigned int frame = 0; frame < currentChunk; ++frame) {
				for(unsigned int ch = 0; ch < gSamples.size() && ch < context->audioOutChannels; ++ch) {
					audioWrite(context, framesProcessed + frame, ch, gOutput[ch][gReadPtr + frame]);
				}
				for(unsigned int ch = (unsigned int)gSamples.size(); ch < context->audioOutChannels; ++ch) {
					audioWrite(context, framesProcessed + frame, ch, 0.0f);
				}
			}
		}

		gReadPtr += currentChunk;
		framesProcessed += currentChunk;
	}

	if(gBenchConfig.enabled) {
		const uint64_t modelEnd = nsNow();
		const uint64_t blockEnd = nsNow();
		recordBenchmarkBlock(gBenchState, modelEnd - modelStart, blockEnd - blockStart, framesProcessed);
	}

	if(gAppConfig.writeAudioToOutputs && framesProcessed < context->audioFrames) {
		zeroHardwareOutputs(context, framesProcessed);
	} else if(!gAppConfig.writeAudioToOutputs && framesProcessed == 0) {
		zeroHardwareOutputs(context, 0);
	}

	if(gReadPtr >= nFramesTotal && gRuntimeConfig.stopWhenInputEnds) {
		Bela_requestStop();
	}
}

static void writeBenchmarkCsv(
	const BenchmarkState& state,
	const ExperimentConfig& config,
	const std::string& csvFile,
	unsigned int callbackFrames
)
{
	if(!gBenchConfig.writeCsv) {
		return;
	}

	FILE* f = fopen(csvFile.c_str(), "w");
	if(!f) {
		rt_printf("WARNING: could not open %s for writing\n", csvFile.c_str());
		return;
	}

	// Keep benchmark data row-oriented and append configuration columns so the
	// CSV can be loaded directly by tools such as pandas.
	fprintf(
		f,
		"block_idx,model_ns,block_ns,frames_processed,overrun,warmup,"
		"execution_mode,history_mode,scratch_mode,inference_chunk_frames,effective_chunk_frames"
	);
	if(gBenchConfig.sampleSystemCpu) {
		fprintf(f, ",syscpu_x100,process_rss_kb");
	}
	fprintf(f, "\n");

	for(size_t i = 0; i < state.rows.size(); ++i) {
		const BenchmarkRow& row = state.rows[i];
		fprintf(
			f,
			"%zu,%u,%u,%u,%u,%u,%s,%s,%s,%u,%u",
			i,
			row.modelNs,
			row.blockNs,
			row.framesProcessed,
			(unsigned)row.overrun,
			(unsigned)row.warmup,
			executionModeName(config.executionMode),
			historyModeName(config.historyMode),
			scratchModeName(config.scratchMode),
			config.inferenceChunkFrames,
			config.inferenceChunkFrames == 0 ? callbackFrames : config.inferenceChunkFrames
		);
		if(gBenchConfig.sampleSystemCpu) {
			fprintf(f, ",%u,%u", (unsigned)row.sysCpu_x100, row.memRssKb);
		}
		fprintf(f, "\n");
	}

	fclose(f);
	rt_printf("Wrote benchmark CSV: %s (%zu rows)\n", csvFile.c_str(), state.rows.size());
}

void cleanup(BelaContext *context, void *userData)
{
	if(!gSweepConfig.enabled && gAppConfig.writeOutputFile) {
		AudioFileUtilities::write(gAppConfig.outputFile, gOutput, context->audioSampleRate);
		rt_printf("Wrote output audio: %s\n", gAppConfig.outputFile.c_str());
	}

	if(!gBenchConfig.enabled) {
		return;
	}

	if(gSweepConfig.enabled) {
		for(size_t i = 0; i < kSweepCaseCount; ++i) {
			ExperimentConfig config;
			config.executionMode = kSweepCases[i].executionMode;
			config.historyMode = kSweepCases[i].historyMode;
			config.scratchMode = kSweepCases[i].scratchMode;
			config.inferenceChunkFrames = kSweepCases[i].inferenceChunkFrames;
			const CompletedStats modelStats = computeCompletedStats(
				gSweepStates[i].modelNs, gSweepStates[i].model, gSweepStates[i].blockBudgetNs
			);
			const CompletedStats blockStats = computeCompletedStats(
				gSweepStates[i].blockNs, gSweepStates[i].block, gSweepStates[i].blockBudgetNs
			);
			rt_printf("\n--- Sweep case: %s ---\n", kSweepCases[i].name);
			printCompletedStats("Model", modelStats);
			printCompletedStats("Block", blockStats);
			printMonitorSummary(gSweepStates[i]);
			const std::string csvFile = gSweepConfig.csvPrefix + "_" + kSweepCases[i].name + ".csv";
			writeBenchmarkCsv(gSweepStates[i], config, csvFile, context->audioFrames);
		}
		return;
	}

	const CompletedStats modelStats = computeCompletedStats(
		gBenchState.modelNs, gBenchState.model, gBenchState.blockBudgetNs
	);
	const CompletedStats blockStats = computeCompletedStats(
		gBenchState.blockNs, gBenchState.block, gBenchState.blockBudgetNs
	);

	rt_printf("\n--- Benchmark summary ---\n");
	rt_printf("Processor: %s\n", gProcessor ? gProcessor->name() : "none");
	rt_printf("Blocks seen: %" PRIu64 "\n", gBenchState.blocksSeen);
	rt_printf("Measured blocks: %" PRIu64 "\n", gBenchState.block.count);
	rt_printf("Warmup blocks skipped: %" PRIu64 "\n", gBenchConfig.warmupBlocks);
	rt_printf("Block budget: %.0f ns (%.3f ms)\n", gBenchState.blockBudgetNs, gBenchState.blockBudgetNs / 1e6);
	printCompletedStats("Model", modelStats);
	printCompletedStats("Block", blockStats);
	printMonitorSummary(gBenchState);
	writeBenchmarkCsv(gBenchState, gExperimentConfig, gBenchConfig.csvFile, context->audioFrames);
}
