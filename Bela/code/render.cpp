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
	{ "naive_control", ExecutionMode::LayeredBlock, HistoryMode::Shift, ScratchMode::PerChunk, 0 },
	{ "fused_only", ExecutionMode::FusedStreaming, HistoryMode::Shift, ScratchMode::PerChunk, 0 },
	{ "circular_only", ExecutionMode::LayeredBlock, HistoryMode::Circular, ScratchMode::PerChunk, 0 },
	{ "persistent_only", ExecutionMode::LayeredBlock, HistoryMode::Shift, ScratchMode::Persistent, 0 },
	{ "all_combined", ExecutionMode::FusedStreaming, HistoryMode::Circular, ScratchMode::Persistent, 0 }
};

constexpr size_t kSweepCaseCount = sizeof(kSweepCases) / sizeof(kSweepCases[0]);

constexpr int kNumConvLayers = 2;
constexpr int kHiddenChannels = 8;
constexpr int kKernelSize = 5;
constexpr int kMaxHistory = (kKernelSize - 1) * 2;

static const int kDilations[kNumConvLayers] = { 1, 2 };

static const float conv0_w[kHiddenChannels][1][kKernelSize] = {
    { { -0.7646027394f, -0.1384194614f, 0.1450108644f, 0.8371081715f, 0.5800434574f } },
    { { 1.961103469f, 0.5542248935f, -0.4902758673f, -1.193715155f, -2.707175441f } },
    { { -3.474359926f, -2.161216017f, -0.5471432954f, -0.2462144829f, 0.3282859772f } },
    { { 3.945326775f, 3.186610088f, 1.365690038f, 0.2276150063f, -4.817850966f } },
    { { 1.652460294f, 0.5725059286f, -0.8847818896f, -1.105977362f, -0.4033564497f } },
    { { 0.7153965952f, 0.5576713616f, 0.2140556742f, -0.2703861147f, -0.07886261679f } },
    { { -0.3777914867f, -0.2518609911f, 0.06868936121f, 0.8128241077f, 1.453924812f } },
    { { 0.7153318822f, 0.2771911044f, -0.2950744014f, -0.152008025f, -1.135589363f } }
};
static const float conv1_w[kHiddenChannels][kHiddenChannels][kKernelSize] = {
    {
        { 0.1190085383f, -0.03173561022f, -0.1110746358f, -0.0872729281f, 0.142810246f },
        { -0.5474392762f, -0.2062814664f, 0.1507441485f, 0.7616546452f, 0.1269424409f },
        { 0.3014882971f, -0.1824797587f, -0.09520683065f, -0.142810246f, 0.1666119536f },
        { -1.007605624f, 0.2459509792f, 0.6426461069f, 0.6664478146f, -0.5474392762f },
        { -0.1824797587f, 0.1110746358f, -0.03966951277f, 0.142810246f, -0.07933902554f },
        { 0.0872729281f, 0.03966951277f, 0.1745458562f, 0.01586780511f, 0.1507441485f },
        { 0.142810246f, 0.1190085383f, 0.1586780511f, -0.01586780511f, -0.04760341533f },
        { -0.06347122043f, -0.1190085383f, 0.0872729281f, -0.01586780511f, 0.03966951277f }
    },
    {
        { 0.0f, 0.1400370849f, -0.06535063963f, -0.1307012793f, -0.0746864453f },
        { -0.3267531982f, 0.02800741699f, 0.009335805662f, -0.03734322265f, -1.185647319f },
        { 0.2800741699f, -0.08402225096f, 0.06535063963f, -0.08402225096f, -0.3174173925f },
        { -0.1213654736f, 0.578819951f, 0.2427309472f, 0.1400370849f, -0.9522521775f },
        { -0.1307012793f, 0.1773803076f, -0.1773803076f, -0.02800741699f, 0.01867161132f },
        { 0.1120296679f, 0.04667902831f, 0.1213654736f, -0.01867161132f, -0.1400370849f },
        { -0.06535063963f, 0.09335805662f, -0.01867161132f, 0.09335805662f, 0.05601483397f },
        { -0.1213654736f, 0.05601483397f, -0.1307012793f, -0.05601483397f, -0.1120296679f }
    },
    {
        { -0.05411375687f, -0.04509479739f, -0.1533223111f, 0.02705687843f, -0.09920855425f },
        { -1.001104502f, -0.2795877438f, 0.4599669334f, 1.145407854f, 0.2164550275f },
        { 0.414872136f, -0.4238910954f, -0.4509479739f, -0.07215167582f, 0.1713602301f },
        { -0.9469907451f, 0.8658201098f, 0.9109149072f, 0.8026873935f, -0.7305357177f },
        { -0.02705687843f, -0.009018959478f, -0.02705687843f, 0.04509479739f, 0.1623412706f },
        { 0.0f, 0.1533223111f, -0.07215167582f, 0.01803791896f, -0.1803791896f },
        { 0.009018959478f, -0.06313271634f, -0.03607583791f, -0.1533223111f, 0.01803791896f },
        { -0.09920855425f, -0.06313271634f, -0.009018959478f, -0.009018959478f, -0.1172464732f }
    },
    {
        { 0.1164821126f, 0.08320150897f, -0.1331224144f, -0.1164821126f, 0.1164821126f },
        { -0.2995254323f, 0.64897177f, 0.5324896574f, 0.1497627161f, -2.113318328f },
        { 0.06656120718f, -0.2496045269f, 0.1664030179f, 0.3494463377f, 0.1164821126f },
        { 0.2828851305f, 1.064979315f, 0.5324896574f, -0.2662448287f, -1.614109274f },
        { -0.03328060359f, 0.06656120718f, 0.04992090538f, 0.0f, -0.01664030179f },
        { -0.01664030179f, 0.1664030179f, 0.1331224144f, -0.03328060359f, -0.1164821126f },
        { 0.0f, 0.1331224144f, -0.1497627161f, 0.08320150897f, 0.1996836215f },
        { -0.1331224144f, 0.06656120718f, 0.04992090538f, -0.09984181076f, -0.03328060359f }
    },
    {
        { -0.03084708238f, 0.2035907437f, 0.1912519108f, 0.2251837014f, 0.1789130778f },
        { 0.1388118707f, 0.04627062357f, -0.2220989931f, 0.1573201201f, -0.388673238f },
        { 0.1418965789f, -0.1048800801f, 0.1203036213f, 0.05552474828f, -0.2868778661f },
        { -0.1326424542f, 0.2313531179f, -0.1017953719f, 0.1388118707f, -0.3917579462f },
        { 0.03084708238f, 0.03393179062f, -0.01233883295f, -0.0246776659f, 0.1511507037f },
        { -0.1141342048f, 0.07711770595f, 0.09254124714f, 0.05860945652f, 0.1480659954f },
        { -0.02159295767f, -0.05244004005f, -0.06169416476f, 0.003084708238f, 0.1388118707f },
        { -0.1480659954f, -0.1141342048f, -0.1850824943f, -0.09871066362f, -0.1388118707f }
    },
    {
        { 0.006953307427f, 0.02781322971f, -0.09734630398f, -0.09734630398f, 0.0764863817f },
        { 0.5701712091f, 0.02085992228f, -0.09039299656f, -0.06257976685f, 0.8830700433f },
        { -0.1182062263f, 0.1042996114f, 0.146019456f, -0.09734630398f, 0.09734630398f },
        { 0.3754786011f, -0.4311050605f, -0.5562645942f, -0.1390661485f, 0.6883774353f },
        { 0.06257976685f, -0.03476653714f, -0.006953307427f, 0.146019456f, -0.1321128411f },
        { -0.08343968913f, 0.0f, 0.03476653714f, 0.1529727634f, -0.01390661485f },
        { -0.1738326857f, 0.09734630398f, 0.06257976685f, -0.06953307427f, -0.02085992228f },
        { -0.06257976685f, 0.1042996114f, -0.09734630398f, 0.08343968913f, -0.02085992228f }
    },
    {
        { -0.03476306074f, 0.0588297951f, 0.112311427f, -0.08022244787f, -0.1016151006f },
        { 0.06952612149f, 0.09359285585f, 0.3396083626f, -0.005348163191f, 0.264734078f },
        { 0.06417795829f, 0.1149855086f, -0.2326450988f, 0.1230077534f, -0.1016151006f },
        { -0.1283559166f, -0.02139265276f, 0.2085783645f, -0.1952079565f, 0.1363781614f },
        { -0.02406673436f, -0.09091877425f, 0.08289652946f, -0.05615571351f, 0.03743714234f },
        { 0.1337040798f, -0.01069632638f, 0.1684671405f, 0.04545938713f, 0.1337040798f },
        { 0.0588297951f, -0.05080755032f, -0.01337040798f, 0.03208897915f, 0.01871857117f },
        { 0.02674081596f, -0.04813346872f, -0.01604448957f, -0.04278530553f, 0.112311427f }
    },
    {
        { 0.1341606462f, 0.1032004971f, -0.0928804474f, 0.02064009942f, -0.06192029826f },
        { -1.05264507f, -0.6501631318f, -0.1857608948f, 0.9081643745f, 0.3715217896f },
        { 0.5985628832f, -0.1444806959f, -0.2064009942f, -0.1857608948f, 0.05160024855f },
        { -1.310646313f, 0.4231220381f, 0.5882428335f, 0.9700846728f, -0.6192029826f },
        { -0.3612017399f, -0.2270410936f, 0.04128019884f, 0.06192029826f, 0.3096014913f },
        { -0.0928804474f, -0.1135205468f, 0.03096014913f, 0.03096014913f, 0.03096014913f },
        { 0.03096014913f, 0.1032004971f, -0.03096014913f, -0.01032004971f, 0.01032004971f },
        { -0.05160024855f, 0.1341606462f, 0.1135205468f, 0.06192029826f, 0.0f }
    }
};
static const float conv_b[kNumConvLayers][kHiddenChannels] = {
    { -0.2737801567f, -0.008838878078f, 0.009667944312f, 0.002145026576f, -0.002881555312f, -0.04583922602f, -0.009709850046f, 0.2886511905f },
    { 0.0563939255f, 0.0110842365f, 0.03481889316f, 0.1074190703f, -0.1393902266f, -0.01350409294f, 0.006349779313f, -0.1065426633f },
};
static const float final_w[kHiddenChannels] = { -0.007946970873f, 0.09337690775f, 0.05165531067f, 0.1827803301f, 0.2423826116f, -0.01986742718f, 0.0f, 0.2523163252f };
static const float final_b = 0.1286791038f;
static const float conv_input_scales[kNumConvLayers] = { 0.004711962122f, 0.007862796464f };
static const float conv_output_scales[kNumConvLayers] = { 0.02852800512f, 0.04233029133f };
static const float tanh_output_scales[kNumConvLayers] = { 0.007862796464f, 0.007873678771f };
static const float final_output_scale = 0.005644102266f;


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

static inline float quantizeDequantize(float value, float scale)
{
	const float q = std::max(-128.0f, std::min(127.0f, std::round(value / scale)));
	return q * scale;
}

template <typename StateT>
static inline void computeLayer(
	int layer,
	const float* layerInput,
	int inputChannels,
	StateT& state,
	float* layerOutput
)
{
	for(int out = 0; out < kHiddenChannels; ++out) {
		float z = conv_b[layer][out];
		for(int in = 0; in < inputChannels; ++in) {
			const float* weights = layer == 0 ? conv0_w[out][in] : conv1_w[out][in];
			z += weights[0] * layerInput[in];
			for(int tap = 1; tap < kKernelSize; ++tap) {
				z += weights[tap]
					* state.previous(layer, in, tap * kDilations[layer]);
			}
		}
		const float quantizedConv = quantizeDequantize(z, conv_output_scales[layer]);
		layerOutput[out] = quantizeDequantize(std::tanh(quantizedConv), tanh_output_scales[layer]);
	}
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
				float layerInput[kHiddenChannels] = {};
				float layerOutput[kHiddenChannels] = {};
				layerInput[0] = quantizeDequantize(input[ch][startFrame + frame] * inputGain, conv_input_scales[0]);

				for(int layer = 0; layer < kNumConvLayers; ++layer) {
					const int inputChannels = layer == 0 ? 1 : kHiddenChannels;
					computeLayer(layer, layerInput, inputChannels, state, layerOutput);
					state.push(layer, layerInput, inputChannels);
					for(int channel = 0; channel < kHiddenChannels; ++channel) {
						layerInput[channel] = layerOutput[channel];
					}
				}

				float y = final_b;
				for(int channel = 0; channel < kHiddenChannels; ++channel) {
					y += final_w[channel] * layerInput[channel];
				}
				y = quantizeDequantize(y, final_output_scale);
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
			scratch_.assign((size_t)maxChunkFrames * (size_t)kNumConvLayers * (size_t)kHiddenChannels, 0.0f);
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

			for(int layer = 0; layer < kNumConvLayers; ++layer) {
				const int inputChannels = layer == 0 ? 1 : kHiddenChannels;
				for(unsigned int frame = 0; frame < frames; ++frame) {
					float layerInput[kHiddenChannels] = {};
					if(layer == 0) {
						layerInput[0] = quantizeDequantize(input[ch][startFrame + frame] * inputGain, conv_input_scales[0]);
					} else {
						const float* previous = &scratch[((size_t)(layer - 1) * frames + frame) * kHiddenChannels];
						for(int channel = 0; channel < kHiddenChannels; ++channel) {
							layerInput[channel] = previous[channel];
						}
					}
					float* current = &scratch[((size_t)layer * frames + frame) * kHiddenChannels];
					computeLayer(layer, layerInput, inputChannels, state, current);
					state.push(layer, layerInput, inputChannels);
				}
			}

			for(unsigned int frame = 0; frame < frames; ++frame) {
				const float* last = &scratch[((size_t)(kNumConvLayers - 1) * frames + frame) * kHiddenChannels];
				float y = final_b;
				for(int channel = 0; channel < kHiddenChannels; ++channel) {
					y += final_w[channel] * last[channel];
				}
				y = quantizeDequantize(y, final_output_scale);
				output[ch][startFrame + frame] = applyOutputPostprocess(y, outputGain, clipOutput);
			}
		}
	}

private:
	std::vector<float>& prepareScratch(std::vector<float>& localScratch, unsigned int frames)
	{
		const size_t required = (size_t)frames * (size_t)kNumConvLayers * (size_t)kHiddenChannels;
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
