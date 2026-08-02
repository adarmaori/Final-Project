#include <Bela.h>
#include <cmath>
#include <cstdint>
#include <ctime>

// Generated from Bela/distortion_tcn_c4_l4_k5.pt.
// distortion TCN: 4 causal Tanh layers, 4 hidden channels,
// kernel 5, dilations {1, 2, 4, 8}.
namespace {
constexpr int kLayers = 4;
constexpr int kHidden = 4;
constexpr int kKernel = 5;
constexpr int kMaxHistory = 32;
constexpr int kMaxChannels = 2;
constexpr int kDilations[kLayers] = {1, 2, 4, 8};

static const float kWeights[kLayers][kHidden][kHidden][kKernel] = {
	{
		{
			{ 0.855707943f, 1.46667445f, 1.89066386f, 1.77176356f, 1.33743739f }
		},
		{
			{ 0.723808527f, 1.19102228f, 1.29076326f, 1.66805732f, 1.1745683f }
		},
		{
			{ 0.140954599f, 0.917004108f, 0.852920592f, 1.24106634f, 0.866287649f }
		},
		{
			{ -0.0118546057f, 1.42554021f, 2.76618457f, 2.84185338f, 1.74315417f }
		}
	},
	{
		{
			{ 0.0858607739f, -0.0993471742f, 0.196816653f, 0.164370283f, -0.248362988f },
			{ -0.043580588f, -0.0384883545f, 0.105753735f, -0.00523626851f, -0.797524989f },
			{ 0.0981224254f, 0.00615077512f, 0.113642268f, 0.227636397f, -0.253160983f },
			{ 0.0305374525f, 0.0994825363f, 0.669134796f, 0.747676015f, -1.19265401f }
		},
		{
			{ 0.0324399583f, -0.133992344f, -0.132910565f, 0.477805644f, 0.652603388f },
			{ -0.012240001f, -0.0976848826f, 0.145417795f, 0.621135294f, 0.951698959f },
			{ -0.199350804f, -0.178593263f, 0.0393565409f, 0.057287205f, 0.275951117f },
			{ -0.205530211f, -0.585445344f, -0.33378014f, 0.555756509f, 1.48886704f }
		},
		{
			{ 0.0706454888f, 0.292947292f, 0.340306342f, 0.478280663f, 0.65587759f },
			{ 0.152869642f, 0.332908899f, 0.56406641f, 0.598187268f, 0.783503056f },
			{ -0.0264923237f, -0.131138086f, 0.174978733f, 0.440012515f, 0.0967746824f },
			{ -0.306874305f, -0.219340831f, 0.345433414f, 0.794249237f, 0.22130093f }
		},
		{
			{ -0.0718175471f, 0.0641419068f, 0.35719645f, 0.635351777f, 0.366351753f },
			{ -0.0196205489f, 0.145770535f, 0.580339193f, 0.831655681f, 0.633998394f },
			{ 0.00475098938f, -0.0887624547f, 0.0611313134f, 0.346415639f, 0.22324042f },
			{ -0.339139551f, -0.376924902f, 0.23913683f, 0.81791985f, 0.511199415f }
		}
	},
	{
		{
			{ 0.0918368176f, -0.239184693f, 0.329920739f, -0.124025032f, -0.102233812f },
			{ 3.87430191e-05f, -0.00710535422f, -0.0168880131f, 0.245545536f, 1.10967183f },
			{ 0.234833166f, 0.0750755072f, 0.0659058988f, 0.451398164f, 0.530139983f },
			{ -0.234621927f, -0.0659435093f, -0.0603599697f, -0.0348950252f, 0.489663184f }
		},
		{
			{ -0.0568581447f, -0.219973654f, 0.320193499f, -0.666518986f, 0.622147262f },
			{ 0.0896070153f, -0.0196628794f, -0.228226885f, 0.417245418f, -0.0628923923f },
			{ -0.109855585f, 0.00283582974f, -0.0620495901f, 0.697493911f, 0.692665398f },
			{ 0.0839132369f, -0.0375892967f, 0.0898382813f, 0.260427147f, 0.278776973f }
		},
		{
			{ -0.185953066f, -0.135339096f, -0.149949521f, 0.249616534f, -0.284539133f },
			{ 0.0891986191f, -0.133666545f, -0.0677328706f, -0.222858965f, 0.201387376f },
			{ -0.697451115f, -0.127586454f, -0.430181742f, -0.662282467f, 0.378887117f },
			{ -0.140564531f, -0.0860389322f, 0.0567574762f, -0.132036343f, 0.415259212f }
		},
		{
			{ -0.0175191723f, 0.0910447538f, 0.137709394f, -0.165893823f, 0.213370368f },
			{ 0.0644023418f, 0.0702446476f, 0.0665652677f, 0.227321804f, -0.406294525f },
			{ 0.145430028f, 0.0352343023f, 0.527023554f, 0.0276233461f, -0.278596163f },
			{ 0.358495921f, -0.00669965101f, -0.0218841657f, 0.230913848f, 0.303934574f }
		}
	},
	{
		{
			{ 0.0677911639f, -0.114818603f, -0.206631944f, 0.0309182089f, 0.639596879f },
			{ -0.058557719f, -0.0119776241f, -0.0898817629f, 0.1664626f, 0.168925181f },
			{ -0.260519356f, 0.126470625f, 0.204917729f, 0.174404636f, 0.403492481f },
			{ 0.254842788f, 0.0903072059f, 0.152480721f, 0.114260159f, -0.119630933f }
		},
		{
			{ 0.0267634876f, 0.2174391f, 0.0976936147f, 0.0197102185f, -0.490455478f },
			{ 0.124629974f, 0.0898549482f, -0.0876813978f, -0.190679848f, -0.265655756f },
			{ -0.128709078f, -0.22376439f, -0.186241731f, 0.103677183f, -0.296341658f },
			{ 0.321675122f, -0.126840398f, -0.157176852f, -0.283385545f, 0.206315085f }
		},
		{
			{ 0.166148856f, -0.0276220422f, 0.0954028592f, 0.347360373f, -0.323726773f },
			{ -0.0453629643f, -0.0487158597f, 0.0490831845f, -0.0242954958f, -0.791559935f },
			{ -0.314182013f, 0.00768807111f, 0.0900780782f, -0.073946774f, 0.129363179f },
			{ -0.238167763f, -0.056568455f, -0.183866248f, 0.0176050328f, -0.254805565f }
		},
		{
			{ -0.064058803f, 0.0679525211f, -0.119543664f, 0.167668045f, -0.46305567f },
			{ 0.00756535353f, 0.115546763f, 0.048994638f, -0.0238877311f, -0.515089631f },
			{ -0.0203794669f, -0.0205921624f, -0.0637679771f, 0.049019888f, -0.176707104f },
			{ -0.175378993f, -0.0734188408f, 0.0375056639f, -0.277569592f, -0.0387656204f }
		}
	}
};
static const float kBiases[kLayers][kHidden] = {
	{ -0.176987931f, 0.149018884f, -0.107377782f, 0.00974338409f },
	{ 0.0926863775f, 0.0309681632f, 0.250195444f, -0.0560729727f },
	{ 0.00767934322f, -0.0513255484f, -0.14021945f, -0.0766234547f },
	{ -0.273521632f, 0.0993816629f, -0.0123260692f, 0.0800716206f }
};
static const float kFinalWeight[kHidden] = { 0.118685767f };
static const float kFinalBias = 0.0448959693f;

struct VoiceState {
    float history[kLayers][kMaxHistory][kHidden] = {};
    void reset() {
        for(int layer = 0; layer < kLayers; ++layer)
            for(int delay = 0; delay < kMaxHistory; ++delay)
                for(int channel = 0; channel < kHidden; ++channel)
                    history[layer][delay][channel] = 0.0f;
    }
};

VoiceState gState[kMaxChannels];
uint64_t gLatencyBlocks = 0;
uint64_t gLatencyTotalNs = 0;
uint64_t gLatencyMaxNs = 0;

static uint64_t nowNs()
{
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

static float processSample(float input, VoiceState& state)
{
    float layerInput[kHidden] = { input };
    float layerOutput[kHidden] = {};
    for(int layer = 0; layer < kLayers; ++layer) {
        const int inputChannels = layer == 0 ? 1 : kHidden;
        for(int out = 0; out < kHidden; ++out) {
            float value = kBiases[layer][out];
            for(int in = 0; in < inputChannels; ++in) {
                value += kWeights[layer][out][in][0] * layerInput[in];
                for(int tap = 1; tap < kKernel; ++tap)
                    value += kWeights[layer][out][in][tap] * state.history[layer][tap * kDilations[layer] - 1][in];
            }
            layerOutput[out] = std::tanh(value);
        }
        const int historyLength = (kKernel - 1) * kDilations[layer];
        for(int delay = historyLength - 1; delay > 0; --delay)
            for(int channel = 0; channel < inputChannels; ++channel)
                state.history[layer][delay][channel] = state.history[layer][delay - 1][channel];
        for(int channel = 0; channel < inputChannels; ++channel)
            state.history[layer][0][channel] = layerInput[channel];
        for(int channel = 0; channel < kHidden; ++channel)
            layerInput[channel] = layerOutput[channel];
    }
    float output = kFinalBias;
    for(int channel = 0; channel < kHidden; ++channel)
        output += kFinalWeight[channel] * layerInput[channel];
    return output;
}
}

bool setup(BelaContext*, void*)
{
    for(int channel = 0; channel < kMaxChannels; ++channel)
        gState[channel].reset();
    return true;
}

void render(BelaContext* context, void*)
{
    const uint64_t startNs = nowNs();
    const unsigned int channels = context->audioInChannels < kMaxChannels ? context->audioInChannels : kMaxChannels;
    for(unsigned int frame = 0; frame < context->audioFrames; ++frame) {
        float outputs[kMaxChannels] = {};
        if(channels == 0)
            outputs[0] = processSample(0.0f, gState[0]);
        for(unsigned int channel = 0; channel < channels; ++channel)
            outputs[channel] = processSample(audioRead(context, frame, channel), gState[channel]);
        for(unsigned int channel = 0; channel < context->audioOutChannels; ++channel) {
            const unsigned int inputChannel = channel < channels ? channel : 0;
            audioWrite(context, frame, channel, outputs[inputChannel]);
        }
    }
    const uint64_t elapsedNs = nowNs() - startNs;
    ++gLatencyBlocks;
    gLatencyTotalNs += elapsedNs;
    if(elapsedNs > gLatencyMaxNs)
        gLatencyMaxNs = elapsedNs;
}

void cleanup(BelaContext*, void*)
{
    const double averageNs = gLatencyBlocks > 0
        ? static_cast<double>(gLatencyTotalNs) / static_cast<double>(gLatencyBlocks)
        : 0.0;
    rt_printf("distortion average latency: %.0f ns (%.3f ms)\n", averageNs, averageNs / 1e6);
    rt_printf("distortion maximum latency: %llu ns (%.3f ms)\n",
              static_cast<unsigned long long>(gLatencyMaxNs),
              static_cast<double>(gLatencyMaxNs) / 1e6);
}
