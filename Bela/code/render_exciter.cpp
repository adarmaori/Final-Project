#include <Bela.h>
#include <cmath>
#include <cstdint>
#include <ctime>

// Generated from Bela/exciter_tcn_c8_l2_k5.pt.
// exciter TCN: 2 causal Tanh layers, 8 hidden channels,
// kernel 5, dilations {1, 2}.
namespace {
constexpr int kLayers = 2;
constexpr int kHidden = 8;
constexpr int kKernel = 5;
constexpr int kMaxHistory = 8;
constexpr int kMaxChannels = 2;
constexpr int kDilations[kLayers] = {1, 2};

static const float kWeights[kLayers][kHidden][kHidden][kKernel] = {
	{
		{
			{ -0.765082002f, -0.141004056f, 0.143365756f, 0.837108195f, 0.577519596f }
		},
		{
			{ 1.96397769f, 0.554611087f, -0.500915587f, -1.20214343f, -2.70717549f }
		},
		{
			{ -3.47435999f, -2.172611f, -0.536793351f, -0.252177507f, 0.316251189f }
		},
		{
			{ 3.9551363f, 3.19774461f, 1.36292255f, 0.210870042f, -4.81785107f }
		},
		{
			{ 1.65246034f, 0.57658273f, -0.883430958f, -1.10195184f, -0.40560016f }
		},
		{
			{ 0.715396583f, 0.559998751f, 0.214696959f, -0.271248758f, -0.0811872482f }
		},
		{
			{ -0.376034945f, -0.249342203f, 0.0669541061f, 0.812868357f, 1.45392478f }
		},
		{
			{ 0.715330303f, 0.275429547f, -0.297038317f, -0.149400368f, -1.13558936f }
		}
	},
	{
		{
			{ 0.117228776f, -0.0354299434f, -0.110624686f, -0.0883247554f, 0.143341705f },
			{ -0.544799447f, -0.202315778f, 0.150688618f, 0.762922525f, 0.123691916f },
			{ 0.302663058f, -0.182732731f, -0.0937218145f, -0.140904874f, 0.164891705f },
			{ -1.00760567f, 0.245233625f, 0.642388165f, 0.665633857f, -0.547450721f },
			{ -0.182622835f, 0.10783381f, -0.0362333767f, 0.140250474f, -0.0768640339f },
			{ 0.0909025148f, 0.0433024541f, 0.175234571f, 0.0196067188f, 0.147915125f },
			{ 0.138930768f, 0.122180872f, 0.159186468f, -0.0151355732f, -0.0504845157f },
			{ -0.0635527596f, -0.117858648f, 0.0892695263f, -0.0164494887f, 0.0365601219f }
		},
		{
			{ 0.0026241662f, 0.139012635f, -0.0672849566f, -0.129823893f, -0.0761640221f },
			{ -0.330190182f, 0.0304022636f, 0.0109711336f, -0.0401541628f, -1.18564737f },
			{ 0.282117397f, -0.0876551643f, 0.0680297762f, -0.0806742236f, -0.319202423f },
			{ -0.121977724f, 0.576922297f, 0.242957637f, 0.135565519f, -0.949358702f },
			{ -0.130724803f, 0.173750997f, -0.176416576f, -0.031324327f, 0.0208140574f },
			{ 0.112346277f, 0.0472098924f, 0.118154347f, -0.0169496834f, -0.136936158f },
			{ -0.0646566227f, 0.0942290425f, -0.0221666079f, 0.0889137238f, 0.0535735488f },
			{ -0.119682051f, 0.0567430332f, -0.129428908f, -0.0531542487f, -0.11552342f }
		},
		{
			{ -0.0575096346f, -0.0490627252f, -0.148897797f, 0.0311014913f, -0.100381784f },
			{ -0.999175489f, -0.278600037f, 0.464403182f, 1.1454078f, 0.215814516f },
			{ 0.412624121f, -0.423502475f, -0.454053551f, -0.0693974197f, 0.171533912f },
			{ -0.947661579f, 0.864798725f, 0.913022041f, 0.799434185f, -0.73451525f },
			{ -0.0286163688f, -0.0111932512f, -0.0260481406f, 0.0470964797f, 0.160183549f },
			{ -0.00311381673f, 0.149624854f, -0.0757328272f, 0.021382153f, -0.176448613f },
			{ 0.00825930014f, -0.0594951771f, -0.0378865749f, -0.149210781f, 0.0153372483f },
			{ -0.0954404697f, -0.0603014566f, -0.00724396249f, -0.0128014144f, -0.120315827f }
		},
		{
			{ 0.115596719f, 0.0864877254f, -0.126697332f, -0.109728046f, 0.118366696f },
			{ -0.306704551f, 0.648854852f, 0.527354181f, 0.151656568f, -2.11331844f },
			{ 0.065781191f, -0.256529689f, 0.164536491f, 0.343264401f, 0.117140435f },
			{ 0.275801182f, 1.07013512f, 0.529041409f, -0.261205226f, -1.61043811f },
			{ -0.0363453589f, 0.0686378703f, 0.0556434579f, 0.00502837496f, -0.0139775006f },
			{ -0.0156826805f, 0.16932863f, 0.136794403f, -0.0359951742f, -0.108604558f },
			{ 0.00509935571f, 0.140239313f, -0.142363399f, 0.082866028f, 0.193074673f },
			{ -0.126895726f, 0.0738756657f, 0.0419151671f, -0.102448128f, -0.0411255993f }
		},
		{
			{ -0.030736912f, 0.202921033f, 0.192694291f, 0.225677013f, 0.179380968f },
			{ 0.13942337f, 0.0448252223f, -0.221423835f, 0.158633351f, -0.389653236f },
			{ 0.140441507f, -0.104521722f, 0.120033145f, 0.0539899766f, -0.287770987f },
			{ -0.131971076f, 0.232669711f, -0.101011895f, 0.138402492f, -0.391757935f },
			{ 0.030313246f, 0.0353658237f, -0.0117962575f, -0.0248582214f, 0.151952058f },
			{ -0.115076214f, 0.076053746f, 0.0924638733f, 0.0598217323f, 0.148026407f },
			{ -0.022570258f, -0.0513670668f, -0.0627252087f, 0.0037601965f, 0.138150156f },
			{ -0.146891713f, -0.114401884f, -0.184476331f, -0.0978908166f, -0.138377056f }
		},
		{
			{ 0.00811797567f, 0.0284735411f, -0.100463755f, -0.0990469456f, 0.0741496533f },
			{ 0.566736162f, 0.0218448602f, -0.0912050903f, -0.0631156117f, 0.883070052f },
			{ -0.120034836f, 0.105418622f, 0.144615725f, -0.0969850719f, 0.0956060588f },
			{ 0.37285012f, -0.431271613f, -0.553028107f, -0.139764428f, 0.689498782f },
			{ 0.0619967319f, -0.0325207822f, -0.00619433448f, 0.144861326f, -0.130721286f },
			{ -0.0803234205f, 0.000579122803f, 0.0314471237f, 0.153503016f, -0.0111835059f },
			{ -0.170460239f, 0.0959581882f, 0.0639360994f, -0.0669967234f, -0.023612123f },
			{ -0.0629207119f, 0.105109408f, -0.0968318135f, 0.0812230036f, -0.0230572633f }
		},
		{
			{ -0.0341110155f, 0.0594315603f, 0.113323383f, -0.0805664882f, -0.102121092f },
			{ 0.070007138f, 0.0927046612f, 0.339608371f, -0.00603535818f, 0.26517272f },
			{ 0.0630779415f, 0.116302125f, -0.233937144f, 0.123799771f, -0.100356147f },
			{ -0.127894089f, -0.0214501731f, 0.208054587f, -0.1965148f, 0.137412041f },
			{ -0.0250890665f, -0.0918649808f, 0.0829875171f, -0.0569086038f, 0.0384226441f },
			{ 0.133524328f, -0.0114373341f, 0.169471934f, 0.0454333574f, 0.133526906f },
			{ 0.0590769202f, -0.0508745946f, -0.0124192778f, 0.0318304524f, 0.018128477f },
			{ 0.0272854604f, -0.0484313183f, -0.0161445662f, -0.0419559181f, 0.111293934f }
		},
		{
			{ 0.138185516f, 0.102368906f, -0.0972593501f, 0.0190928467f, -0.0583053492f },
			{ -1.05177915f, -0.652755618f, -0.186052695f, 0.908144057f, 0.374690861f },
			{ 0.601445854f, -0.145510271f, -0.209129438f, -0.184499249f, 0.0562383048f },
			{ -1.3106463f, 0.427674204f, 0.587840974f, 0.967035055f, -0.621029973f },
			{ -0.359825373f, -0.222068831f, 0.0425751358f, 0.0576278791f, 0.307600856f },
			{ -0.0956074968f, -0.118520662f, 0.0279054064f, 0.0347256735f, 0.03449893f },
			{ 0.0352644362f, 0.10587728f, -0.0357870385f, -0.0130216982f, 0.00786365289f },
			{ -0.0489993431f, 0.132544369f, 0.113391101f, 0.0596044101f, 0.00253964565f }
		}
	}
};
static const float kBiases[kLayers][kHidden] = {
	{ -0.273782372f, -0.00879803859f, 0.00969549362f, 0.00210535922f, -0.0029031029f, -0.0458382182f, -0.00972019415f, 0.28866443f },
	{ 0.0563753508f, 0.011052331f, 0.0348014161f, 0.107392021f, -0.139384463f, -0.0134855676f, 0.00634139823f, -0.106549054f }
};
static const float kFinalWeight[kHidden] = { -0.00733053638f };
static const float kFinalBias = 0.128671989f;

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
    rt_printf("exciter average latency: %.0f ns (%.3f ms)\n", averageNs, averageNs / 1e6);
    rt_printf("exciter maximum latency: %llu ns (%.3f ms)\n",
              static_cast<unsigned long long>(gLatencyMaxNs),
              static_cast<double>(gLatencyMaxNs) / 1e6);
}
