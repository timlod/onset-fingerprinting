#pragma once

#include <juce_audio_basics/juce_audio_basics.h>
#include <juce_dsp/juce_dsp.h>
#include <vector>
#include <memory>

namespace onset
{

/** Simple attack / release envelope follower for multi-channel audio. */
class EnvelopeFollower
{
public:
    EnvelopeFollower(int numChannels, float attack, float release, float floor);

    /** Processes a block of samples and writes the envelope to \p out. */
    void process(const juce::AudioBuffer<float>& in, juce::AudioBuffer<float>& out);

private:
    float attackCoeff;
    float releaseCoeff;
    juce::AudioBuffer<float> state; // one sample per channel
};

/** Exponential moving min / max tracker for onset thresholds. */
class MinMaxTracker
{
public:
    MinMaxTracker(int numChannels, float alphaMin, float alphaMax);
    void process(const juce::AudioBuffer<float>& in);
    const std::vector<float>& getMin() const { return minVal; }
    const std::vector<float>& getMax() const { return maxVal; }

private:
    float alphaMin, alphaMax;
    std::vector<float> minVal, maxVal;
};

/**
    Port of the Python AmplitudeOnsetDetector to a JUCE friendly C++ class.

    The detector compares fast and slow envelope followers and flags an onset
    whenever their relative difference crosses a threshold.
*/
class AmplitudeOnsetDetector
{
public:
    AmplitudeOnsetDetector(int numChannels,
                           int blockSize,
                           float floor = -70.0f,
                           float hipassFreq = 2000.0f,
                           std::pair<float, float> fastAR = {3.0f, 383.0f},
                           std::pair<float, float> slowAR = {2205.0f, 2205.0f},
                           float onThreshold = 0.5f,
                           float offThreshold = 0.1f,
                           int cooldown = 1323,
                           bool backtrack = false,
                           int backtrackBufferSize = 80,
                           int backtrackSmoothSize = 5,
                           float sampleRate = 44100.0f);

    /**
        Process one block of audio.  The input buffer must contain
        `blockSize` samples.

        Detected channel indices and sample offsets are returned in \p channels
        and \p deltas respectively.
    */
    void processBlock(const juce::AudioBuffer<float>& input,
                      std::vector<int>& channels,
                      std::vector<int>& deltas,
                      juce::AudioBuffer<float>* relativeEnvelope = nullptr);

    /** Initialise the min/max tracker with an existing buffer that contains
        a representative slice of the incoming audio. */
    void initMinMaxTracker(const juce::AudioBuffer<float>& buffer);

private:
    int nChannels;
    int blockSize;
    float floor;
    float onThresholdValue;
    float offThresholdValue;
    int cooldown;
    bool manual;
    bool backtrack;
    float sampleRate;

    std::unique_ptr<juce::dsp::ProcessorDuplicator<
        juce::dsp::IIR::Filter<float>,
        juce::dsp::IIR::Coefficients<float>>> hipass;

    EnvelopeFollower fastSlide;
    EnvelopeFollower slowSlide;
    MinMaxTracker minmax;

    std::vector<char> state;
    std::vector<float> prevValues;
    std::vector<int> debounce;

    juce::AudioBuffer<float> backtrackBuffer;
    int backtrackIndex = 0;
    float bAlpha = 0.0f;
    float bTol = 0.0f;

    void backtrackOnsets(std::vector<int>& channels, std::vector<int>& deltas);
};

} // namespace onset

