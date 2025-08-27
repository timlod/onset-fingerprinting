#include "AmplitudeOnsetDetector.h"
#include <cmath>

namespace onset
{

EnvelopeFollower::EnvelopeFollower(int numChannels, float attack, float release, float floor)
    : attackCoeff(1.0f / attack),
      releaseCoeff(1.0f / release),
      state(1, numChannels)
{
    for (int ch = 0; ch < numChannels; ++ch)
        state.setSample(0, ch, floor);
}

void EnvelopeFollower::process(const juce::AudioBuffer<float>& in, juce::AudioBuffer<float>& out)
{
    auto numSamples = in.getNumSamples();
    auto numChannels = in.getNumChannels();
    out.setSize(numChannels, numSamples, false, false, true);

    for (int ch = 0; ch < numChannels; ++ch)
    {
        float y = state.getSample(0, ch);
        const float* x = in.getReadPointer(ch);
        float* o = out.getWritePointer(ch);
        for (int i = 0; i < numSamples; ++i)
        {
            float coeff = x[i] > y ? attackCoeff : releaseCoeff;
            y += coeff * (x[i] - y);
            o[i] = y;
        }
        state.setSample(0, ch, y);
    }
}

MinMaxTracker::MinMaxTracker(int numChannels, float amin, float amax)
    : alphaMin(amin), alphaMax(amax),
      minVal(numChannels, 0.0f), maxVal(numChannels, 0.0f)
{
}

void MinMaxTracker::process(const juce::AudioBuffer<float>& in)
{
    auto numSamples = in.getNumSamples();
    auto numChannels = in.getNumChannels();
    for (int ch = 0; ch < numChannels; ++ch)
    {
        auto* x = in.getReadPointer(ch);
        float mn = minVal[ch];
        float mx = maxVal[ch];
        for (int i = 0; i < numSamples; ++i)
        {
            float v = x[i];
            mn += alphaMin * (v - mn);
            if (mn > v) mn = v;
            mx += alphaMax * (v - mx);
            if (mx < v) mx = v;
        }
        minVal[ch] = mn;
        maxVal[ch] = mx;
    }
}

AmplitudeOnsetDetector::AmplitudeOnsetDetector(int numChannels,
                                               int blockSz,
                                               float fl,
                                               float hipassFreq,
                                               std::pair<float, float> fastAR,
                                               std::pair<float, float> slowAR,
                                               float onThresh,
                                               float offThresh,
                                               int cool,
                                               bool useBacktrack,
                                               int backtrackBufferSize,
                                               int backtrackSmoothSize,
                                               float sr)
    : nChannels(numChannels),
      blockSize(blockSz),
      floor(fl),
      onThresholdValue(onThresh),
      offThresholdValue(offThresh),
      cooldown(cool),
      manual(onThresh > 1.0f),
      backtrack(useBacktrack),
      sampleRate(sr),
      fastSlide(numChannels, fastAR.first, fastAR.second, fl),
      slowSlide(numChannels, slowAR.first, slowAR.second, fl),
      minmax(numChannels, 1.0e-4f, 1.0e-5f),
      state(numChannels, 0),
      prevValues(numChannels, 0.0f),
      debounce(numChannels, 0)
{
    if (hipassFreq > 0.0f)
    {
        hipass = std::make_unique<juce::dsp::ProcessorDuplicator<
            juce::dsp::IIR::Filter<float>,
            juce::dsp::IIR::Coefficients<float>>>();
        auto coeff = juce::dsp::IIR::Coefficients<float>::makeHighPass(sr, hipassFreq);
        hipass->state = *coeff;
    }

    if (backtrack)
    {
        backtrackBuffer.setSize(numChannels, backtrackBufferSize);
        backtrackBuffer.clear();
        bAlpha = 2.0f / (backtrackSmoothSize + 1.0f);
        bTol = std::pow(1.0f - bAlpha, (float)backtrackBufferSize);
    }
}

void AmplitudeOnsetDetector::processBlock(const juce::AudioBuffer<float>& input,
                                          std::vector<int>& channels,
                                          std::vector<int>& deltas,
                                          juce::AudioBuffer<float>* relativeEnvelope)
{
    jassert(input.getNumSamples() == blockSize);
    juce::AudioBuffer<float> x(input);
    if (hipass)
    {
        juce::dsp::AudioBlock<float> block(x);
        juce::dsp::ProcessContextReplacing<float> ctx(block);
        hipass->process(ctx);
    }

    // Convert to dB and clip
    for (int ch = 0; ch < nChannels; ++ch)
    {
        float* data = x.getWritePointer(ch);
        for (int i = 0; i < blockSize; ++i)
        {
            float v = std::abs(data[i]) + 1.0e-10f;
            v = 20.0f * std::log10(v);
            if (v < floor)
                v = floor;
            data[i] = v;
        }
    }

    juce::AudioBuffer<float> fast(blockSize, nChannels);
    juce::AudioBuffer<float> slow(blockSize, nChannels);
    fastSlide.process(x, fast);
    slowSlide.process(x, slow);

    juce::AudioBuffer<float> rel(blockSize, nChannels);
    for (int ch = 0; ch < nChannels; ++ch)
    {
        const float* f = fast.getReadPointer(ch);
        const float* s = slow.getReadPointer(ch);
        float* r = rel.getWritePointer(ch);
        for (int i = 0; i < blockSize; ++i)
        {
            float diff = f[i] - s[i];
            float amp = std::pow(10.0f, diff / 20.0f) - 1.0e-10f;
            if (amp < 0.0f) amp = 0.0f;
            if (amp > -floor) amp = -floor;
            r[i] = amp;
        }
    }

    if (backtrack)
    {
        for (int ch = 0; ch < nChannels; ++ch)
        {
            const float* r = rel.getReadPointer(ch);
            for (int i = 0; i < blockSize; ++i)
                backtrackBuffer.setSample(ch, (backtrackIndex + i) % backtrackBuffer.getNumSamples(), r[i]);
        }
        backtrackIndex = (backtrackIndex + blockSize) % backtrackBuffer.getNumSamples();
    }

    std::vector<float> onThresh(nChannels), offThresh(nChannels);
    if (manual)
    {
        std::fill(onThresh.begin(), onThresh.end(), onThresholdValue);
        std::fill(offThresh.begin(), offThresh.end(), offThresholdValue);
    }
    else
    {
        minmax.process(rel);
        const auto& mi = minmax.getMin();
        const auto& ma = minmax.getMax();
        for (int ch = 0; ch < nChannels; ++ch)
        {
            onThresh[ch] = ma[ch] * onThresholdValue + mi[ch];
            offThresh[ch] = ma[ch] * offThresholdValue + mi[ch];
        }
    }

    std::vector<int> onIndex(nChannels, 0);
    std::vector<char> on(nChannels, 0);

    for (int ch = 0; ch < nChannels; ++ch)
    {
        const float* r = rel.getReadPointer(ch);
        if (debounce[ch] > 0)
            debounce[ch] -= blockSize;
        for (int i = 0; i < blockSize; ++i)
        {
            if (!state[ch] && debounce[ch] < 1 && r[i] > onThresh[ch] &&
                ((i == 0 ? prevValues[ch] : r[i - 1]) <= onThresh[ch]))
            {
                on[ch] = 1;
                onIndex[ch] = i;
                state[ch] = 1;
                debounce[ch] = cooldown;
                break;
            }
        }
        // turn off after off threshold
        if (state[ch])
        {
            for (int i = onIndex[ch]; i < blockSize; ++i)
            {
                if (r[i] < offThresh[ch])
                {
                    state[ch] = 0;
                    break;
                }
            }
        }
        prevValues[ch] = r[blockSize - 1];
    }

    for (int ch = 0; ch < nChannels; ++ch)
    {
        if (on[ch])
        {
            channels.push_back(ch);
            deltas.push_back(onIndex[ch]);
        }
    }

    if (backtrack && !channels.empty())
        backtrackOnsets(channels, deltas);

    if (relativeEnvelope)
        *relativeEnvelope = rel;
}

void AmplitudeOnsetDetector::backtrackOnsets(std::vector<int>& channels, std::vector<int>& deltas)
{
    int N = backtrackBuffer.getNumSamples();
    for (size_t j = 0; j < channels.size(); ++j)
    {
        int ch = channels[j];
        int delta = deltas[j];
        int i = blockSize - delta;
        auto readSample = [this, N, ch](int idx)
        {
            int pos = (backtrackIndex - idx + N) % N;
            return backtrackBuffer.getSample(ch, pos);
        };
        float currentSmoothed = readSample(i);
        i++;
        float prev = readSample(i);
        float prevSmoothed = bAlpha * prev + (1.0f - bAlpha) * currentSmoothed;
        while (currentSmoothed > prevSmoothed && std::abs(prevSmoothed - prev) > bTol && (i + 1 < N))
        {
            deltas[j] -= 1;
            i++;
            currentSmoothed = prevSmoothed;
            prev = readSample(i);
            prevSmoothed = bAlpha * prev + (1.0f - bAlpha) * currentSmoothed;
        }
    }
}

void AmplitudeOnsetDetector::initMinMaxTracker(const juce::AudioBuffer<float>& buffer)
{
    // Similar to processBlock but only updates the min/max tracker.
    juce::AudioBuffer<float> x(buffer);
    if (hipass)
    {
        juce::dsp::AudioBlock<float> block(x);
        juce::dsp::ProcessContextReplacing<float> ctx(block);
        hipass->process(ctx);
    }

    for (int ch = 0; ch < nChannels; ++ch)
    {
        float* data = x.getWritePointer(ch);
        for (int i = 0; i < x.getNumSamples(); ++i)
        {
            float v = std::abs(data[i]) + 1.0e-10f;
            v = 20.0f * std::log10(v);
            if (v < floor) v = floor;
            data[i] = v;
        }
    }

    juce::AudioBuffer<float> fastBuf(x.getNumChannels(), x.getNumSamples());
    juce::AudioBuffer<float> slowBuf(x.getNumChannels(), x.getNumSamples());
    fastSlide.process(x, fastBuf);
    slowSlide.process(x, slowBuf);

    juce::AudioBuffer<float> rel(x.getNumChannels(), x.getNumSamples());
    for (int ch = 0; ch < nChannels; ++ch)
    {
        const float* f = fastBuf.getReadPointer(ch);
        const float* s = slowBuf.getReadPointer(ch);
        float* r = rel.getWritePointer(ch);
        for (int i = 0; i < x.getNumSamples(); ++i)
        {
            float diff = f[i] - s[i];
            float amp = std::pow(10.0f, diff / 20.0f) - 1.0e-10f;
            if (amp < 0.0f) amp = 0.0f;
            if (amp > -floor) amp = -floor;
            r[i] = amp;
        }
    }
    minmax.process(rel);
}

} // namespace onset

