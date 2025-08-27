# JUCE Amplitude Onset Detector

This folder contains a C++ port of the Python
`AmplitudeOnsetDetector` class found in `onset_fingerprinting/detection.py`.
The class is implemented as a JUCE-friendly module so it can be used directly
inside audio plug-ins or applications.

## Structure

- `AmplitudeOnsetDetector.h` – public API and helper classes.
- `AmplitudeOnsetDetector.cpp` – implementation of the onset detector.

## Using the module

Add the two source files to your JUCE project and include the header:

```cpp
#include "AmplitudeOnsetDetector.h"

onset::AmplitudeOnsetDetector detector (numChannels, blockSize);
```

The `processBlock` function expects a `juce::AudioBuffer<float>` containing
`blockSize` samples.  Detected channel indices and sample offsets are returned
in the supplied `channels` and `deltas` vectors.

## Porting notes

The original implementation makes heavy use of NumPy arrays and Python's
flexible typing.  Converting it to C++ required the following changes:

### Arrays and memory

* **Python**: `numpy.ndarray` objects store multi-channel audio and support
  vectorised operations.
* **C++**: JUCE's `AudioBuffer<float>` is used to hold sample blocks.  All
  operations are expressed as explicit loops over channels and samples.

### Envelope followers

* The Python version calls a compiled C extension for fast attack–release
  envelope following.
* In C++ the `EnvelopeFollower` class implements the same exponential
  smoothing in pure C++.  Each call processes one block and updates the per
  channel state.

### Min/Max tracking

* Python relies on a C library (`EMA_MinMaxTracker.so`) to maintain the recent
  range of the relative envelope.
* `MinMaxTracker` replicates this behaviour with simple exponential moving
  averages for the minimum and maximum values.

### High-pass filtering

* The Python detector designs a Butterworth filter using SciPy and stores the
  filter state between calls.
* JUCE's `dsp::IIR::Filter` and `dsp::ProcessorDuplicator` are used to create a
  high-pass filter that is applied in-place to each block.

### Control flow and state

* Python makes heavy use of boolean masking and broadcasting to find threshold
  crossings.  In C++ these checks are implemented explicitly with loops and
  per-channel state vectors (`state`, `prevValues`, `debounce`).
* Optional backtracking is implemented with a circular buffer and mirrors the
  logic from the Python version.

### Boilerplate

* C++ requires forward declarations, header guards (`#pragma once`) and
  explicit namespaces.  Constructors initialise members using an initialiser
  list and `std::vector` replaces Python lists.
* The JUCE module is self-contained and does not depend on any other files in
  the repository.

## Initialisation

Call `initMinMaxTracker` with a buffer containing representative audio before
processing real-time data.  This pre-fills the min/max tracker similarly to the
Python class' `init_minmax_tracker` method.

## Notes

This module focuses on clarity rather than maximal optimisation.  Further
performance improvements are possible by using SIMD operations or custom DSP
code where appropriate.

