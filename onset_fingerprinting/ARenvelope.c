#include <math.h>
#include <stdio.h>

// gcc -shared -o ARenvelope.so -fPIC -Ofast ARenvelope.c

void process(float *x, float *y, float attack, float release, int size,
             int num_samples) {
    int i, j, index, prev_index;
    float xi, yi;
    for (j = 0; j < num_samples; ++j) {
        for (i = 0; i < size; ++i) {
            index = j * size + i;
            prev_index =
                (j > 0) ? (j - 1) * size + i : (num_samples - 1) * size + i;
            xi = x[index];
            yi = y[prev_index];

            if (xi > yi) {
                y[index] = yi + attack * (xi - yi);
            } else {
                y[index] = yi + release * (xi - yi);
            }
        }
    }
}

void backtrack_onsets(float *buffer, long *channels, long *deltas, float alpha,
                      float tol, long buffer_length, long n_onsets,
                      long n_channels, long block_size) {
    long channel, i, j, idx;
    float current_smoothed, prev, prev_smoothed;
    float omba = 1.0 - alpha;
    long N = buffer_length;
    for (j = 0; j < n_onsets; j++) {
        channel = channels[j];
        i = block_size - deltas[j];
        // flat index into 2D array, moving backwards
        idx = (N - i) * n_channels + channel;
        current_smoothed = buffer[idx];
        idx -= n_channels;
        prev = buffer[idx];
        prev_smoothed = alpha * prev + omba * current_smoothed;
        while ((current_smoothed > prev_smoothed) &&
               (fabsf(prev_smoothed - prev) > tol) && (i + 1 < N)) {
            deltas[j] -= 1;
            i += 1;
            idx -= n_channels;
            current_smoothed = prev_smoothed;
            prev = buffer[idx];
            prev_smoothed = alpha * prev + omba * current_smoothed;
        }
    }
}
