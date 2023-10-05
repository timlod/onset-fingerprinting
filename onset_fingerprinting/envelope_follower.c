#include <math.h>
#include <stdio.h>

// gcc -shared -o envelope_follower.so -fPIC -Ofast envelope_follower.c

void ar_envelope(float *x, float *y, float attack, float release, int size,
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

void minmax_envelope(float *x, float *min_val, float *max_val, float alpha_min,
            float alpha_max, float minmin, int n_samples, int n_channels) {
    int i, j, index;
    float xi, current_min, current_max, ialpha_min, ialpha_max;
    ialpha_min = 1.0 - alpha_min;
    ialpha_max = 1.0 - alpha_max;

    for (j = 0; j < n_channels; ++j) {
        current_min = min_val[j];
        current_max = max_val[j];
        for (i = 0; i < n_samples; ++i) {
            index = i * n_channels + j;
            xi = x[index];
            if (xi < minmin) {
                current_min = minmin;
            } else if (xi < current_min) {
                current_min = xi;
            } else {
                current_min = current_min * ialpha_min + xi * alpha_min;
            }

            if (xi > current_max) {
                current_max = xi;
            } else {
                current_max = current_max * ialpha_max + xi * alpha_max;
            }
        }
        min_val[j] = current_min;
        max_val[j] = current_max;
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
