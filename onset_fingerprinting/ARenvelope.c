#include <stdio.h>

// gcc -shared -o ARenvelope.so -fPIC -Ofast ARenvelope.c

void process(float *x, float *y, float attack, float release, int size, int num_samples) {
    int index, prev_index;
    for (int j = 0; j < num_samples; ++j) {
        for (int i = 0; i < size; ++i) {
            index = j * size + i;
            prev_index = (j > 0) ? (j - 1) * size + i : (num_samples - 1) * size + i;

            float xi = x[index];
            float yi = y[prev_index];

            if (xi > yi) {
                y[index] = yi + attack * (xi - yi);
            } else {
                y[index] = yi + release * (xi - yi);
            }
        }
    }
}
