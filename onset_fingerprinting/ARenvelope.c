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
