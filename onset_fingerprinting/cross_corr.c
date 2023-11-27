#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define min(a, b) ((a) < (b) ? (a) : (b))

typedef struct {
    float *data;
    int size;
    int start;
} CircularBuffer;

typedef struct {
    CircularBuffer *pyramid;
    CircularBuffer buffer1;
    CircularBuffer buffer2;
    int n;
} CrossCorrelationData;

void init_circular_buffer(CircularBuffer *cb, int size) {
    cb->data = (float *)calloc(size, sizeof(float));
    cb->size = size;
    cb->start = 0;
}

void update_circular_buffer(CircularBuffer *cb, float *new_data,
                            int block_size) {
    int end = (cb->start + block_size) % cb->size;
    if (end < cb->start) {
        memcpy(cb->data + cb->start, new_data,
               (cb->size - cb->start) * sizeof(float));
        memcpy(cb->data, new_data + cb->size - cb->start, end * sizeof(float));
    } else {
        memcpy(cb->data + cb->start, new_data, block_size * sizeof(float));
    }
    cb->start = end;
}

void write_cb_single(CircularBuffer *cb, float new_data) {
    int index = cb->start;
    cb->data[index] = new_data;
    cb->start = (index + 1) % cb->size;
}

float index_cb(const CircularBuffer *cb, int index) {
    int adjusted_index;
    // C handles modulo of negative numbers differently than Python, hence the
    // if to handle the negative case correctly.
    if (index < 0) {
        adjusted_index = (cb->start + index + cb->size) % cb->size;
    } else {
        adjusted_index = (cb->start + index) % cb->size;
    }
    return cb->data[adjusted_index];
}

void init_cross_correlation_data(CrossCorrelationData *ccd, int n) {
    ccd->n = n;

    // Initialize circular buffers for each channel
    init_circular_buffer(&ccd->buffer1, n);
    init_circular_buffer(&ccd->buffer2, n);

    // Initialize pyramid for cross-correlation
    int total_rows = 2 * n - 1;
    ccd->pyramid =
        (CircularBuffer *)malloc(total_rows * sizeof(CircularBuffer));
    for (int i = 0; i < total_rows; ++i) {
        int row_size = (i < n) ? (i + 1) : (2 * n - 1 - i);
        init_circular_buffer(&ccd->pyramid[i], row_size);
    }
}

float get_pyramid_value(CrossCorrelationData *ccd, int row, int index) {
    CircularBuffer *pyramid_row = &ccd->pyramid[row];
    int actual_index = (pyramid_row->start + index) % pyramid_row->size;
    return pyramid_row->data[actual_index];
}

void update_cross_correlation_data(CrossCorrelationData *ccd, float *a,
                                   float *b, int block_size) {
    int i, n, total_rows, lag, offset, row_index;
    n = ccd->n;
    total_rows = 2 * n - 1;
    CircularBuffer *current_row;

    for (i = 0; i < block_size; ++i) {
        row_index = 0;
        write_cb_single(&ccd->buffer1, a[i]);
        write_cb_single(&ccd->buffer2, b[i]);
        // lag here is equal to -(n - offset - 1), i.e. sliding the newest
        // element of b over a
        for (offset = 0; offset < ccd->n - 1; ++offset) {
            current_row = &ccd->pyramid[row_index++];
            write_cb_single(current_row,
                            index_cb(&ccd->buffer1, offset) * b[i]);
        }
        for (lag = 0; lag < ccd->n; ++lag) {
            current_row = &ccd->pyramid[row_index++];
            write_cb_single(current_row,
                            a[i] * index_cb(&ccd->buffer2, -lag - 1));
        }
    }
}

void calculate_cross_correlation(const CrossCorrelationData *ccd,
                                 float *cross_corr_result) {
    int total_rows = 2 * ccd->n - 1;

    for (int lag = 0; lag < total_rows; ++lag) {
        cross_corr_result[lag] = 0;
        CircularBuffer *current_row = &ccd->pyramid[lag];

        for (int i = 0; i < current_row->size; ++i) {
            cross_corr_result[lag] += current_row->data[i];
        }
    }
}
