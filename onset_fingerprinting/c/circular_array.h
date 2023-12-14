#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * A circular array structure for storing float values, wrapping around once a
   number of items is reached.
 */
typedef struct {
    float *data;
    int size;
    int start;
} CircularArray;


/**
 * Initialize a CircularArray of a given size.
 *
 * @param cb Pointer to the CircularArray to initialize.
 * @param size Size of the array to allocate.
 */
inline void init_circular_array(CircularArray *cb, int size) {
    cb->data = (float *)calloc(size, sizeof(float));
    cb->size = size;
    cb->start = 0;
}

inline void free_circular_array(CircularArray *cb) {
    free(cb->data);
    cb->data = NULL;
}


/**
 * Update the CircularArray with a block of new data.
 *
 * @param cb Pointer to the CircularArray to update.
 * @param new_data Pointer to the new float data to insert.
 * @param block_size Number of elements in new_data to insert.
 */
inline void write_circular_array_multi(CircularArray *cb, float *new_data,
                            int block_size) {
    int end = (cb->start + block_size) % cb->size;
    if (end < cb->start) {
        memcpy(cb->data + cb->start, new_data,
               (cb->size - cb->start) * sizeof(float));
        memcpy(cb->data, new_data + cb->size - cb->start, end *
sizeof(float));
    } else {
        memcpy(cb->data + cb->start, new_data, block_size * sizeof(float));
    }
    cb->start = end;
}


/**
 * Write a single value into the CircularArray.
 *
 * @param cb Pointer to the CircularArray.
 * @param new_data Float value to insert into the array.
 */
inline void write_circular_array(CircularArray *cb, float new_data) {
    cb->data[cb->start++] = new_data;
    if (cb->start == cb->size) {
        cb->start = 0;
    }
}


/**
 * Retrieve a value from the CircularArray at a specified index.
 *
 * @param cb Pointer to the CircularArray.
 * @param index Index to retrieve the value from. Can be negative for reverse
 *        indexing.
 * @return The float value at the specified index.
 */
inline float index_circular_array(const CircularArray *cb, int index) {
    int adjusted_index;
    // C handles modulo of negative numbers differently than Python, hence
    // the if to handle the negative case correctly.
    if (index < 0) {
        adjusted_index = (cb->start + index + cb->size) % cb->size;
    } else {
        adjusted_index = (cb->start + index) % cb->size;
    }
    return cb->data[adjusted_index];
}
