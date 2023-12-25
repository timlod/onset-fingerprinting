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
    int sizem1;
} CircularArray;

/**
 * Initialize a CircularArray of a given size.
 *
 * @param cb Pointer to the CircularArray to initialize.
 * @param size Size of the array to allocate. Needs to be a power of 2.
 */
inline void init_circular_array(CircularArray *cb, int size) {
    cb->data = (float *)calloc(size, sizeof(float));
    cb->size = size;
    cb->start = 0;
    cb->sizem1 = size - 1;
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
    // int end = (cb->start + block_size) % cb->size;
    int end = (cb->start + block_size) & cb->sizem1;
    if (end < cb->start) {
        memcpy(cb->data + cb->start, new_data,
               (cb->size - cb->start) * sizeof(float));
        memcpy(cb->data, new_data + cb->size - cb->start, end * sizeof(float));
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
    // cb->start &= cb->sizem1;
}

/**
 * Write a single value into the CircularArray (size is power of 2).
 *
 * @param cb Pointer to the CircularArray.
 * @param new_data Float value to insert into the array.
 */
inline void write_circular_array_p2(CircularArray *cb, float new_data) {
    cb->data[cb->start++] = new_data;
    cb->start &= cb->sizem1;
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
    // int adjusted_index;
    //  C handles modulo of negative numbers differently than Python, hence
    //  the if to handle the negative case correctly.
    /* if (index < 0) { */
    /*     // adjusted_index = (cb->start + index + cb->size) % cb->size; */
    /*     adjusted_index = (cb->start + index + cb->size) & cb->sizem1; */
    /* } else { */
    /*     // adjusted_index = (cb->start + index) % cb->size; */
    /*     adjusted_index = (cb->start + index) & cb->sizem1; */
    /* } */
    int adjusted_index;
    adjusted_index = (cb->start + index) % cb->size;
    return cb->data[adjusted_index];
}

/**
 * Retrieve a value from the CircularArray at a specified index (size is power
   of 2).
 *
 * @param cb Pointer to the CircularArray.
 * @param index Index to retrieve the value from.
 * @return The float value at the specified index.
 */
inline float index_circular_array_p2(const CircularArray *cb, int index) {
    return cb->data[(cb->start + index) & cb->sizem1];
}
