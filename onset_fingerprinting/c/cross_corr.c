#include "circular_array.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stddef.h>
#include <stdlib.h>
#include <valgrind/callgrind.h>

#define min(a, b) ((a) < (b) ? (a) : (b))

typedef struct {
    PyObject_HEAD int n;
    CircularArray buffer1;
    CircularArray buffer2;
    CircularArray *pyramid;
    float *pyramid_data;
    PyObject *output_array;
} CrossCorrelation;

static PyTypeObject CrossCorrelationType;

void update_cross_correlation_data(CrossCorrelation *self, PyArrayObject *a,
                                   PyArrayObject *b, int block_size) {
    // CALLGRIND_START_INSTRUMENTATION;

    // There are 2 strategies here:
    // 1: Write individual values to circular arrays immediately
    // 2: Create an intermediate array on the stack just for row updates, and
    // then batch update those with memcpy after the fact

    int n, i, j, lag, offset, row_index, total_updates, updates_count, idx;
    float data;
    n = self->n;
    // block_size^2 would be going up and down all the way, then we have to
    // fill in with rows of block_size on both ends
    total_updates =
        block_size * block_size + (2 * block_size * (n - block_size));
    // printf("Making %d updates!\n", total_updates);
    float updates[total_updates];
    CircularArray *current_row;
    float *data1 = (float *)PyArray_DATA(a);
    float *data2 = (float *)PyArray_DATA(b);

    // Update buffers with new data
    write_circular_array_multi(&self->buffer1, data1, block_size);
    write_circular_array_multi(&self->buffer2, data2, block_size);

    // Compute new multiplications of first half of data
    j = 0;
    row_index = 0;
    for (offset = 0; offset < n - 1; ++offset) {
        current_row = &self->pyramid[row_index++];
        for (i = min(offset, block_size - 1); i >= 0; --i) {
            data = index_circular_array(&self->buffer1, offset - i) *
                   data2[block_size - i - 1];
            // updates[j++] = data;
            write_circular_array(current_row, data);
            //  manually inlined
            /* current_row->data[current_row->start++] = data; */
            /* if (current_row->start == current_row->size) { */
            /*     current_row->start = 0; */
            /* } */
        }
    }

    // Continuing from the last update of the first n-1 rows
    for (lag = 0; lag < n; ++lag) {
        current_row = &self->pyramid[row_index++];

        // Determine the number of elements to update based on lag and
        // block_size
        updates_count = lag <= n - block_size ? block_size : n - lag;

        for (i = 0; i < updates_count; ++i) {
            idx = i - updates_count + block_size;
            data = index_circular_array(&self->buffer2,
                                        n - block_size + idx - lag) *
                   data1[idx];
            // updates[j++] = data;
            write_circular_array(current_row, data);
            // manually inlined
            /* current_row->data[current_row->start++] = data; */
            /* if (current_row->start == current_row->size) { */
            /*     current_row->start = 0; */
            /* } */
        }
    }
    // Strategy 2: batch updates
    /* row_index = 0; */
    /* int start, len; */
    /* float *sub_array; */
    /* start = 0; */
    /* for (offset = 0; offset < self->n - 1; ++offset) { */
    /*     current_row = &self->pyramid[row_index++]; */
    /*     len = min(offset + 1, block_size); */
    /*     sub_array = &updates[start]; */
    /*     start += len; */
    /*     write_circular_array_multi(current_row, sub_array, len); */
    /* } */
    /* for (offset = n; offset >= 1; --offset) { */
    /*     current_row = &self->pyramid[row_index++]; */
    /*     len = min(offset, block_size); */
    /*     sub_array = &updates[start]; */
    /*     start += len; */
    /*     write_circular_array_multi(current_row, sub_array, len); */
    /* } */
    // CALLGRIND_STOP_INSTRUMENTATION;
}

// Strategy: Create 2 contiguous arrays - one for small buffers (<block_size)
// which needn't be circular and are updated directly, and another for
// >block_size circular buffers which is used to update the 'center' at the end
// as has been done with the updates array Actually we can just use the updates
// array as has been done and ignore the first/last block_size entries. In that
// case index batch updates from block_size until blocksize and adjust start
// accordingly (I think block_size*(block_size+1)/2)

static int CrossCorrelation_init(CrossCorrelation *self, PyObject *args,
                                 PyObject *kwds) {

    if (!PyArg_ParseTuple(args, "i", &self->n)) {
        return -1; // Return -1 to indicate error during initialization
    }
    int n, i, total_rows, row_size, total_size, current_offset;
    n = self->n;

    init_circular_array(&self->buffer1, n);
    init_circular_array(&self->buffer2, n);

    total_rows = 2 * n - 1;
    // Allocate each circular array individually - currently using big
    // contiguous block instead for everything
    /* self->pyramid = */
    /*     (CircularArray *)malloc(total_rows * sizeof(CircularArray)); */
    /* for (int i = 0; i < total_rows; ++i) { */
    /*     int row_size = (i < n) ? (i + 1) : (2 * n - 1 - i); */
    /*     init_circular_array(&self->pyramid[i], row_size, 64); */
    /* } */

    // Calculate total number of floats we need to store
    total_size = 0;
    for (int i = 0; i < total_rows; ++i) {
        total_size += (i < n) ? (i + 1) : (2 * n - 1 - i);
    }

    // Allocate big block of memory for the entire pyramid and the array of
    // CircularArrays
    self->pyramid_data = (float *)malloc(total_size * sizeof(float));
    self->pyramid = (CircularArray *)malloc(total_rows * sizeof(CircularArray));

    // Initialize each circular array
    current_offset = 0;
    for (i = 0; i < total_rows; ++i) {
        row_size = (i < n) ? (i + 1) : (2 * n - 1 - i);
        self->pyramid[i].data = &self->pyramid_data[current_offset];
        self->pyramid[i].size = row_size;
        self->pyramid[i].start = 0;
        current_offset += row_size;
    }

    npy_intp dim[1] = {2 * n - 1};
    self->output_array = PyArray_ZEROS(1, dim, NPY_FLOAT, 0);

    return 0;
}

static void CrossCorrelation_dealloc(CrossCorrelation *self) {
    printf("Deallocating CrossCorrelation\n");
    free(self->pyramid_data);
    /* for (int i = 0; i < 2 * self->n - 1; ++i) { */
    /*     free_circular_array(&self->pyramid[i]); */
    /* } */
    // Safe decrement
    Py_XDECREF(self->output_array);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

void calculate_cross_correlation(const CrossCorrelation *self) {
    int lag, i;
    float sum;
    float *result_data =
        (float *)PyArray_DATA((PyArrayObject *)self->output_array);
    int total_rows = 2 * self->n - 1;
    CircularArray *current_row;

    for (lag = 0; lag < total_rows; ++lag) {
        sum = 0;
        current_row = &self->pyramid[lag];
        for (i = 0; i < current_row->size; ++i) {
            sum += current_row->data[i];
        }
        result_data[lag] = sum;
    }
}

static PyObject *CrossCorrelation_update(CrossCorrelation *self,
                                         PyObject *args) {
    PyObject *array1_obj, *array2_obj;
    int blocksize;

    // Parse the blocksize and two 1D float arrays
    if (!PyArg_ParseTuple(args, "iO!O!", &blocksize, &PyArray_Type, &array1_obj,
                          &PyArray_Type, &array2_obj)) {
        return NULL;
    }

    PyArrayObject *a = (PyArrayObject *)array1_obj;
    PyArrayObject *b = (PyArrayObject *)array2_obj;

    update_cross_correlation_data(self, a, b, blocksize);
    // Not doing currently as the update itself is already bottlenecking
    // calculate_cross_correlation(self);
    Py_INCREF(self->output_array);
    return self->output_array;
}

static PyMethodDef CrossCorrelation_methods[] = {
    {"update", (PyCFunction)CrossCorrelation_update, METH_VARARGS,
     "Process two 1D float arrays and update internal state."},
    {NULL, NULL, 0, NULL} // Sentinel
};

static PyTypeObject CrossCorrelationType = {
    PyVarObject_HEAD_INIT(NULL, 0).tp_name = "online_cc.CrossCorrelation",
    .tp_doc = "Online cross correlation",
    .tp_basicsize = sizeof(CrossCorrelation),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc)CrossCorrelation_init,
    .tp_methods = CrossCorrelation_methods,
    .tp_dealloc = (destructor)CrossCorrelation_dealloc,
};

static PyModuleDef online_cc_module = {
    PyModuleDef_HEAD_INIT,
    "online_cc",
    NULL, // Module docstring
    -1,   // Size of the module's state
    NULL, // Module-level methods; none in this case
    NULL, // Process-level initialization of the module
    NULL, // Module deallocation function
    NULL, // Optional sub-interpreter initialization function
};

PyMODINIT_FUNC PyInit_online_cc(void) {
    PyObject *m;
    import_array(); // Required for NumPy API
    if (PyType_Ready(&CrossCorrelationType) < 0) {
        printf("error in type prep\n");
        return NULL;
    };
    m = PyModule_Create(&online_cc_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&CrossCorrelationType);
    if (PyModule_AddObject(m, "CrossCorrelation",
                           (PyObject *)&CrossCorrelationType) < 0) {
        Py_DECREF(&CrossCorrelationType);
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
