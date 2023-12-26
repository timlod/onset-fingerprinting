#include "circular_array.h"
#include <Python.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <numpy/arrayobject.h>
#include <stddef.h>
#include <stdlib.h>

#define min(a, b) ((a) < (b) ? (a) : (b))
#define ALIGN_SIZE 32

typedef struct {
    PyObject_HEAD int n;
    int block_size;
    CircularArray buffer1;
    CircularArray buffer2;
    PyObject *output_array;
    float *result_data;
    // intermediate storage
    int total_updates;
    int row_updates;
    int *circular_index;
    int *data_index;
    // Last partial sums
    float *last_sum;
    float *cumsum;
    int *offsets;
    // circular sum of blocks
    CircularArray *block_sums;
} CrossCorrelation;

static PyTypeObject CrossCorrelationType;

/**
TODO: add normalizing and, if adding normalizing, get min_samples required to
have a result (will introduce latency)
**/

// https://stackoverflow.com/questions/19494114/parallel-prefix-cumulative-sum-with-sse
// Thank you!!!
inline __m128 scan_SSE(__m128 x) {
    x = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 4)));
    x = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 8)));
    return x;
}

inline __m256 scan_AVX(__m256 x) {
    __m256 t0, t1;
    // shift1_AVX + add
    t0 = _mm256_permute_ps(x, _MM_SHUFFLE(2, 1, 0, 3));
    t1 = _mm256_permute2f128_ps(t0, t0, 41);
    x = _mm256_add_ps(x, _mm256_blend_ps(t0, t1, 0x11));
    // shift2_AVX + add
    t0 = _mm256_permute_ps(x, _MM_SHUFFLE(1, 0, 3, 2));
    t1 = _mm256_permute2f128_ps(t0, t0, 41);
    x = _mm256_add_ps(x, _mm256_blend_ps(t0, t1, 0x33));
    // shift3_AVX + add
    x = _mm256_add_ps(x, _mm256_permute2f128_ps(x, x, 41));
    return x;
}

void update_cross_correlation_data(CrossCorrelation *self, PyArrayObject *a,
                                   PyArrayObject *b) {
    int block_size, i, j, k, offset, total_updates, total_rows, lag, n;
    total_updates = self->total_updates;
    block_size = self->block_size;
    total_rows = 2 * self->n - 1;
    n = self->n;

    float *cumsum = self->cumsum;
    float cs, sum, sum2;
    CircularArray *current_row;
    float *data1 = (float *)PyArray_DATA(a);
    float *data2 = (float *)PyArray_DATA(b);
    float *result_data = self->result_data;

    // Update buffers with new data
    write_circular_array_multi(&self->buffer1, data1, block_size);
    float *b1 = rearrange_circular_array(&self->buffer1);
    write_circular_array_multi(&self->buffer2, data2, block_size);
    float *b2 = rearrange_circular_array(&self->buffer2);

    int bsm1 = block_size - 1;
    cs = 0;
    j = 0;
    for (offset = 0; offset < block_size; ++offset) {
        for (i = offset; i >= 0; --i) {
            cs += b1[offset - i] * b2[n - i - 1];
            cumsum[j++] = cs;
        }
    }
    for (; offset < n - 1; ++offset) {
        for (i = 0; i < block_size; ++i) {
            cs += b1[offset - bsm1 + i] * data2[i];
            cumsum[j++] = cs;
        }
    }
    int inter = n - block_size;
    for (lag = 0; lag <= inter; ++lag) {
        for (i = 0; i < block_size; ++i) {
            cs += b2[inter + i - lag] * data1[i];
            cumsum[j++] = cs;
        }
    }
    for (lag = n - block_size + 1; lag < n; ++lag) {
        for (i = 0; i < n - lag; ++i) {
            cs += b2[i] * b1[i + lag];
            cumsum[j++] = cs;
        }
        // cumsum[offset + lag] = cs;
    }

    k = 0;
    result_data[0] = cumsum[0];
    for (i = 2; i <= block_size; ++i) {
        k += i;
        result_data[i - 1] = cumsum[k] - cumsum[k - i];
    }
    // Center blocks' data is shifted by 1 index at each iteration, so we have
    // to account for 'mid-block' sums (when dropping off the old sum, it
    // contained data from 2 blocks)
    for (i = 0; i < self->row_updates; ++i) {
        current_row = &self->block_sums[i];
        offset = self->offsets[i];
        j = k;
        k += offset;
        sum = cumsum[k] - cumsum[j];
        j = k;
        k += block_size - offset;
        sum2 = cumsum[k] - cumsum[j];
        result_data[block_size + i] -=
            current_row->data[current_row->start] - (sum + sum2);
        write_circular_array(current_row, self->last_sum[i] + sum);
        // Push new sum to block sums
        self->last_sum[i] = sum2;
    }
    j = block_size;
    for (i = block_size + self->row_updates; i < total_rows; ++i) {
        k += j;
        result_data[i] = cumsum[k] - cumsum[k - j--];
    }
}

static int CrossCorrelation_init(CrossCorrelation *self, PyObject *args,
                                 PyObject *kwds) {

    if (!PyArg_ParseTuple(args, "ii", &self->n, &self->block_size)) {
        return -1; // Return -1 to indicate error during initialization
    }
    int n, block_size, i, j, offset, lag, total_rows, total_updates, row_size,
        row_updates, updates_count, idx, total_size;
    n = self->n;
    block_size = self->block_size;

    init_circular_array(&self->buffer1, n);
    init_circular_array(&self->buffer2, n);

    total_rows = 2 * n - 1;

    // Calculate total number of floats we need to store
    total_size = 0;
    for (i = 0; i < total_rows; ++i) {
        total_size += (i < n) ? (i + 1) : (2 * n - 1 - i);
    }

    npy_intp dim[1] = {2 * n - 1};
    self->output_array = PyArray_ZEROS(1, dim, NPY_FLOAT, 0);
    self->result_data =
        (float *)PyArray_DATA((PyArrayObject *)self->output_array);

    // One dry-run to pre-compute all indices:
    total_updates =
        block_size * block_size + (2 * block_size * (n - block_size));
    self->total_updates = total_updates;
    row_updates = (total_rows - 2 * block_size);
    self->row_updates = row_updates;

    posix_memalign((void **)&self->circular_index, ALIGN_SIZE,
                   total_updates * sizeof(int));
    posix_memalign((void **)&self->data_index, ALIGN_SIZE,
                   total_updates * sizeof(int));
    posix_memalign((void **)&self->cumsum, ALIGN_SIZE,
                   total_updates * sizeof(float));

    // Compute new multiplications of first half of data
    j = 0;
    for (offset = 0; offset < n - 1; ++offset) {
        for (i = min(offset, block_size - 1); i >= 0; --i) {
            self->circular_index[j] = offset - i;
            self->data_index[j++] = block_size - i - 1;
        }
    }
    // Continuing from the last update of the first n-1 rows
    for (lag = 0; lag < n; ++lag) {
        // Determine the number of elements to update based on lag and
        // block_size
        updates_count = lag <= n - block_size ? block_size : n - lag;

        for (i = 0; i < updates_count; ++i) {
            idx = i - updates_count + block_size;
            self->circular_index[j] = n - block_size + idx - lag;
            self->data_index[j++] = idx;
        }
    }

    self->block_sums =
        (CircularArray *)malloc(row_updates * sizeof(CircularArray));
    self->offsets = (int *)malloc(row_updates * sizeof(int));
    self->last_sum = (float *)calloc(row_updates, sizeof(float));
    for (i = block_size; i < total_rows - block_size; ++i) {
        row_size = (i < n) ? (i + 1) : (2 * n - 1 - i);
        self->offsets[i - block_size] = block_size - (row_size % block_size);
        init_circular_array(&self->block_sums[i - block_size],
                            row_size / block_size);
    }
    printf("End init\n");
    return 0;
}

static void CrossCorrelation_dealloc(CrossCorrelation *self) {
    printf("Deallocating CrossCorrelation\n");
    free(self->circular_index);
    free(self->data_index);
    free(self->last_sum);
    free(self->cumsum);
    for (int i = 0; i < self->row_updates; ++i) {
        free_circular_array(&self->block_sums[i]);
    }
    free(self->block_sums);
    free(self->offsets);
    // Safe decrement
    Py_XDECREF(self->output_array);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *CrossCorrelation_update(CrossCorrelation *self,
                                         PyObject *args) {
    PyObject *array1_obj, *array2_obj;

    // Parse the blocksize and two 1D float arrays
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &array1_obj,
                          &PyArray_Type, &array2_obj)) {
        return NULL;
    }

    PyArrayObject *a = (PyArrayObject *)array1_obj;
    PyArrayObject *b = (PyArrayObject *)array2_obj;

    update_cross_correlation_data(self, a, b);
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
