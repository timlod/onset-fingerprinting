#include "circular_array.h"
#include <Python.h>
#include <emmintrin.h>
#include <immintrin.h>
#include <numpy/arrayobject.h>
#include <stddef.h>
#include <stdlib.h>

#define ALIGN_SIZE 16

typedef struct {
    PyObject_HEAD int n;
    int block_size;
    CircularArray buffer1;
    CircularArray buffer2;
    PyObject *output_array;
    float *result_data;
    int row_updates;
    // Cumulative sums of rows/lags in center block
    float *cumsum;
    // Right partial sum of last iteration for each row/lag in center block
    float *last_sum;
    // Offsets
    int *offsets;
    // Circular arrays for total sums of each row/lag in center block
    // - think of these as partial dot-products
    CircularArray *block_sums;
} CrossCorrelation;

static PyTypeObject CrossCorrelationType;

/*
TODO: add normalizing and, if adding normalizing, get min_samples required to
have a result (will introduce latency)
*/

// The following two functions are from this SO question:
// https://stackoverflow.com/questions/19494114/parallel-prefix-cumulative-sum-with-sse
// Thank you!!!
/**
  Performs a cumulative sum over the input vector using SSE SIMD instructions.
*/
inline __m128 scan_SSE(__m128 x) {
    x = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 4)));
    x = _mm_add_ps(x, _mm_castsi128_ps(_mm_slli_si128(_mm_castps_si128(x), 8)));
    return x;
}

/**
  Performs a cumulative sum over the input vector using AVX2 SIMD instructions.
*/
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
/**
   Update intermediate data given new buffers and compute the updated
   cross-correlation.
 */
void update_cross_correlation_data(CrossCorrelation *self, PyArrayObject *a,
                                   PyArrayObject *b) {
    int block_size, i, j, offset, lag, n, nmbs, bsm1;
    block_size = self->block_size;
    n = self->n;
    nmbs = n - block_size;
    bsm1 = block_size - 1;

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

    for (offset = 0; offset < block_size - 1; ++offset) {
        cs = 0;
        for (i = offset; i >= 0; --i) {
            cs += b1[offset - i] * b2[n - i - 1];
        }
        result_data[offset] = cs;
    }
    /* __m256 b_vec, data_vec, product, cs_vec, t0; */
    /* j = 0; */
    /* for (offset = block_size - 1; offset < n - 1; ++offset) { */
    /*     cs_vec = _mm256_setzero_ps(); */
    /*     for (i = 0; i < block_size; i += 8) { */
    /*         b_vec = _mm256_loadu_ps(&b1[offset - bsm1 + i]); */
    /*         data_vec = _mm256_load_ps(&data2[i]); */
    /*         product = _mm256_mul_ps(b_vec, data_vec); */
    /*         product = scan_AVX(product); */
    /*         product = _mm256_add_ps(product, cs_vec); */
    /*         _mm256_store_ps(&cumsum[j], product); */
    /*         t0 = _mm256_permute2f128_ps(product, product, 0x11); */
    /*         cs_vec = _mm256_permute_ps(t0, 0xff); */
    /*         j += 8; */
    /*     } */
    /* } */
    /* for (lag = 0; lag <= nmbs; ++lag) { */
    /*     cs_vec = _mm256_setzero_ps(); */
    /*     for (i = 0; i < block_size; i += 8) { */
    /*         b_vec = _mm256_loadu_ps(&b2[i + nmbs - lag]); */
    /*         data_vec = _mm256_load_ps(&data1[i]); */
    /*         product = _mm256_mul_ps(b_vec, data_vec); */
    /*         product = scan_AVX(product); */
    /*         product = _mm256_add_ps(product, cs_vec); */
    /*         _mm256_store_ps(&cumsum[j], product); */
    /*         t0 = _mm256_permute2f128_ps(product, product, 0x11); */
    /*         cs_vec = _mm256_permute_ps(t0, 0xff); */
    /*         j += 8; */
    /*     } */
    /* } */
    j = 0;
    __m128 cs_vec, data_vec, b_vec, product;
    for (offset = block_size - 1; offset < n - 1; ++offset) {
        cs_vec = _mm_setzero_ps();
        for (i = 0; i < block_size; i += 4) {
            b_vec = _mm_loadu_ps(&b1[offset - bsm1 + i]);
            data_vec = _mm_load_ps(&data2[i]);
            product = _mm_mul_ps(b_vec, data_vec);
            product = scan_SSE(product);
            product = _mm_add_ps(product, cs_vec);
            _mm_store_ps(&cumsum[j], product);
            cs_vec = _mm_shuffle_ps(product, product, _MM_SHUFFLE(3, 3, 3, 3));
            j += 4;
        }
    }
    for (lag = 0; lag <= nmbs; ++lag) {
        cs_vec = _mm_setzero_ps();
        for (i = 0; i < block_size; i += 4) {
            b_vec = _mm_loadu_ps(&b2[i + nmbs - lag]);
            data_vec = _mm_load_ps(&data1[i]);
            product = _mm_mul_ps(b_vec, data_vec);
            product = scan_SSE(product);
            product = _mm_add_ps(product, cs_vec);
            _mm_store_ps(&cumsum[j], product);
            cs_vec = _mm_shuffle_ps(product, product, _MM_SHUFFLE(3, 3, 3, 3));
            j += 4;
        }
    }
    for (lag = nmbs + 1; lag < n; ++lag) {
        cs = 0;
        for (i = 0; i < n - lag; ++i) {
            cs += b1[i + lag] * b2[i];
        }
        result_data[lag + n - 1] = cs;
    }

    // Center blocks' data is shifted by 1 index at each iteration, so we have
    // to account for 'mid-block' sums (when dropping off the old sum, it
    // contained data from 2 blocks)
    for (i = 0; i < self->row_updates; ++i) {
        current_row = &self->block_sums[i];
        offset = self->offsets[i];
        sum = cumsum[i * block_size + offset - 1];
        sum2 = cumsum[(i + 1) * block_size - 1] - sum;
        result_data[bsm1 + i] -=
            current_row->data[current_row->start] - (sum + sum2);
        write_circular_array(current_row, self->last_sum[i] + sum);
        // Push new right sum to block sums to subtract in next iteration
        self->last_sum[i] = sum2;
    }
}

static int CrossCorrelation_init(CrossCorrelation *self, PyObject *args,
                                 PyObject *kwds) {

    if (!PyArg_ParseTuple(args, "ii", &self->n, &self->block_size)) {
        return -1;
    }
    int n, block_size, i, total_rows, total_updates, row_size, row_updates;
    n = self->n;
    block_size = self->block_size;

    init_circular_array(&self->buffer1, n);
    init_circular_array(&self->buffer2, n);

    total_rows = 2 * n - 1;

    npy_intp dim[1] = {2 * n - 1};
    self->output_array = PyArray_ZEROS(1, dim, NPY_FLOAT, 0);
    self->result_data =
        (float *)PyArray_DATA((PyArrayObject *)self->output_array);

    // One dry-run to pre-compute all indices:
    total_updates = block_size + (2 * block_size * (n - block_size));
    row_updates = total_rows - 2 * (block_size - 1);
    self->row_updates = row_updates;

    int ret = posix_memalign((void **)&self->cumsum, ALIGN_SIZE,
                             total_updates * sizeof(float));
    if (ret != 0) {
        printf("Failed to aligned allocated data: %d!\n", ret);
    };

    self->block_sums =
        (CircularArray *)malloc(row_updates * sizeof(CircularArray));
    self->offsets = (int *)malloc(row_updates * sizeof(int));
    self->last_sum = (float *)calloc(row_updates, sizeof(float));
    for (i = block_size - 1; i < total_rows - block_size + 1; ++i) {
        row_size = (i < n) ? (i + 1) : (2 * n - 1 - i);
        self->offsets[i - block_size + 1] =
            block_size - (row_size % block_size);
        init_circular_array(&self->block_sums[i - block_size + 1],
                            row_size / (block_size));
    }
    return 0;
}

static void CrossCorrelation_dealloc(CrossCorrelation *self) {
    free(self->last_sum);
    free(self->cumsum);
    for (int i = 0; i < self->row_updates; ++i) {
        free_circular_array(&self->block_sums[i]);
    }
    free(self->block_sums);
    free(self->offsets);
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
        printf("Error in PyType_Ready type preparation.\n");
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
