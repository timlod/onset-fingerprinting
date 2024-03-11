import numpy as np
import online_cc
from time import time

block_size = 64
n = 256
cc = online_cc.CrossCorrelation(n, block_size)

n_samples = n * 10000
print(f"Sample size: {n_samples}.")
np.random.seed(0)
a = np.random.rand(n_samples).astype(np.float32) * 1.1
b = np.random.rand(n_samples).astype(np.float32) * 1.1
t = np.linspace(0, 10, n_samples)
f = 300
a = (np.sin(2 * np.pi * t * f) + 0.01 * np.random.rand(n_samples)).astype(
    np.float32
)
b = (
    np.sin(2 * np.pi * t * f + 0.5) + 0.01 * np.random.rand(n_samples)
).astype(np.float32)


t = time()
for i in range(n - block_size, len(a) - block_size + 1, block_size):
    out = np.correlate(
        a[i - (n - block_size) : i + block_size],
        b[i - (n - block_size) : i + block_size],
        mode="full",
    )

print(time() - t)
# print("CC (np):", out)

t = time()
for i in range(0, len(a) - block_size + 1, block_size):
    res = cc.update(a[i : i + block_size], b[i : i + block_size])

print(time() - t)
# # print("CC (mine):", res)

print((2 * n - 1) - sum(np.abs(out - res) < 0.001), "wrong.")
avg_error = np.mean(
    np.abs(out[block_size:-block_size] - res[block_size:-block_size])
)
print(f"Average error: {avg_error}")
# print(np.abs(out - res) < 0.01)
# print(np.abs(out - res))
