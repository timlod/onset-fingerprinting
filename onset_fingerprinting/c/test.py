import numpy as np
import online_cc
from time import time

block_size = 64
n = 512
cc = online_cc.CrossCorrelation(n)

a = np.empty(10000, dtype=np.float32)
b = np.empty(10000, dtype=np.float32)


t = time()
for i in range(0, len(a), block_size):
    res = cc.update(block_size, a[i : i + block_size], b[i : i + block_size])

print(time() - t)
# print("CC (mine):", res)

t = time()
for i in range(n, len(a), block_size):
    out = np.correlate(
        a[i - (n - block_size) : i + block_size],
        b[i - (n - block_size) : i + block_size],
        mode="full",
    )

# print("CC (np):", out)
print(time() - t)
