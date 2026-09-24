# Copyright © 2026 Apple Inc.

import mlx.core as mx
from time_utils import time_fn


def time_partition(shape, k, dtype):
    x = mx.random.normal(shape).astype(dtype)
    mx.eval(x)
    kth = shape[-1] - k
    name = f"{shape[0]}x{shape[1]} {dtype} k={k}"
    time_fn(mx.partition, x, kth, axis=-1, msg=f"partition {name}")
    time_fn(mx.argpartition, x, kth, axis=-1, msg=f"argpartition {name}")
    time_fn(mx.topk, x, k, axis=-1, msg=f"topk {name}")
    time_fn(mx.sort, x, axis=-1, msg=f"sort {name}")


def warmup():
    # Let the GPU clock settle before the first timing
    x = mx.random.normal((2048, 8192))
    for _ in range(50):
        mx.eval(mx.sort(x, axis=-1))


if __name__ == "__main__":
    warmup()
    # MoE routers, top-k over long rows and the DeepSeek sparse attention
    # indexer, which keeps 2048 of every context row
    cases = (
        ((4096, 256), 8),
        ((1024, 512), 32),
        ((1024, 1024), 32),
        ((2048, 4096), 32),
        ((2048, 8192), 32),
        ((4096, 8192), 2048),
        ((2048, 32768), 2048),
    )
    for dtype in (mx.float32, mx.bfloat16):
        for shape, k in cases:
            time_partition(shape, k, dtype)
