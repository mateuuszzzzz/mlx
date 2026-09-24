// Copyright © 2023 Apple Inc.

#pragma once

#if defined __METAL__ || defined MLX_METAL_JIT
#define MTL_CONST constant
#else
#define MTL_CONST
#endif

static MTL_CONST constexpr int MAX_REDUCE_SPECIALIZED_DIMS = 4;
static MTL_CONST constexpr int REDUCE_N_READS = 4;
static MTL_CONST constexpr int REDUCE_N_WRITES = 4;
static MTL_CONST constexpr int SOFTMAX_N_READS = 4;
static MTL_CONST constexpr int RMS_N_READS = 4;
static MTL_CONST constexpr int RMS_LOOPED_LIMIT = 4096;
// Partition kernels: rows per threadgroup of the simdgroup kernel, the wide
// digit of the radix select at many threads per row, threads per group of
// the split-row kernels and the record a group leaves after the last pass,
// its histogram, the prefix over it and two counts
static MTL_CONST constexpr int SIMD_PARTITION_ROWS_PER_GROUP = 8;
static MTL_CONST constexpr int RADIX_PARTITION_WIDE_DIGIT_BITS = 11;
static MTL_CONST constexpr int SPLIT_PARTITION_THREADS = 256;
static MTL_CONST constexpr int SPLIT_PARTITION_RECORD_SIZE =
    2 * (1 << RADIX_PARTITION_WIDE_DIGIT_BITS) + 2;

// Instantiate a templated kernel.
// Extra args are used as template parameters:
// e.g. instantiate_kernel(binary_int, binary, a, b) ->
// [[host_name(binary_int)]] [kernel] binary<a, b>
#define instantiate_kernel(name, func, ...) \
  template [[host_name(                     \
      name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;
