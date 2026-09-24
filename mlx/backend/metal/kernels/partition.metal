// Copyright © 2026 Apple Inc.

#include <metal_stdlib>

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/partition.h"

#define instantiate_radix_partition_bn(tname, type, bn)        \
  instantiate_kernel("radix_partition_" #tname "_bn" #bn,      \
                     radix_partition, type, type, false, bn)   \
  instantiate_kernel("radix_argpartition_" #tname "_bn" #bn,   \
                     radix_partition, type, uint32_t, true, bn)

#define instantiate_radix_partition(tname, type)   \
  instantiate_radix_partition_bn(tname, type, 64)  \
  instantiate_radix_partition_bn(tname, type, 128) \
  instantiate_radix_partition_bn(tname, type, 256)

instantiate_radix_partition(uint8, uint8_t)
instantiate_radix_partition(uint16, uint16_t)
instantiate_radix_partition(uint32, uint32_t)
instantiate_radix_partition(uint64, uint64_t)
instantiate_radix_partition(int8, int8_t)
instantiate_radix_partition(int16, int16_t)
instantiate_radix_partition(int32, int32_t)
instantiate_radix_partition(int64, int64_t)
instantiate_radix_partition(float16, half)
instantiate_radix_partition(float32, float)
instantiate_radix_partition(bfloat16, bfloat16_t)

#define instantiate_simd_partition_n(tname, type, n)         \
  instantiate_kernel("simd_partition_" #tname "_n" #n,       \
                     simd_partition, type, type, false, n)   \
  instantiate_kernel("simd_argpartition_" #tname "_n" #n,    \
                     simd_partition, type, uint32_t, true, n)

#define instantiate_simd_partition(tname, type)   \
  instantiate_simd_partition_n(tname, type, 4)    \
  instantiate_simd_partition_n(tname, type, 8)    \
  instantiate_simd_partition_n(tname, type, 16)

instantiate_simd_partition(uint8, uint8_t)
instantiate_simd_partition(uint16, uint16_t)
instantiate_simd_partition(uint32, uint32_t)
instantiate_simd_partition(int8, int8_t)
instantiate_simd_partition(int16, int16_t)
instantiate_simd_partition(int32, int32_t)
instantiate_simd_partition(float16, half)
instantiate_simd_partition(float32, float)
instantiate_simd_partition(bfloat16, bfloat16_t)

#define instantiate_split_partition(tname, type)                       \
  instantiate_kernel("split_partition_histogram_" #tname,              \
                     split_partition_histogram, type)                  \
  instantiate_kernel("split_partition_" #tname,                        \
                     split_partition, type, type, false)               \
  instantiate_kernel("split_argpartition_" #tname,                     \
                     split_partition, type, uint32_t, true)

instantiate_split_partition(uint8, uint8_t)
instantiate_split_partition(uint16, uint16_t)
instantiate_split_partition(uint32, uint32_t)
instantiate_split_partition(uint64, uint64_t)
instantiate_split_partition(int8, int8_t)
instantiate_split_partition(int16, int16_t)
instantiate_split_partition(int32, int32_t)
instantiate_split_partition(int64, int64_t)
instantiate_split_partition(float16, half)
instantiate_split_partition(float32, float)
instantiate_split_partition(bfloat16, bfloat16_t) // clang-format on
