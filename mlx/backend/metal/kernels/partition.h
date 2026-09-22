// Copyright © 2026 Apple Inc.

// Radix select partition, one threadgroup per row:
// - `RadixLayout` sizes the threadgroup memory from the threads per row
// - `RadixKeyTraits` maps values to unsigned keys with the same order
// - `threadgroup_exclusive_sum` scans per thread counts over the threadgroup
// - `radix_select_row` finds the key of the k-th smallest element with one
//   8-bit histogram pass per digit, over the row and then over the candidates
// - `radix_partition_write` writes the row as below | equal | above the
//   threshold, keeping the input order inside each side

using namespace metal;

static constant constexpr int radix_bits = 8;
static constant constexpr int radix_bins = 1 << radix_bits;
static constant constexpr int radix_digit_mask = radix_bins - 1;
// Elements per thread in the write phase
static constant constexpr int radix_n_reads = 8;
// Capacity of the candidates cache per thread
static constant constexpr int radix_candidates_per_thread = 8;
static constant constexpr int radix_simd_size = 32;
// Output parts of a row: below, equal and above the threshold
static constant constexpr int radix_threshold_sides = 3;

// Sizes that follow from the number of threads per row
template <int BLOCK_THREADS>
struct RadixLayout {
  static constant constexpr int simd_groups = BLOCK_THREADS / radix_simd_size;
  static constant constexpr int bins_per_thread = radix_bins / BLOCK_THREADS;
  // Scan scratch per value: one sum per simdgroup and the total after them
  static constant constexpr int scan_size = simd_groups + 1;
  // Candidates cache, sized per thread so small groups keep more rows per core
  static constant constexpr int candidates_cache_size =
      radix_candidates_per_thread * BLOCK_THREADS;

  static_assert(
      BLOCK_THREADS % radix_simd_size == 0,
      "BLOCK_THREADS must be a multiple of the simd size");
  static_assert(
      radix_bins % BLOCK_THREADS == 0,
      "BLOCK_THREADS must divide the number of bins");
};

///////////////////////////////////////////////////////////////////////////////
// Radix keys
///////////////////////////////////////////////////////////////////////////////

// Unsigned key of the same width as T, so the digits cover the whole value.
// Signed ints and positive floats flip the sign bit, negative floats flip every
// bit, NaN gets the largest key so it sorts last like LessThan in sort.h.
template <typename T>
struct RadixKeyTraits {
  using KeyT = metal::conditional_t<
      sizeof(T) == 1,
      uint8_t,
      metal::conditional_t<
          sizeof(T) == 2,
          uint16_t,
          metal::conditional_t<sizeof(T) == 4, uint32_t, uint64_t>>>;
  static METAL_FUNC KeyT to_key(T v) {
    constexpr KeyT sign_bit = KeyT(1) << (8 * sizeof(T) - 1);
    if constexpr (metal::is_floating_point_v<T>) {
      if (metal::isnan(v)) {
        return KeyT(~KeyT(0));
      }
      KeyT bits = as_type<KeyT>(v);
      return bits ^ ((bits & sign_bit) ? KeyT(~KeyT(0)) : sign_bit);
    } else if constexpr (metal::is_signed_v<T>) {
      return as_type<KeyT>(v) ^ sign_bit;
    } else {
      return v;
    }
  }
};

///////////////////////////////////////////////////////////////////////////////
// Threadgroup scan
///////////////////////////////////////////////////////////////////////////////

// Exclusive scan of N values per thread over the threadgroup. `values` become
// the sums before this thread, `totals` the sums over all threads.
template <int BLOCK_THREADS, int N>
METAL_FUNC void threadgroup_exclusive_sum(
    thread uint* values,
    thread uint* totals,
    uint simd_lane_id,
    uint simd_group_id,
    threadgroup uint* sums) {
  constexpr int simd_groups = RadixLayout<BLOCK_THREADS>::simd_groups;
  constexpr int scan_size = RadixLayout<BLOCK_THREADS>::scan_size;
  constexpr int last_lane = radix_simd_size - 1;

  // Scan inside each simdgroup, the last lane stores the simdgroup sum
  uint prev[N];
  MLX_MTL_PRAGMA_UNROLL
  for (int c = 0; c < N; c++) {
    prev[c] = simd_prefix_exclusive_sum(values[c]);
    if (simd_lane_id == last_lane) {
      sums[c * scan_size + simd_group_id] = prev[c] + values[c];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Scan the simdgroup sums, the total lands right after them
  if (simd_group_id == 0) {
    bool valid = simd_lane_id < simd_groups;
    MLX_MTL_PRAGMA_UNROLL
    for (int c = 0; c < N; c++) {
      threadgroup uint* group_sums = sums + c * scan_size;
      uint sum = valid ? group_sums[simd_lane_id] : 0;
      uint scanned = simd_prefix_exclusive_sum(sum);
      if (valid) {
        group_sums[simd_lane_id] = scanned;
      }
      if (simd_lane_id == last_lane) {
        group_sums[simd_groups] = scanned + sum;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  MLX_MTL_PRAGMA_UNROLL
  for (int c = 0; c < N; c++) {
    threadgroup uint* group_sums = sums + c * scan_size;
    values[c] = prev[c] + group_sums[simd_group_id];
    totals[c] = group_sums[simd_groups];
  }
}

///////////////////////////////////////////////////////////////////////////////
// Radix select
///////////////////////////////////////////////////////////////////////////////

// Bin of the k-th key: its digit, the keys below it and the keys in it
struct RadixBin {
  uint digit;
  uint below;
  uint count;
};

// Source of a select pass: the row, the row while it fills the candidates
// cache, or the cache only
enum class RadixPass { Row, FillCache, Cache };

// Key of the k-th element with the counts of keys below and equal to it
template <typename KeyT>
struct RadixThreshold {
  KeyT key;
  uint n_below;
  uint n_equal;
};

// Counts the key if its fixed digits match the prefix. Returns whether it did.
template <typename KeyT>
METAL_FUNC bool radix_count_key(
    KeyT key,
    KeyT prefix,
    KeyT fixed,
    int shift,
    threadgroup atomic_uint* hist) {
  if ((key & fixed) != prefix) {
    return false;
  }
  uint digit = uint((key >> shift) & KeyT(radix_digit_mask));
  atomic_fetch_add_explicit(&hist[digit], 1u, memory_order_relaxed);
  return true;
}

// Scans the histogram from the bottom, `bins_per_thread` bins per thread.
// Returns the bin of the k-th smallest key with the count below it.
template <int BLOCK_THREADS>
METAL_FUNC RadixBin radix_find_bin(
    threadgroup atomic_uint* hist,
    uint k,
    uint lid_x,
    uint simd_lane_id,
    uint simd_group_id,
    threadgroup uint* bin_count_simdgroup_sums,
    threadgroup RadixBin& kth_bin) {
  constexpr int bins_per_thread = RadixLayout<BLOCK_THREADS>::bins_per_thread;

  uint first = lid_x * bins_per_thread;
  uint count = 0;
  MLX_MTL_PRAGMA_UNROLL
  for (int j = 0; j < bins_per_thread; j++) {
    count += atomic_load_explicit(&hist[first + j], memory_order_relaxed);
  }
  uint total;
  threadgroup_exclusive_sum<BLOCK_THREADS, 1>(
      &count, &total, simd_lane_id, simd_group_id, bin_count_simdgroup_sums);

  // The sum from the bottom reaches k inside exactly one bin
  uint below = count;
  MLX_MTL_PRAGMA_UNROLL
  for (int j = 0; j < bins_per_thread; j++) {
    uint digit = first + j;
    uint c = atomic_load_explicit(&hist[digit], memory_order_relaxed);
    if (below < k && below + c >= k) {
      kth_bin.digit = digit;
      kth_bin.below = below;
      kth_bin.count = c;
    }
    below += c;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return kth_bin;
}

// Finds the key of the k-th smallest element of the row, one histogram pass
// per digit from the top. Once the keys that still match the prefix fit in the
// candidates cache, the later passes read the cache instead of the row.
template <typename T, int BLOCK_THREADS>
METAL_FUNC RadixThreshold<typename RadixKeyTraits<T>::KeyT> radix_select_row(
    const device T* in,
    int n,
    uint k,
    uint lid_x,
    uint simd_lane_id,
    uint simd_group_id,
    threadgroup atomic_uint* hist,
    threadgroup typename RadixKeyTraits<T>::KeyT* candidates_cache,
    threadgroup atomic_uint& candidates_cache_size,
    threadgroup uint* bin_count_simdgroup_sums,
    threadgroup RadixBin& kth_bin) {
  using KeyT = typename RadixKeyTraits<T>::KeyT;

  KeyT prefix = 0;
  KeyT fixed = 0;
  uint n_below = 0;
  uint n_candidates = 0;
  RadixPass pass = RadixPass::Row;
  RadixBin bin = {0, 0, 0};

  MLX_MTL_PRAGMA_UNROLL
  for (int shift = 8 * sizeof(KeyT) - radix_bits; shift >= 0;
       shift -= radix_bits) {
    MLX_MTL_PRAGMA_UNROLL
    for (int j = 0; j < RadixLayout<BLOCK_THREADS>::bins_per_thread; j++) {
      atomic_store_explicit(
          &hist[lid_x + j * BLOCK_THREADS], 0u, memory_order_relaxed);
    }
    if (lid_x == 0) {
      atomic_store_explicit(&candidates_cache_size, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Count the keys that still match the fixed prefix
    if (pass == RadixPass::Cache) {
      for (uint i = lid_x; i < n_candidates; i += BLOCK_THREADS) {
        radix_count_key(candidates_cache[i], prefix, fixed, shift, hist);
      }
    } else {
      for (int i = lid_x; i < n; i += BLOCK_THREADS) {
        KeyT key = RadixKeyTraits<T>::to_key(in[i]);
        if (radix_count_key(key, prefix, fixed, shift, hist) &&
            pass == RadixPass::FillCache) {
          uint slot = atomic_fetch_add_explicit(
              &candidates_cache_size, 1u, memory_order_relaxed);
          candidates_cache[slot] = key;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (pass == RadixPass::FillCache) {
      n_candidates =
          atomic_load_explicit(&candidates_cache_size, memory_order_relaxed);
      pass = RadixPass::Cache;
    }

    bin = radix_find_bin<BLOCK_THREADS>(
        hist,
        k,
        lid_x,
        simd_lane_id,
        simd_group_id,
        bin_count_simdgroup_sums,
        kth_bin);
    n_below += bin.below;
    k -= bin.below;
    prefix |= KeyT(bin.digit) << shift;
    fixed |= KeyT(radix_digit_mask) << shift;

    // Once the candidates fit, the next pass copies them to the cache
    if (pass == RadixPass::Row &&
        bin.count <= RadixLayout<BLOCK_THREADS>::candidates_cache_size) {
      pass = RadixPass::FillCache;
    }
  }

  RadixThreshold<KeyT> result;
  result.key = prefix;
  result.n_below = n_below;
  result.n_equal = bin.count;
  return result;
}

///////////////////////////////////////////////////////////////////////////////
// Partition write
///////////////////////////////////////////////////////////////////////////////

// Writes the row as below, equal and above the threshold, keeping the input
// order inside each side. Works in chunks of BLOCK_THREADS * N_READS elements,
// one threadgroup scan per chunk gives each thread its output positions.
template <
    typename T,
    typename U,
    bool ARG_PARTITION,
    int BLOCK_THREADS,
    int N_READS>
METAL_FUNC void radix_partition_write(
    const device T* in,
    device U* out,
    int n,
    RadixThreshold<typename RadixKeyTraits<T>::KeyT> threshold,
    uint lid_x,
    uint simd_lane_id,
    uint simd_group_id,
    threadgroup uint* threshold_side_count_simdgroup_sums) {
  using KeyT = typename RadixKeyTraits<T>::KeyT;

  // Where the next element of each side goes in the output
  uint threshold_side_offset[radix_threshold_sides] = {
      0, threshold.n_below, threshold.n_below + threshold.n_equal};

  for (int j = 0; j < n; j += BLOCK_THREADS * N_READS) {
    T vals[N_READS];
    int threshold_side[N_READS];
    uint pos[radix_threshold_sides] = {0, 0, 0};

    // Side of the threshold for N_READS consecutive elements
    MLX_MTL_PRAGMA_UNROLL
    for (int r = 0; r < N_READS; r++) {
      int i = j + lid_x * N_READS + r;
      if (i < n) {
        vals[r] = in[i];
        KeyT key = RadixKeyTraits<T>::to_key(vals[r]);
        threshold_side[r] =
            key < threshold.key ? 0 : (key == threshold.key ? 1 : 2);
        pos[threshold_side[r]]++;
      }
    }

    // Offsets of this thread inside the chunk and the chunk totals
    uint totals[radix_threshold_sides];
    threadgroup_exclusive_sum<BLOCK_THREADS, radix_threshold_sides>(
        pos,
        totals,
        simd_lane_id,
        simd_group_id,
        threshold_side_count_simdgroup_sums);

    // Add the row offsets of each side, then write
    MLX_MTL_PRAGMA_UNROLL
    for (int c = 0; c < radix_threshold_sides; c++) {
      pos[c] += threshold_side_offset[c];
      threshold_side_offset[c] += totals[c];
    }
    MLX_MTL_PRAGMA_UNROLL
    for (int r = 0; r < N_READS; r++) {
      int i = j + lid_x * N_READS + r;
      if (i < n) {
        if constexpr (ARG_PARTITION) {
          out[pos[threshold_side[r]]] = U(i);
        } else {
          out[pos[threshold_side[r]]] = U(vals[r]);
        }
        pos[threshold_side[r]]++;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

///////////////////////////////////////////////////////////////////////////////
// Partition kernel
///////////////////////////////////////////////////////////////////////////////

template <
    typename T,
    typename U,
    bool ARG_PARTITION,
    int BLOCK_THREADS,
    int N_READS = radix_n_reads>
[[kernel, max_total_threads_per_threadgroup(BLOCK_THREADS)]] void
radix_partition(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant int& axis_size [[buffer(2)]],
    const constant int& kth [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  using KeyT = typename RadixKeyTraits<T>::KeyT;
  using Layout = RadixLayout<BLOCK_THREADS>;

  threadgroup atomic_uint hist[radix_bins];
  threadgroup KeyT candidates_cache[Layout::candidates_cache_size];
  threadgroup atomic_uint candidates_cache_size;
  threadgroup uint bin_count_simdgroup_sums[Layout::scan_size];
  threadgroup uint threshold_side_count_simdgroup_sums
      [radix_threshold_sides * Layout::scan_size];
  threadgroup RadixBin kth_bin;

  // Move to this row
  int64_t offset = int64_t(gid.y) * axis_size;
  in += offset;
  out += offset;

  // kth is zero based, k is a count
  uint k = kth + 1;
  auto threshold = radix_select_row<T, BLOCK_THREADS>(
      in,
      axis_size,
      k,
      lid.x,
      simd_lane_id,
      simd_group_id,
      hist,
      candidates_cache,
      candidates_cache_size,
      bin_count_simdgroup_sums,
      kth_bin);
  radix_partition_write<T, U, ARG_PARTITION, BLOCK_THREADS, N_READS>(
      in,
      out,
      axis_size,
      threshold,
      lid.x,
      simd_lane_id,
      simd_group_id,
      threshold_side_count_simdgroup_sums);
}
