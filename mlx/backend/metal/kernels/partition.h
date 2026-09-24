// Copyright © 2026 Apple Inc.

// Radix select partition, three kernels that share the keys, the histogram
// bookkeeping and the write phase:
// - `RadixLayout` sizes the digits and the threadgroup memory from the
//   threads per row
// - `RadixKeyTraits` maps values to unsigned keys with the same order
// - `threadgroup_exclusive_sum` scans per thread counts over the threadgroup
// - `RadixSelectState` holds the prefix of the k-th key fixed so far, one
//   digit per histogram pass, and `radix_find_bin` locates the digit
// - `radix_select_row` runs the passes over the row and then over the
//   candidates, `radix_replay` recovers the state from a histogram history
// - `radix_partition_write` writes a range of the row as below | equal |
//   above the threshold, keeping the input order inside each side
// - `radix_partition` handles one row per threadgroup
// - `simd_partition` handles short rows with a small k in one simdgroup
// - `split_partition_histogram` and `split_partition` handle a few very
//   long rows with several threadgroups per row

using namespace metal;

// Digit bits at few threads per row, more threads afford wider digits
static constant constexpr int radix_bits = 8;
// Elements per thread in the write phase
static constant constexpr int radix_n_reads = 8;
// Capacity of the candidates cache per thread
static constant constexpr int radix_candidates_per_thread = 8;
// Keys a thread loads together in a select pass, so the loads are in flight
// at the same time
static constant constexpr int radix_n_loads = 16;
static constant constexpr int radix_simd_size = 32;
// Output parts of a row: below, equal and above the threshold
static constant constexpr int radix_threshold_sides = 3;

// Sizes that follow from the number of threads per row
template <int BLOCK_THREADS>
struct RadixLayout {
  static constant constexpr int simd_groups = BLOCK_THREADS / radix_simd_size;
  // Wide digits need fewer passes and leave fewer matching keys per pass,
  // many threads spread their bins at no extra cost per thread
  static constant constexpr int wide_digit_threads = 256;
  static constant constexpr int digit_bits = BLOCK_THREADS >= wide_digit_threads
      ? RADIX_PARTITION_WIDE_DIGIT_BITS
      : radix_bits;
  static constant constexpr int bins = 1 << digit_bits;
  static constant constexpr int digit_mask = bins - 1;
  static constant constexpr int bins_per_thread = bins / BLOCK_THREADS;
  static constant constexpr int bins_per_lane = bins / radix_simd_size;
  // Scan scratch per value: one sum per simdgroup and the total after them
  static constant constexpr int scan_size = simd_groups + 1;
  // Candidates cache, sized per thread so small groups keep more rows per core
  static constant constexpr int candidates_cache_size =
      radix_candidates_per_thread * BLOCK_THREADS;

  // Digit passes over a key of `key_bits`, the last digit may be narrower
  static METAL_FUNC constexpr int passes(int key_bits) {
    return (key_bits + digit_bits - 1) / digit_bits;
  }
  static METAL_FUNC constexpr int shift(int key_bits, int pass) {
    int shift = key_bits - digit_bits * (pass + 1);
    return shift > 0 ? shift : 0;
  }

  static_assert(
      BLOCK_THREADS % radix_simd_size == 0,
      "BLOCK_THREADS must be a multiple of the simd size");
  static_assert(
      bins % BLOCK_THREADS == 0,
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

// Side of the threshold a key goes to: 0 below, 1 equal, 2 above
template <typename KeyT>
METAL_FUNC int radix_threshold_side(KeyT key, KeyT threshold) {
  return key < threshold ? 0 : (key == threshold ? 1 : 2);
}

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

// Prefix of the k-th key fixed so far with the counts around it, k counts
// from the bottom of the keys that still match the prefix
template <typename KeyT>
struct RadixSelectState {
  KeyT prefix;
  KeyT fixed;
  uint k;
  uint n_below;
  uint n_equal;
};

template <typename KeyT>
METAL_FUNC RadixSelectState<KeyT> radix_select_start(uint k) {
  RadixSelectState<KeyT> state;
  state.prefix = 0;
  state.fixed = 0;
  state.k = k;
  state.n_below = 0;
  state.n_equal = 0;
  return state;
}

// Fixes the digit of the bin found by the pass at `shift`
template <int DIGIT_BITS, typename KeyT>
METAL_FUNC void radix_select_advance(
    thread RadixSelectState<KeyT>& state,
    RadixBin bin,
    int shift) {
  state.n_below += bin.below;
  state.k -= bin.below;
  state.prefix |= KeyT(bin.digit) << shift;
  state.fixed |= KeyT((1 << DIGIT_BITS) - 1) << shift;
  state.n_equal = bin.count;
}

// Counts the key if its fixed digits match the prefix. Returns whether it did.
template <int DIGIT_BITS, typename KeyT>
METAL_FUNC bool radix_count_key(
    KeyT key,
    const thread RadixSelectState<KeyT>& state,
    int shift,
    threadgroup atomic_uint* hist) {
  if ((key & state.fixed) != state.prefix) {
    return false;
  }
  uint digit = uint((key >> shift) & KeyT((1 << DIGIT_BITS) - 1));
  atomic_fetch_add_explicit(&hist[digit], 1u, memory_order_relaxed);
  return true;
}

// A histogram lives in threadgroup memory during a select pass or in device
// memory as the history of the split kernels
METAL_FUNC uint radix_bin_count(threadgroup atomic_uint* hist, uint bin) {
  return atomic_load_explicit(&hist[bin], memory_order_relaxed);
}

METAL_FUNC uint radix_bin_count(const device uint* hist, uint bin) {
  return hist[bin];
}

// Sum of the BINS_PER_THREAD bins of one thread from `first`
template <int BINS_PER_THREAD, typename HistPtr>
METAL_FUNC uint radix_bins_sum(HistPtr hist, uint first) {
  uint sum = 0;
  MLX_MTL_PRAGMA_UNROLL
  for (int j = 0; j < BINS_PER_THREAD; j++) {
    sum += radix_bin_count(hist, first + j);
  }
  return sum;
}

// Walks the bins of one thread with `below` keys before them. Returns whether
// the sum from the bottom reaches k inside them, then `bin` is that bin.
template <int BINS_PER_THREAD, typename HistPtr>
METAL_FUNC bool radix_bins_walk(
    HistPtr hist,
    uint first,
    uint below,
    uint k,
    thread RadixBin& bin) {
  bool found = false;
  MLX_MTL_PRAGMA_UNROLL
  for (int j = 0; j < BINS_PER_THREAD; j++) {
    uint count = radix_bin_count(hist, first + j);
    if (!found && below < k && below + count >= k) {
      found = true;
      bin.digit = first + j;
      bin.below = below;
      bin.count = count;
    }
    below += count;
  }
  return found;
}

// Scans the histogram from the bottom with the whole threadgroup,
// `bins_per_thread` bins per thread. Returns the bin of the k-th smallest key.
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
  uint below = radix_bins_sum<bins_per_thread>(hist, first);
  uint total;
  threadgroup_exclusive_sum<BLOCK_THREADS, 1>(
      &below, &total, simd_lane_id, simd_group_id, bin_count_simdgroup_sums);

  // The sum from the bottom reaches k inside exactly one bin
  RadixBin bin;
  if (radix_bins_walk<bins_per_thread>(hist, first, below, k, bin)) {
    kth_bin = bin;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return kth_bin;
}

// The same scan with one simdgroup over a histogram in device memory, without
// a barrier. All 32 lanes must call together.
template <int BINS_PER_LANE>
METAL_FUNC RadixBin
radix_find_bin_simd(const device uint* hist, uint k, uint simd_lane_id) {
  uint first = simd_lane_id * BINS_PER_LANE;
  uint below =
      simd_prefix_exclusive_sum(radix_bins_sum<BINS_PER_LANE>(hist, first));
  RadixBin bin = {0, 0, 0};
  bool found = radix_bins_walk<BINS_PER_LANE>(hist, first, below, k, bin);
  uint owner = simd_min(found ? simd_lane_id : uint(radix_simd_size));
  bin.digit = simd_shuffle(bin.digit, owner);
  bin.below = simd_shuffle(bin.below, owner);
  bin.count = simd_shuffle(bin.count, owner);
  return bin;
}

// Source of a select pass: the row, the row while it fills the candidates
// cache, or the cache only
enum class RadixPass { Row, FillCache, Cache };

// Keys of radix_n_loads elements of a thread, BLOCK_THREADS apart from `i`
template <typename T, int BLOCK_THREADS>
METAL_FUNC void radix_load_keys(
    const device T* in,
    int i,
    thread typename RadixKeyTraits<T>::KeyT* keys) {
  MLX_MTL_PRAGMA_UNROLL
  for (int r = 0; r < radix_n_loads; r++) {
    keys[r] = RadixKeyTraits<T>::to_key(in[i + r * BLOCK_THREADS]);
  }
}

// Counts a key of the row, the fill pass also appends it to the cache
template <int DIGIT_BITS, typename KeyT>
METAL_FUNC void radix_count_row_key(
    KeyT key,
    const thread RadixSelectState<KeyT>& state,
    int shift,
    RadixPass pass,
    threadgroup atomic_uint* hist,
    threadgroup KeyT* candidates_cache,
    threadgroup atomic_uint& candidates_cache_size) {
  if (radix_count_key<DIGIT_BITS>(key, state, shift, hist) &&
      pass == RadixPass::FillCache) {
    uint slot = atomic_fetch_add_explicit(
        &candidates_cache_size, 1u, memory_order_relaxed);
    candidates_cache[slot] = key;
  }
}

// Finds the k-th smallest element of the row, one histogram pass per digit
// from the top. Once the keys that still match the prefix fit in the
// candidates cache, the later passes read the cache instead of the row.
template <typename T, int BLOCK_THREADS>
METAL_FUNC RadixSelectState<typename RadixKeyTraits<T>::KeyT> radix_select_row(
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
  using Layout = RadixLayout<BLOCK_THREADS>;
  constexpr int key_bits = 8 * sizeof(KeyT);

  auto state = radix_select_start<KeyT>(k);
  uint n_candidates = 0;
  RadixPass pass = RadixPass::Row;

  MLX_MTL_PRAGMA_UNROLL
  for (int p = 0; p < Layout::passes(key_bits); p++) {
    int shift = Layout::shift(key_bits, p);
    MLX_MTL_PRAGMA_UNROLL
    for (int j = 0; j < Layout::bins_per_thread; j++) {
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
        radix_count_key<Layout::digit_bits>(
            candidates_cache[i], state, shift, hist);
      }
    } else {
      int i = lid_x;
      for (; i + (radix_n_loads - 1) * BLOCK_THREADS < n;
           i += radix_n_loads * BLOCK_THREADS) {
        KeyT keys[radix_n_loads];
        radix_load_keys<T, BLOCK_THREADS>(in, i, keys);
        MLX_MTL_PRAGMA_UNROLL
        for (int r = 0; r < radix_n_loads; r++) {
          radix_count_row_key<Layout::digit_bits>(
              keys[r],
              state,
              shift,
              pass,
              hist,
              candidates_cache,
              candidates_cache_size);
        }
      }
      for (; i < n; i += BLOCK_THREADS) {
        radix_count_row_key<Layout::digit_bits>(
            RadixKeyTraits<T>::to_key(in[i]),
            state,
            shift,
            pass,
            hist,
            candidates_cache,
            candidates_cache_size);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (pass == RadixPass::FillCache) {
      n_candidates =
          atomic_load_explicit(&candidates_cache_size, memory_order_relaxed);
      pass = RadixPass::Cache;
    }

    RadixBin bin = radix_find_bin<BLOCK_THREADS>(
        hist,
        state.k,
        lid_x,
        simd_lane_id,
        simd_group_id,
        bin_count_simdgroup_sums,
        kth_bin);
    radix_select_advance<Layout::digit_bits>(state, bin, shift);

    // Once the candidates fit, the next pass copies them to the cache
    if (pass == RadixPass::Row && bin.count <= Layout::candidates_cache_size) {
      pass = RadixPass::FillCache;
    }
  }
  return state;
}

// Recovers the state after `passes` digit passes from a histogram history,
// one histogram per pass with the digits of BLOCK_THREADS per row. One
// simdgroup calls together.
template <typename KeyT, int BLOCK_THREADS>
METAL_FUNC RadixSelectState<KeyT>
radix_replay(const device uint* hist, int passes, uint k, uint simd_lane_id) {
  using Layout = RadixLayout<BLOCK_THREADS>;
  constexpr int key_bits = 8 * sizeof(KeyT);

  auto state = radix_select_start<KeyT>(k);
  for (int p = 0; p < passes; p++) {
    RadixBin bin = radix_find_bin_simd<Layout::bins_per_lane>(
        hist + p * Layout::bins, state.k, simd_lane_id);
    radix_select_advance<Layout::digit_bits>(
        state, bin, Layout::shift(key_bits, p));
  }
  return state;
}

///////////////////////////////////////////////////////////////////////////////
// Partition write
///////////////////////////////////////////////////////////////////////////////

// Writes one chunk of BLOCK_THREADS * N_READS elements from `j`, one
// threadgroup scan gives each thread its output positions. A FULL chunk
// ends before `end`, so it needs no bounds checks.
template <
    typename T,
    typename U,
    bool ARG_PARTITION,
    int BLOCK_THREADS,
    int N_READS,
    bool FULL>
METAL_FUNC void radix_partition_write_chunk(
    const device T* in,
    device U* out,
    int j,
    int end,
    typename RadixKeyTraits<T>::KeyT threshold,
    thread uint* side_offset,
    uint lid_x,
    uint simd_lane_id,
    uint simd_group_id,
    threadgroup uint* threshold_side_count_simdgroup_sums) {
  using KeyT = typename RadixKeyTraits<T>::KeyT;

  T vals[N_READS];
  int threshold_side[N_READS];
  uint pos[radix_threshold_sides] = {0, 0, 0};

  // Side of the threshold for N_READS consecutive elements
  MLX_MTL_PRAGMA_UNROLL
  for (int r = 0; r < N_READS; r++) {
    int i = j + lid_x * N_READS + r;
    if (FULL || i < end) {
      vals[r] = in[i];
      KeyT key = RadixKeyTraits<T>::to_key(vals[r]);
      threshold_side[r] = radix_threshold_side(key, threshold);
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

  // Add the offsets of each side, then write
  MLX_MTL_PRAGMA_UNROLL
  for (int c = 0; c < radix_threshold_sides; c++) {
    pos[c] += side_offset[c];
    side_offset[c] += totals[c];
  }
  MLX_MTL_PRAGMA_UNROLL
  for (int r = 0; r < N_READS; r++) {
    int i = j + lid_x * N_READS + r;
    if (FULL || i < end) {
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

// Writes the elements in [start, end) of the row below, equal and above the
// threshold, keeping the input order inside each side. `side_offset` holds
// where each side continues in the output and moves along.
template <
    typename T,
    typename U,
    bool ARG_PARTITION,
    int BLOCK_THREADS,
    int N_READS>
METAL_FUNC void radix_partition_write(
    const device T* in,
    device U* out,
    int start,
    int end,
    typename RadixKeyTraits<T>::KeyT threshold,
    thread uint* side_offset,
    uint lid_x,
    uint simd_lane_id,
    uint simd_group_id,
    threadgroup uint* threshold_side_count_simdgroup_sums) {
  constexpr int chunk_size = BLOCK_THREADS * N_READS;

  for (int j = start; j < end; j += chunk_size) {
    if (j + chunk_size <= end) {
      radix_partition_write_chunk<
          T,
          U,
          ARG_PARTITION,
          BLOCK_THREADS,
          N_READS,
          true>(
          in,
          out,
          j,
          end,
          threshold,
          side_offset,
          lid_x,
          simd_lane_id,
          simd_group_id,
          threshold_side_count_simdgroup_sums);
    } else {
      radix_partition_write_chunk<
          T,
          U,
          ARG_PARTITION,
          BLOCK_THREADS,
          N_READS,
          false>(
          in,
          out,
          j,
          end,
          threshold,
          side_offset,
          lid_x,
          simd_lane_id,
          simd_group_id,
          threshold_side_count_simdgroup_sums);
    }
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

  threadgroup atomic_uint hist[Layout::bins];
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
  auto state = radix_select_row<T, BLOCK_THREADS>(
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
  uint side_offset[radix_threshold_sides] = {
      0, state.n_below, state.n_below + state.n_equal};
  radix_partition_write<T, U, ARG_PARTITION, BLOCK_THREADS, N_READS>(
      in,
      out,
      0,
      axis_size,
      state.prefix,
      side_offset,
      lid.x,
      simd_lane_id,
      simd_group_id,
      threshold_side_count_simdgroup_sums);
}

///////////////////////////////////////////////////////////////////////////////
// Simdgroup partition
///////////////////////////////////////////////////////////////////////////////

static constant constexpr int simd_partition_threads =
    SIMD_PARTITION_ROWS_PER_GROUP * radix_simd_size;
// Keys of 16 bits or less are packed with the element id in the low half, so
// one reduction gives the threshold and the lane that removes it
static constant constexpr int simd_partition_id_bits = 16;
static constant constexpr uint simd_partition_id_mask =
    (1u << simd_partition_id_bits) - 1;
// The three side counts of a lane travel in one prefix sum
static constant constexpr int simd_partition_count_bits = 10;
static constant constexpr uint simd_partition_count_mask =
    (1u << simd_partition_count_bits) - 1;

// First lane of the simdgroup with `flag` set
METAL_FUNC uint simd_first_lane(bool flag) {
  return ctz(uint(uint64_t(simd_ballot(flag))));
}

// Rows that fit in the registers of one simdgroup, N_PER elements per lane.
// The threshold is the extreme key removed once per round with simd_max or
// simd_min from the end of the order closer to kth, so the kernel needs no
// threadgroup memory, barriers or atomics. Keys of at most 32 bits.
template <typename T, typename U, bool ARG_PARTITION, int N_PER>
[[kernel, max_total_threads_per_threadgroup(simd_partition_threads)]] void
simd_partition(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const constant int& axis_size [[buffer(2)]],
    const constant int& kth [[buffer(3)]],
    const constant int& n_rows [[buffer(4)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  using KeyT = typename RadixKeyTraits<T>::KeyT;
  constexpr int capacity = N_PER * radix_simd_size;
  constexpr bool packed = sizeof(KeyT) <= 2;
  static_assert(sizeof(KeyT) <= 4, "the keys must fit in 32 bits");
  static_assert(
      capacity <= simd_partition_id_mask, "the element id must fit its bits");
  static_assert(
      capacity <= simd_partition_count_mask,
      "the side counts must fit their bits");

  int row = gid.y * SIMD_PARTITION_ROWS_PER_GROUP + simd_group_id;
  if (row >= n_rows) {
    return;
  }
  int64_t offset = int64_t(row) * axis_size;
  in += offset;
  out += offset;

  // Lane l holds the elements l, l + 32, l + 64, ...
  T vals[N_PER];
  uint keys[N_PER];
  bool alive[N_PER];
  MLX_MTL_PRAGMA_UNROLL
  for (int r = 0; r < N_PER; r++) {
    int i = simd_lane_id + r * radix_simd_size;
    alive[r] = i < axis_size;
    vals[r] = alive[r] ? in[i] : T(0);
    uint key = uint(RadixKeyTraits<T>::to_key(vals[r]));
    keys[r] = packed
        ? ((key << simd_partition_id_bits) | (simd_partition_id_mask - i))
        : key;
  }

  // Remove one extreme element per round until the kth one is the threshold
  bool from_top = axis_size - kth <= kth + 1;
  int rounds = from_top ? axis_size - kth : kth + 1;
  uint threshold = 0;
  for (int t = 0; t < rounds; t++) {
    uint local = from_top ? 0u : ~0u;
    MLX_MTL_PRAGMA_UNROLL
    for (int r = 0; r < N_PER; r++) {
      if (alive[r]) {
        local = from_top ? max(local, keys[r]) : min(local, keys[r]);
      }
    }
    threshold = from_top ? simd_max(local) : simd_min(local);
    if constexpr (packed) {
      // The id inside the threshold names the element to remove
      uint id = simd_partition_id_mask - (threshold & simd_partition_id_mask);
      if (simd_lane_id == id % radix_simd_size) {
        alive[id / radix_simd_size] = false;
      }
    } else {
      bool hit = false;
      MLX_MTL_PRAGMA_UNROLL
      for (int r = 0; r < N_PER; r++) {
        hit = hit || (alive[r] && keys[r] == threshold);
      }
      if (simd_lane_id == simd_first_lane(hit)) {
        bool removed = false;
        MLX_MTL_PRAGMA_UNROLL
        for (int r = 0; r < N_PER; r++) {
          if (!removed && alive[r] && keys[r] == threshold) {
            alive[r] = false;
            removed = true;
          }
        }
      }
    }
  }
  uint threshold_key =
      packed ? (threshold >> simd_partition_id_bits) : threshold;

  // Side of the threshold, the three counts packed in one prefix sum
  int threshold_side[N_PER];
  uint packed_count = 0;
  MLX_MTL_PRAGMA_UNROLL
  for (int r = 0; r < N_PER; r++) {
    int i = simd_lane_id + r * radix_simd_size;
    if (i < axis_size) {
      uint key = packed ? (keys[r] >> simd_partition_id_bits) : keys[r];
      threshold_side[r] = radix_threshold_side(key, threshold_key);
      packed_count += 1u << (simd_partition_count_bits * threshold_side[r]);
    }
  }
  uint prefix = simd_prefix_exclusive_sum(packed_count);
  uint total = simd_sum(packed_count);
  uint pos[radix_threshold_sides];
  uint side_start = 0;
  MLX_MTL_PRAGMA_UNROLL
  for (int c = 0; c < radix_threshold_sides; c++) {
    int bits = simd_partition_count_bits * c;
    pos[c] = side_start + ((prefix >> bits) & simd_partition_count_mask);
    side_start += (total >> bits) & simd_partition_count_mask;
  }
  MLX_MTL_PRAGMA_UNROLL
  for (int r = 0; r < N_PER; r++) {
    int i = simd_lane_id + r * radix_simd_size;
    if (i < axis_size) {
      if constexpr (ARG_PARTITION) {
        out[pos[threshold_side[r]]] = U(i);
      } else {
        out[pos[threshold_side[r]]] = U(vals[r]);
      }
      pos[threshold_side[r]]++;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////
// Split-row partition
///////////////////////////////////////////////////////////////////////////////

// A few very long rows get several threadgroups per row. One dispatch per
// digit counts the keys that match the prefix into a device histogram of
// that pass, every group replays the finished passes from the histogram
// history, so no dispatch searches the bin. The last pass records per group
// what the write dispatch needs to place the chunk of every group.

static constant constexpr int split_partition_threads = SPLIT_PARTITION_THREADS;
using SplitLayout = RadixLayout<split_partition_threads>;
static_assert(
    SplitLayout::digit_bits == RADIX_PARTITION_WIDE_DIGIT_BITS,
    "the host sizes the histogram history");

// Record of a group after the last pass: its histogram of that pass, the
// exclusive prefix over it, and the elements below and above the prefix
struct SplitRecord {
  static constant constexpr int hist = 0;
  static constant constexpr int prefix = SplitLayout::bins;
  static constant constexpr int outside_below = 2 * SplitLayout::bins;
  static constant constexpr int outside_above = 2 * SplitLayout::bins + 1;
  static constant constexpr int size = 2 * SplitLayout::bins + 2;
};
static_assert(
    SplitRecord::size == SPLIT_PARTITION_RECORD_SIZE,
    "the host allocates the records");

// Range of the row that one group handles
struct SplitChunk {
  int start;
  int end;
};

METAL_FUNC SplitChunk split_chunk(int axis_size, uint group, uint groups) {
  int size = (axis_size + groups - 1) / groups;
  SplitChunk chunk;
  chunk.start = group * size;
  chunk.end = min(axis_size, chunk.start + size);
  return chunk;
}

// Adds the sides of the threshold in the chunk of a recorded group, the
// threshold has `digit` in the last pass
METAL_FUNC void
split_record_sides(const device uint* record, uint digit, thread uint* sides) {
  uint below = record[SplitRecord::prefix + digit];
  uint equal = record[SplitRecord::hist + digit];
  uint matched = record[SplitRecord::prefix + SplitLayout::bins - 1] +
      record[SplitRecord::hist + SplitLayout::bins - 1];
  sides[0] += record[SplitRecord::outside_below] + below;
  sides[1] += equal;
  sides[2] += record[SplitRecord::outside_above] + matched - below - equal;
}

// Counts a key of the chunk, the last pass also counts the keys outside the
// prefix on each side of it
template <int DIGIT_BITS, typename KeyT>
METAL_FUNC void split_count_key(
    KeyT key,
    const thread RadixSelectState<KeyT>& state,
    int shift,
    threadgroup atomic_uint* hist,
    bool last,
    thread uint& outside_below,
    thread uint& outside_above) {
  if (!radix_count_key<DIGIT_BITS>(key, state, shift, hist) && last) {
    KeyT masked = key & state.fixed;
    outside_below += masked < state.prefix;
    outside_above += masked > state.prefix;
  }
}

// One digit pass over the chunk of every group. In the last pass the group
// also leaves its record.
template <typename T>
[[kernel, max_total_threads_per_threadgroup(split_partition_threads)]] void
split_partition_histogram(
    const device T* in [[buffer(0)]],
    device uint* hist [[buffer(1)]],
    device uint* records [[buffer(2)]],
    const constant int& axis_size [[buffer(3)]],
    const constant int& kth [[buffer(4)]],
    const constant int& pass [[buffer(5)]],
    const constant int& n_passes [[buffer(6)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  using KeyT = typename RadixKeyTraits<T>::KeyT;
  using Layout = SplitLayout;
  constexpr int threads = split_partition_threads;
  constexpr int key_bits = 8 * sizeof(KeyT);

  threadgroup atomic_uint local_hist[Layout::bins];
  threadgroup atomic_uint outside[2];
  threadgroup uint bin_count_simdgroup_sums[RadixLayout<threads>::scan_size];
  threadgroup RadixSelectState<KeyT> shared_state;

  int row = gid.y;
  in += int64_t(row) * axis_size;
  device uint* row_hist = hist + int64_t(row) * n_passes * Layout::bins;
  int shift = Layout::shift(key_bits, pass);
  bool last = pass == n_passes - 1;

  // One simdgroup replays the finished passes for the whole group
  if (simd_group_id == 0) {
    auto state = radix_replay<KeyT, threads>(
        row_hist, pass, uint(kth + 1), simd_lane_id);
    if (simd_lane_id == 0) {
      shared_state = state;
    }
  }
  MLX_MTL_PRAGMA_UNROLL
  for (int j = 0; j < Layout::bins_per_thread; j++) {
    atomic_store_explicit(
        &local_hist[lid.x + j * threads], 0u, memory_order_relaxed);
  }
  if (lid.x < 2) {
    atomic_store_explicit(&outside[lid.x], 0u, memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  RadixSelectState<KeyT> state = shared_state;

  // Count the keys that still match the prefix, the last pass also counts
  // the others on each side of it
  SplitChunk chunk = split_chunk(axis_size, gid.x, gsize.x);
  uint outside_below = 0;
  uint outside_above = 0;
  int i = chunk.start + lid.x;
  for (; i + (radix_n_loads - 1) * threads < chunk.end;
       i += radix_n_loads * threads) {
    KeyT keys[radix_n_loads];
    radix_load_keys<T, threads>(in, i, keys);
    MLX_MTL_PRAGMA_UNROLL
    for (int r = 0; r < radix_n_loads; r++) {
      split_count_key<Layout::digit_bits>(
          keys[r],
          state,
          shift,
          local_hist,
          last,
          outside_below,
          outside_above);
    }
  }
  for (; i < chunk.end; i += threads) {
    split_count_key<Layout::digit_bits>(
        RadixKeyTraits<T>::to_key(in[i]),
        state,
        shift,
        local_hist,
        last,
        outside_below,
        outside_above);
  }
  if (last) {
    outside_below = simd_sum(outside_below);
    outside_above = simd_sum(outside_above);
    if (simd_lane_id == 0) {
      atomic_fetch_add_explicit(
          &outside[0], outside_below, memory_order_relaxed);
      atomic_fetch_add_explicit(
          &outside[1], outside_above, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Add the bins of this thread to the histogram of the pass
  constexpr int bins_per_thread = Layout::bins_per_thread;
  uint first = lid.x * bins_per_thread;
  uint counts[bins_per_thread];
  uint sum = 0;
  MLX_MTL_PRAGMA_UNROLL
  for (int j = 0; j < bins_per_thread; j++) {
    counts[j] = radix_bin_count(local_hist, first + j);
    sum += counts[j];
    if (counts[j] != 0) {
      atomic_fetch_add_explicit(
          (device atomic_uint*)&row_hist[pass * Layout::bins + first + j],
          counts[j],
          memory_order_relaxed);
    }
  }
  if (last) {
    device uint* record =
        records + (int64_t(row) * gsize.x + gid.x) * SplitRecord::size;
    uint prefix = sum;
    uint total;
    threadgroup_exclusive_sum<threads, 1>(
        &prefix, &total, simd_lane_id, simd_group_id, bin_count_simdgroup_sums);
    MLX_MTL_PRAGMA_UNROLL
    for (int j = 0; j < bins_per_thread; j++) {
      record[SplitRecord::hist + first + j] = counts[j];
      record[SplitRecord::prefix + first + j] = prefix;
      prefix += counts[j];
    }
    if (lid.x == 0) {
      record[SplitRecord::outside_below] = radix_bin_count(outside, 0);
      record[SplitRecord::outside_above] = radix_bin_count(outside, 1);
    }
  }
}

// Writes the chunk of every group below | equal | above the threshold. Each
// side starts after the row counts and the records of the groups before.
template <
    typename T,
    typename U,
    bool ARG_PARTITION,
    int N_READS = radix_n_reads>
[[kernel, max_total_threads_per_threadgroup(split_partition_threads)]] void
split_partition(
    const device T* in [[buffer(0)]],
    device U* out [[buffer(1)]],
    const device uint* hist [[buffer(2)]],
    const device uint* records [[buffer(3)]],
    const constant int& axis_size [[buffer(4)]],
    const constant int& kth [[buffer(5)]],
    const constant int& n_passes [[buffer(6)]],
    uint3 gid [[threadgroup_position_in_grid]],
    uint3 gsize [[threadgroups_per_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]) {
  using KeyT = typename RadixKeyTraits<T>::KeyT;
  using Layout = SplitLayout;
  constexpr int threads = split_partition_threads;

  threadgroup uint threshold_side_count_simdgroup_sums
      [radix_threshold_sides * RadixLayout<threads>::scan_size];
  threadgroup KeyT shared_threshold;
  threadgroup uint shared_side_offset[radix_threshold_sides];

  int row = gid.y;
  int64_t offset = int64_t(row) * axis_size;
  in += offset;
  out += offset;

  // One simdgroup replays the passes and sums the records of the groups
  // before this one, its lanes take every 32nd record
  if (simd_group_id == 0) {
    auto state = radix_replay<KeyT, threads>(
        hist + int64_t(row) * n_passes * Layout::bins,
        n_passes,
        uint(kth + 1),
        simd_lane_id);
    uint digit = uint(state.prefix & KeyT(Layout::digit_mask));
    const device uint* row_records =
        records + int64_t(row) * gsize.x * SplitRecord::size;
    uint side_offset[radix_threshold_sides] = {0, 0, 0};
    for (uint g = simd_lane_id; g < gid.x; g += radix_simd_size) {
      split_record_sides(
          row_records + g * SplitRecord::size, digit, side_offset);
    }
    MLX_MTL_PRAGMA_UNROLL
    for (int c = 0; c < radix_threshold_sides; c++) {
      side_offset[c] = simd_sum(side_offset[c]);
    }
    if (simd_lane_id == 0) {
      shared_threshold = state.prefix;
      shared_side_offset[0] = side_offset[0];
      shared_side_offset[1] = state.n_below + side_offset[1];
      shared_side_offset[2] = state.n_below + state.n_equal + side_offset[2];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  KeyT threshold = shared_threshold;
  uint side_offset[radix_threshold_sides] = {
      shared_side_offset[0], shared_side_offset[1], shared_side_offset[2]};

  SplitChunk chunk = split_chunk(axis_size, gid.x, gsize.x);
  radix_partition_write<T, U, ARG_PARTITION, threads, N_READS>(
      in,
      out,
      chunk.start,
      chunk.end,
      threshold,
      side_offset,
      lid.x,
      simd_lane_id,
      simd_group_id,
      threshold_side_count_simdgroup_sums);
}
