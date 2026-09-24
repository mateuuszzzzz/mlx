// Copyright © 2023-2024 Apple Inc.

#include <algorithm>

#include "mlx/backend/gpu/copy.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/kernels.h"
#include "mlx/backend/metal/kernels/defines.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/primitives.h"

namespace mlx::core {

namespace {

void single_block_sort(
    const Stream& s,
    metal::Device& d,
    const array& in,
    array& out,
    int axis,
    int bn,
    int tn,
    bool argsort) {
  // Prepare shapes
  int n_rows = in.size() / in.shape(axis);

  auto in_nc_str = in.strides();
  in_nc_str.erase(in_nc_str.begin() + axis);

  auto out_nc_str = out.strides();
  out_nc_str.erase(out_nc_str.begin() + axis);

  auto nc_shape = in.shape();
  nc_shape.erase(nc_shape.begin() + axis);

  int nc_dim = nc_shape.size();

  int size_sorted_axis = in.shape(axis);
  int in_stride_sorted_axis = in.strides()[axis];
  int out_stride_sorted_axis = out.strides()[axis];

  // We can only use the contiguous kernel if the sorted axis
  // has the largest or smallest stride.
  // We also need the input to be contiguous
  bool contiguous = in.flags().contiguous;
  auto check_strides = [](array x, int sort_stride) {
    int min_stride = *std::min_element(x.strides().begin(), x.strides().end());
    int max_stride = *std::max_element(x.strides().begin(), x.strides().end());
    return sort_stride == min_stride || sort_stride == max_stride;
  };
  contiguous &= check_strides(in, in_stride_sorted_axis);
  contiguous &= check_strides(out, out_stride_sorted_axis);

  // Prepare kernel name
  std::ostringstream kname;
  kname << (contiguous ? "c" : "nc");
  if (argsort) {
    kname << "arg";
  }

  kname << "_block_sort_" << type_to_name(in) << "_" << type_to_name(out)
        << "_bn" << bn << "_tn" << tn;
  auto kernel = get_sort_kernel(d, kname.str(), in, out, bn, tn);

  // Prepare command encoder
  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);

  // Set inputs
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  compute_encoder.set_bytes(size_sorted_axis, 2);
  compute_encoder.set_bytes(in_stride_sorted_axis, 3);
  compute_encoder.set_bytes(out_stride_sorted_axis, 4);

  if (contiguous) {
    int in_stride_segment_axis = INT32_MAX;
    int out_stride_segment_axis = INT32_MAX;
    for (int i = 0; i < in_nc_str.size(); i++) {
      if (nc_shape[i] == 1) {
        continue;
      }
      if (in_nc_str[i] > INT32_MAX || out_nc_str[i] > INT32_MAX) {
        throw std::runtime_error("[Sort::eval_gpu] Stride too large.");
      }
      in_stride_segment_axis =
          std::min(in_stride_segment_axis, static_cast<int>(in_nc_str[i]));
      out_stride_segment_axis =
          std::min(out_stride_segment_axis, static_cast<int>(out_nc_str[i]));
    }
    compute_encoder.set_bytes(in_stride_segment_axis, 5);
    compute_encoder.set_bytes(out_stride_segment_axis, 6);
  } else {
    compute_encoder.set_bytes(nc_dim, 5);
    if (nc_shape.empty()) {
      int shape = 0;
      int64_t stride = 0;
      compute_encoder.set_bytes(shape, 6);
      compute_encoder.set_bytes(stride, 7);
      compute_encoder.set_bytes(stride, 8);
    } else {
      compute_encoder.set_vector_bytes(nc_shape, 6);
      compute_encoder.set_vector_bytes(in_nc_str, 7);
      compute_encoder.set_vector_bytes(out_nc_str, 8);
    }
  }

  MTL::Size group_dims = MTL::Size(bn, 1, 1);
  MTL::Size grid_dims = MTL::Size(1, n_rows, 1);

  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void multi_block_sort(
    const Stream& s,
    metal::Device& d,
    const array& in,
    array& out,
    int axis,
    int bn,
    int tn,
    int n_blocks,
    bool argsort) {
  // Prepare shapes
  int n_rows = in.size() / in.shape(axis);

  auto nc_str = in.strides();
  nc_str.erase(nc_str.begin() + axis);

  auto nc_shape = in.shape();
  nc_shape.erase(nc_shape.begin() + axis);

  int nc_dim = nc_shape.size();

  if (nc_dim == 0) {
    nc_shape = {0};
    nc_str = {1};
  }

  int size_sorted_axis = in.shape(axis);
  int stride_sorted_axis = in.strides()[axis];

  // Make temporary copies
  array dev_vals_0({n_rows, size_sorted_axis}, in.dtype(), nullptr, {});
  array dev_vals_1({n_rows, size_sorted_axis}, in.dtype(), nullptr, {});

  array dev_idxs_0({n_rows, size_sorted_axis}, uint32, nullptr, {});
  array dev_idxs_1({n_rows, size_sorted_axis}, uint32, nullptr, {});

  array block_partitions({n_rows, n_blocks + 1}, uint32, nullptr, {});

  // Do allocations
  dev_vals_0.set_data(allocator::malloc(dev_vals_0.nbytes()));
  dev_vals_1.set_data(allocator::malloc(dev_vals_1.nbytes()));
  dev_idxs_0.set_data(allocator::malloc(dev_idxs_0.nbytes()));
  dev_idxs_1.set_data(allocator::malloc(dev_idxs_1.nbytes()));
  block_partitions.set_data(allocator::malloc(block_partitions.nbytes()));

  std::vector<array> copies = {
      dev_vals_0, dev_vals_1, dev_idxs_0, dev_idxs_1, block_partitions};

  // Prepare command encoder
  auto& compute_encoder = metal::get_command_encoder(s);

  // Do blockwise sort
  {
    std::ostringstream kname;
    kname << "sort_mbsort_" << type_to_name(dev_vals_0) << "_"
          << type_to_name(dev_idxs_0) << "_bn" << bn << "_tn" << tn;
    auto kernel =
        get_mb_sort_kernel(d, kname.str(), dev_vals_0, dev_idxs_0, bn, tn);
    compute_encoder.set_compute_pipeline_state(kernel);

    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_output_array(dev_vals_0, 1);
    compute_encoder.set_output_array(dev_idxs_0, 2);
    compute_encoder.set_bytes(size_sorted_axis, 3);
    compute_encoder.set_bytes(stride_sorted_axis, 4);
    compute_encoder.set_bytes(nc_dim, 5);
    compute_encoder.set_vector_bytes(nc_shape, 6);
    compute_encoder.set_vector_bytes(nc_str, 7);

    MTL::Size group_dims = MTL::Size(bn, 1, 1);
    MTL::Size grid_dims = MTL::Size(n_blocks, n_rows, 1);

    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  // Do merges
  bool ping = false;
  array dev_vals_in = dev_vals_0;
  array dev_idxs_in = dev_idxs_0;
  array dev_vals_out = dev_vals_1;
  array dev_idxs_out = dev_idxs_1;

  int n_thr_per_group = (n_blocks + 1) < 1024 ? (n_blocks + 1) : 1024;

  for (int merge_tiles = 2; (merge_tiles / 2) < n_blocks; merge_tiles *= 2) {
    dev_vals_in = ping ? dev_vals_1 : dev_vals_0;
    dev_idxs_in = ping ? dev_idxs_1 : dev_idxs_0;
    dev_vals_out = ping ? dev_vals_0 : dev_vals_1;
    dev_idxs_out = ping ? dev_idxs_0 : dev_idxs_1;
    ping = !ping;

    // Do partition
    {
      std::ostringstream kname;
      kname << "partition_mbsort_" << type_to_name(dev_vals_in) << "_"
            << type_to_name(dev_idxs_in) << "_bn" << bn << "_tn" << tn;

      auto kernel =
          get_mb_sort_kernel(d, kname.str(), dev_vals_0, dev_idxs_0, bn, tn);
      compute_encoder.set_compute_pipeline_state(kernel);

      compute_encoder.set_output_array(block_partitions, 0);
      compute_encoder.set_input_array(dev_vals_in, 1);
      compute_encoder.set_input_array(dev_idxs_in, 2);
      compute_encoder.set_bytes(size_sorted_axis, 3);
      compute_encoder.set_bytes(merge_tiles, 4);
      compute_encoder.set_bytes(n_blocks, 5);

      MTL::Size group_dims = MTL::Size(n_thr_per_group, 1, 1);
      MTL::Size grid_dims = MTL::Size(1, n_rows, 1);

      compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
    }

    // Do merge
    {
      std::ostringstream kname;
      kname << "merge_mbsort_" << type_to_name(dev_vals_in) << "_"
            << type_to_name(dev_idxs_in) << "_bn" << bn << "_tn" << tn;

      auto kernel =
          get_mb_sort_kernel(d, kname.str(), dev_vals_0, dev_idxs_0, bn, tn);
      compute_encoder.set_compute_pipeline_state(kernel);

      compute_encoder.set_input_array(block_partitions, 0);
      compute_encoder.set_input_array(dev_vals_in, 1);
      compute_encoder.set_input_array(dev_idxs_in, 2);
      compute_encoder.set_output_array(dev_vals_out, 3);
      compute_encoder.set_output_array(dev_idxs_out, 4);
      compute_encoder.set_bytes(size_sorted_axis, 5);
      compute_encoder.set_bytes(merge_tiles, 6);
      compute_encoder.set_bytes(n_blocks, 7);

      MTL::Size group_dims = MTL::Size(bn, 1, 1);
      MTL::Size grid_dims = MTL::Size(n_blocks, n_rows, 1);

      compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
    }
  }

  // Copy outputs with appropriate strides
  auto strides = out.strides();
  for (int ax = axis + 1; ax < strides.size(); ax++) {
    strides[ax] *= out.shape(axis);
  }
  strides[axis] = 1;
  copy_gpu_inplace(
      (argsort) ? dev_idxs_out : dev_vals_out,
      out,
      out.shape(),
      strides,
      out.strides(),
      0,
      0,
      (axis == in.ndim() - 1) ? CopyType::Vector : CopyType::General,
      s);

  compute_encoder.add_temporaries(std::move(copies));
}

void gpu_merge_sort(
    const Stream& s,
    metal::Device& d,
    const array& in,
    array& out,
    int axis_,
    bool argsort) {
  if (out.size() == 0) {
    return;
  }

  // Get size info
  int axis = axis_ < 0 ? axis_ + in.ndim() : axis_;
  int size_sorted_axis = in.shape(axis);

  // Get kernel size
  int tn = 4;
  int potential_bn = (size_sorted_axis + tn - 1) / tn;

  int bn;
  if (potential_bn > 256) {
    bn = 512;
  } else if (potential_bn > 128) {
    bn = 256;
  } else if (potential_bn > 64) {
    bn = 128;
  } else if (potential_bn > 32) {
    bn = 64;
  } else {
    bn = 32;
  }

  if (bn == 512 && size_of(in.dtype()) > 4) {
    bn = 256;
  }

  int n_per_block = bn * tn;
  int n_blocks = (size_sorted_axis + n_per_block - 1) / n_per_block;

  if (n_blocks > 1) {
    return multi_block_sort(s, d, in, out, axis, bn, tn, n_blocks, argsort);
  } else {
    return single_block_sort(s, d, in, out, axis, bn, tn, argsort);
  }
}

// Radix select beats merge sort from this row length on, rows up to 128
// elements sort inside one simdgroup (measured on M3 Pro).
constexpr int RADIX_PARTITION_MIN_AXIS_SIZE = 129;

// One simdgroup per row beats both while the row fits in 16 elements per
// lane and the rounds times the elements per lane stay small, each round
// scans the lane and removes one element (measured on M3 Pro).
constexpr int SIMD_PARTITION_MAX_AXIS_SIZE = 512;
constexpr int SIMD_PARTITION_MAX_LANE_WORK = 96;

// A few very long rows get several threadgroups per row, one histogram
// dispatch per digit plus the write (measured on M3 Pro).
constexpr int SPLIT_PARTITION_MIN_AXIS_SIZE = 32768;
constexpr int SPLIT_PARTITION_MAX_ROWS = 8;
constexpr int SPLIT_PARTITION_MAX_GROUPS = 32;
constexpr int SPLIT_PARTITION_TARGET_GROUPS = 64;
constexpr int SPLIT_PARTITION_ELEMENTS_PER_GROUP = 4096;

bool use_radix_partition(const array& in, int axis) {
  if (axis != in.ndim() - 1 || !in.flags().row_contiguous) {
    return false;
  }
  if (in.dtype() == bool_ || in.dtype() == complex64) {
    return false;
  }
  return in.shape(axis) >= RADIX_PARTITION_MIN_AXIS_SIZE;
}

// Elements per lane of the simdgroup kernel, the smallest instantiated size
// that covers the row
int simd_partition_n_per(int axis_size) {
  constexpr int simd_size = 32;
  int n_per = 4;
  while (n_per * simd_size < axis_size) {
    n_per *= 2;
  }
  return n_per;
}

bool use_simd_partition(const array& in, int axis, int kth) {
  if (axis != in.ndim() - 1 || !in.flags().row_contiguous) {
    return false;
  }
  if (in.dtype() == bool_ || in.dtype() == complex64 ||
      size_of(in.dtype()) > 4) {
    return false;
  }
  int axis_size = in.shape(axis);
  int rounds = std::min(kth + 1, axis_size - kth);
  return axis_size <= SIMD_PARTITION_MAX_AXIS_SIZE &&
      rounds * simd_partition_n_per(axis_size) <= SIMD_PARTITION_MAX_LANE_WORK;
}

bool use_split_partition(const array& in, int axis) {
  if (axis != in.ndim() - 1 || !in.flags().row_contiguous) {
    return false;
  }
  if (in.dtype() == bool_ || in.dtype() == complex64) {
    return false;
  }
  int axis_size = in.shape(axis);
  int n_rows = in.size() / axis_size;
  return axis_size >= SPLIT_PARTITION_MIN_AXIS_SIZE &&
      n_rows <= SPLIT_PARTITION_MAX_ROWS;
}

void gpu_split_partition(
    const Stream& s,
    metal::Device& d,
    const array& in,
    array& out,
    int axis,
    int kth,
    bool arg_partition) {
  int axis_size = in.shape(axis);
  int n_rows = in.size() / axis_size;
  constexpr int digit_bits = RADIX_PARTITION_WIDE_DIGIT_BITS;
  int key_bits = 8 * size_of(in.dtype());
  int n_passes = (key_bits + digit_bits - 1) / digit_bits;
  // Groups per row, enough groups in total without making the chunks tiny
  int groups = std::min(
      {SPLIT_PARTITION_MAX_GROUPS,
       std::max(1, SPLIT_PARTITION_TARGET_GROUPS / n_rows),
       (axis_size + SPLIT_PARTITION_ELEMENTS_PER_GROUP - 1) /
           SPLIT_PARTITION_ELEMENTS_PER_GROUP});

  // Histogram history of every row and the record of every group
  array hist({n_rows, n_passes, 1 << digit_bits}, uint32, nullptr, {});
  array records(
      {n_rows, groups, SPLIT_PARTITION_RECORD_SIZE}, uint32, nullptr, {});
  hist.set_data(allocator::malloc(hist.nbytes()));
  records.set_data(allocator::malloc(records.nbytes()));
  array zero = array(0, uint32);
  fill_gpu(zero, hist, s);

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.add_temporary(std::move(zero));
  compute_encoder.add_temporary(hist);
  compute_encoder.add_temporary(records);

  MTL::Size group_dims = MTL::Size(SPLIT_PARTITION_THREADS, 1, 1);
  MTL::Size grid_dims = MTL::Size(groups, n_rows, 1);

  std::string hist_name = "split_partition_histogram_";
  concatenate(hist_name, type_to_name(in));
  auto hist_kernel = get_split_partition_histogram_kernel(d, hist_name, in);
  for (int pass = 0; pass < n_passes; pass++) {
    compute_encoder.set_compute_pipeline_state(hist_kernel);
    compute_encoder.set_input_array(in, 0);
    compute_encoder.set_input_array(hist, 1);
    compute_encoder.set_output_array(hist, 1);
    compute_encoder.set_output_array(records, 2);
    compute_encoder.set_bytes(axis_size, 3);
    compute_encoder.set_bytes(kth, 4);
    compute_encoder.set_bytes(pass, 5);
    compute_encoder.set_bytes(n_passes, 6);
    compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
  }

  std::string kernel_name =
      arg_partition ? "split_argpartition_" : "split_partition_";
  concatenate(kernel_name, type_to_name(in));
  auto kernel =
      get_split_partition_kernel(d, kernel_name, in, out, arg_partition);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  compute_encoder.set_input_array(hist, 2);
  compute_encoder.set_input_array(records, 3);
  compute_encoder.set_bytes(axis_size, 4);
  compute_encoder.set_bytes(kth, 5);
  compute_encoder.set_bytes(n_passes, 6);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gpu_simd_partition(
    const Stream& s,
    metal::Device& d,
    const array& in,
    array& out,
    int axis,
    int kth,
    bool arg_partition) {
  int axis_size = in.shape(axis);
  int n_rows = in.size() / axis_size;

  constexpr int simd_size = 32;
  int n_per = simd_partition_n_per(axis_size);
  std::string kernel_name =
      arg_partition ? "simd_argpartition_" : "simd_partition_";
  concatenate(kernel_name, type_to_name(in), "_n", n_per);
  auto kernel =
      get_simd_partition_kernel(d, kernel_name, in, out, arg_partition, n_per);

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  compute_encoder.set_bytes(axis_size, 2);
  compute_encoder.set_bytes(kth, 3);
  compute_encoder.set_bytes(n_rows, 4);

  int n_groups = (n_rows + SIMD_PARTITION_ROWS_PER_GROUP - 1) /
      SIMD_PARTITION_ROWS_PER_GROUP;
  MTL::Size group_dims =
      MTL::Size(SIMD_PARTITION_ROWS_PER_GROUP * simd_size, 1, 1);
  MTL::Size grid_dims = MTL::Size(1, n_groups, 1);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

void gpu_radix_partition(
    const Stream& s,
    metal::Device& d,
    const array& in,
    array& out,
    int axis,
    int kth,
    bool arg_partition) {
  int axis_size = in.shape(axis);
  int n_rows = in.size() / axis_size;

  // Threads per row, so each thread gets a few elements per pass
  int bn = 256;
  if (axis_size < 1024) {
    bn = 64;
  } else if (axis_size < 4096) {
    bn = 128;
  }

  std::string kernel_name =
      arg_partition ? "radix_argpartition_" : "radix_partition_";
  concatenate(kernel_name, type_to_name(in), "_bn", bn);
  auto kernel =
      get_partition_kernel(d, kernel_name, in, out, arg_partition, bn);

  auto& compute_encoder = metal::get_command_encoder(s);
  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(in, 0);
  compute_encoder.set_output_array(out, 1);
  compute_encoder.set_bytes(axis_size, 2);
  compute_encoder.set_bytes(kth, 3);

  MTL::Size group_dims = MTL::Size(bn, 1, 1);
  MTL::Size grid_dims = MTL::Size(1, n_rows, 1);
  compute_encoder.dispatch_threadgroups(grid_dims, group_dims);
}

} // namespace

void ArgSort::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  out.set_data(allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& in = inputs[0];

  gpu_merge_sort(s, d, in, out, axis_, true);
}

void Sort::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  out.set_data(allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& in = inputs[0];

  gpu_merge_sort(s, d, in, out, axis_, false);
}

void ArgPartition::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  out.set_data(allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& in = inputs[0];

  if (use_simd_partition(in, axis_, kth_)) {
    gpu_simd_partition(s, d, in, out, axis_, kth_, true);
  } else if (use_split_partition(in, axis_)) {
    gpu_split_partition(s, d, in, out, axis_, kth_, true);
  } else if (use_radix_partition(in, axis_)) {
    gpu_radix_partition(s, d, in, out, axis_, kth_, true);
  } else {
    gpu_merge_sort(s, d, in, out, axis_, true);
  }
}

void Partition::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 1);

  out.set_data(allocator::malloc(out.nbytes()));

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& in = inputs[0];

  if (use_simd_partition(in, axis_, kth_)) {
    gpu_simd_partition(s, d, in, out, axis_, kth_, false);
  } else if (use_split_partition(in, axis_)) {
    gpu_split_partition(s, d, in, out, axis_, kth_, false);
  } else if (use_radix_partition(in, axis_)) {
    gpu_radix_partition(s, d, in, out, axis_, kth_, false);
  } else {
    gpu_merge_sort(s, d, in, out, axis_, false);
  }
}

void SearchSorted::eval_gpu(const std::vector<array>& inputs, array& out) {
  assert(inputs.size() == 2);
  auto& a = inputs[0];
  auto v = inputs[1];

  out.set_data(allocator::malloc(out.nbytes()));
  if (out.size() == 0) {
    return;
  }

  auto& s = stream();
  auto& d = metal::device(s.device);
  auto& compute_encoder = metal::get_command_encoder(s);

  if (a.size() == 0) {
    array zero = array(0, out.dtype());
    fill_gpu(zero, out, s);
    compute_encoder.add_temporary(std::move(zero));
    return;
  }

  if (!v.flags().row_contiguous) {
    v = contiguous_copy_gpu(v, s);
    compute_encoder.add_temporary(v);
  }

  int64_t a_stride = a.strides()[0]; // sequence is 1D
  auto n = static_cast<uint32_t>(a.size());

  std::string kernel_name = "searchsorted_";
  concatenate(kernel_name, type_to_name(a), right_ ? "_right" : "_left");
  auto kernel = get_searchsorted_kernel(d, kernel_name, a, right_);

  compute_encoder.set_compute_pipeline_state(kernel);
  compute_encoder.set_input_array(a, 0);
  compute_encoder.set_input_array(v, 1);
  compute_encoder.set_output_array(out, 2);
  compute_encoder.set_bytes(n, 3);
  compute_encoder.set_bytes(a_stride, 4);

  size_t thread_group_size = kernel->maxTotalThreadsPerThreadgroup();
  thread_group_size = std::min(thread_group_size, out.size());
  MTL::Size group_dims = MTL::Size(thread_group_size, 1, 1);
  MTL::Size grid_dims = get_2d_grid_dims(out.shape(), out.strides());
  compute_encoder.dispatch_threads(grid_dims, group_dims);
}

} // namespace mlx::core
