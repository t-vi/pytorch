#pragma once

#include "ATen/TensorImpl.h"
#include "ATen/WrapDimUtils.h"
#include <sstream>
#include <bitset>
#include <algorithm>
#include <numeric>

namespace at {

// This is in an extra file to work around strange interaction of
// bitset on Windows with operator overloading

constexpr size_t dim_bitset_size = 64;

static inline std::bitset<dim_bitset_size> dim_list_to_bitset(IntList dims, int64_t ndims, bool wrap_scalar=true) {
  AT_ASSERT(ndims <= (int64_t) dim_bitset_size, "only tensors with up to %zu dims are supported", dim_bitset_size);
  std::bitset<dim_bitset_size> seen;
  for (size_t i = 0; i < dims.size(); i++) {
    size_t dim = maybe_wrap_dim(dims[i], ndims);
    AT_ASSERT(!seen[dim], "dim %zu appears multiple times in the list of dims", dim);
    seen[dim] = true;
  }
  return seen;
}

// this converts a list of reductions given in terms of initial dims to one that can be applied sequentially
static inline std::vector<int64_t> absolute_to_incremental_reductions(IntList dims, int64_t ndims, bool wrap_scalar=true) {
  std::vector<int64_t> initial_dims(ndims);
  std::vector<int64_t> incremental_dims;
  std::iota(initial_dims.begin(), initial_dims.end(), 0);
  for (size_t i = 0; i < dims.size(); i++) {
    int64_t dim = maybe_wrap_dim(dims[i], ndims);
    auto first_geq = std::lower_bound(initial_dims.begin(), initial_dims.end(), dim);
    AT_ASSERT(first_geq != initial_dims.end() && !(dim < *first_geq), "dim %zu appears multiple times in the list of dims", dim);
    incremental_dims.push_back(first_geq-initial_dims.begin());
    initial_dims.erase(first_geq);
  }
  return incremental_dims;
}

}
