#include "ATen/ATen.h"
#include "ATen/Dispatch.h"
#include "ATen/ExpandUtils.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtils.h"
#include "ATen/WrapDimUtilsMulti.h"
#include "cpu/ReduceOpsKernel.h"

#include <algorithm>
#include <functional>
#include <numeric>
#include <vector>

#include <map>

namespace at {
namespace native {

// ALL REDUCE #################################################################

Tensor _sum_cpu(const Tensor& self) {
  if (self.is_contiguous()) {
    Tensor result = self.type().tensor({});
    sum_kernel(result, self, at::nullopt);
    return result;
  }
  return self._sumall();
}

Tensor _prod_cpu(const Tensor &self) {
  if (self.is_contiguous()) {
    Tensor result = self.type().tensor({});
    prod_kernel(result, self, at::nullopt);
    return result;
  }
  return self._prodall();
}

Tensor _sum_cuda(const Tensor &self_) { return self_._sumall(); }

Tensor _prod_cuda(const Tensor &self_) { return self_._prodall(); }

// \ALL REDUCE ################################################################

// DIM REDUCE #################################################################

static bool _dimreduce_return_trivial(Tensor &result, const Tensor &self,
                                      int64_t ident) {
  if (self.numel() == 1 && self.ndimension() == 0) {
    result.resize_({});
    result.fill_(self);
    return true;
  }
  // Return identity
  if (self.numel() == 0 && self.ndimension() == 1) {
    result.resize_({0});
    result.fill_(ident);
    return true;
  }
  return false;
}

static Tensor &_dimreduce_setup(Tensor &result, const Tensor &self,
                                int64_t dim) {
  IntList self_sizes = self.sizes();
  std::vector<int64_t> result_sizes;
  result_sizes.insert(result_sizes.end(), self_sizes.begin(), self_sizes.end());
  result_sizes[dim] = 1;
  result.resize_(result_sizes);
  return result;
}

Tensor &_sum_out_cpu(Tensor &result, const Tensor &self, int64_t dim_,
                     bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  if (_dimreduce_return_trivial(result, self, 0))
    return result;
  if (self.is_contiguous() && result.is_contiguous()) {
    _dimreduce_setup(result, self, dim);
    sum_kernel(result, self, dim);
    if (!keepdim) result.squeeze_(dim);
    return result;
  }
  return at::_sum_out(result, self, dim, keepdim);
}

Tensor &_prod_out_cpu(Tensor &result, const Tensor &self, int64_t dim_,
                      bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  if (_dimreduce_return_trivial(result, self, 1))
    return result;
  if (self.is_contiguous() && result.is_contiguous()) {
    _dimreduce_setup(result, self, dim);
    prod_kernel(result, self, dim);
    if (!keepdim) result.squeeze_(dim);
    return result;
  }
  return at::_prod_out(result, self, dim, keepdim);
}

Tensor &_sum_out_cuda(Tensor &result, const Tensor &self, int64_t dim,
                      bool keepdim) {
  return at::_sum_out(result, self, dim, keepdim);
}

Tensor &_prod_out_cuda(Tensor &result, const Tensor &self, int64_t dim,
                       bool keepdim) {
  return at::_prod_out(result, self, dim, keepdim);
}

Tensor sum(const Tensor &self, int64_t dim_, bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  Tensor result = self.type().tensor();
  return at::sum_out(result, self, dim, keepdim);
}

Tensor prod(const Tensor &self, int64_t dim_, bool keepdim) {
  int64_t dim = maybe_wrap_dim(dim_, self.dim());
  Tensor result = self.type().tensor();
  return at::prod_out(result, self, dim, keepdim);
}

// \DIM REDUCE ################################################################

// MULTI DIM REDUCE ###########################################################

template <Tensor (reduce_1)(const Tensor &, int64_t, bool)>
inline Tensor reduce_multi(const Tensor &self, IntList dims_, bool keepdim) {
  if (dims_.size() == 1) {
    return reduce_1(self, dims_[0], keepdim);
  }
  if (dims_.size() == 0) {
    return self;
  }
  size_t ndims = self.dim();
  auto seen = dim_list_to_bitset(dims_, ndims);
  Tensor result = self;
  for (size_t i = 0; i < dims_.size(); i++) {
    size_t dim = maybe_wrap_dim(dims_[i], ndims);
    result = reduce_1(result, dim, true);
  }
  if (! keepdim) {
    size_t curdim = 0;
    for (size_t i = 0; i < ndims; i++) {
      if (seen[i]) {
	result.squeeze_(curdim);
      } else {
	curdim++;
      }
    }
  }
  return result;
}

template <Tensor (reduce_1)(const Tensor &, int64_t, bool),
	  Tensor& (reduce_1_out)(Tensor& result, const Tensor &, int64_t, bool)>
inline Tensor& reduce_multi_out(Tensor &result, const Tensor &self, IntList dims_, bool keepdim) {
  if (dims_.size() == 1) {
    return reduce_1_out(result, self, dims_[0], keepdim);
  }
  size_t ndims = self.dim();
  auto seen = dim_list_to_bitset(dims_, ndims);
  Tensor t = self;
  for (size_t i = 0; i < dims_.size(); i++) {
    auto dim = maybe_wrap_dim(dims_[i], ndims);
    if (i + 1 == dims_.size()) {
      reduce_1_out(result, t, dim, true);
    } else {
      t = reduce_1(t, dim, true);
    }
  }
  if (! keepdim) {
    size_t curdim = 0;
    for (size_t i = 0; i < ndims; i++) {
      if (seen[i]) {
	result.squeeze_(curdim);
      } else {
	curdim++;
      }
    }
  }
  return result;
}


Tensor& sum_out(Tensor &result, const Tensor &self, int64_t dim, bool keepdim) {
  if (self.is_cuda()) {
    return _sum_out_cuda(result, self, dim, keepdim);
  }
  else {
    return _sum_out_cpu(result, self, dim, keepdim);
  }
}

Tensor sum(const Tensor &self, IntList dims, bool keepdim) {
  return reduce_multi<sum>(self, dims, keepdim);
}

Tensor& sum_out(Tensor &result, const Tensor &self, IntList dims, bool keepdim)
{
  return reduce_multi_out<sum, sum_out>(result, self, dims, keepdim);
}

}} // namespace at::native
