#include "ATen/ATen.h"
#include "ATen/NativeFunctions.h"
#include "ATen/WrapDimUtilsMulti.h"

namespace at { namespace


native {

Tensor sumproduct_pair(const Tensor& left_, const Tensor& right_, IntList sum_dims_, bool keepdim) {
  // assumes that tensors have been pre-unsqueezed
  AT_ASSERT(left_.dim()==right_.dim(), "number of dimensions must match");
  if (sum_dims_.size() == 0)
    return at::mul(left_, right_);
  int64_t dim = left_.dim();
  auto sum_dims = dim_list_to_vector(sum_dims_, dim);
  std::vector<int64_t> lro, lo, ro;
  int64_t lro_size = 1, lo_size = 1, ro_size = 1, sum_size = 1;
  Tensor left = left_;
  Tensor right = right_;
  for (int64_t i = 0; i < dim; i++) {
    auto sl = left.size(i)>1;
    auto sr = right.size(i)>1;
    if (sum_dims[i]) {
      if (sl && sr) {
	AT_ASSERT(left.size(i)==right.size(i), "sum indexes must match");
	sum_size *= left.size(i);
      } else if (sl) {
	left = left.sum(i, true);
      } else if (sr) {
	right = right.sum(i, true);
      }
    } else if (sl && sr) {
      AT_ASSERT(left.size(i)==right.size(i), "non-broadcast dimensions must match");
      lro.push_back(i);
      lro_size *= left.size(i);
    } else if (sl) {
      lo.push_back(i);
      lo_size *= left.size(i);
    } else {
      ro.push_back(i);
      ro_size *= right.size(i);
    }
  }
  std::vector<int64_t> out_size;
  for (auto& d : lro) out_size.push_back(left.size(d));
  for (auto& d : lo) out_size.push_back(left.size(d));
  for (auto& d : ro) out_size.push_back(right.size(d));

  std::vector<int64_t> lpermutation(lro);
  lpermutation.insert(lpermutation.end(), lo.begin(), lo.end());
  lpermutation.insert(lpermutation.end(), sum_dims_.begin(), sum_dims_.end());
  lpermutation.insert(lpermutation.end(), ro.begin(), ro.end());

  std::vector<int64_t> rpermutation(lro);
  rpermutation.insert(rpermutation.end(), sum_dims_.begin(), sum_dims_.end());
  rpermutation.insert(rpermutation.end(), ro.begin(), ro.end());
  rpermutation.insert(rpermutation.end(), lo.begin(), lo.end());

  std::vector<int64_t> opermutation(lro.size()+lo.size()+ro.size(), -1);
  {
  int64_t i = 0;

  for (auto it = lro.begin(); it != lro.end(); i++, it++)
    opermutation[*it] = i;
  for (auto it = lo.begin(); it != lo.end(); i++, it++)
    opermutation[(*it)+lro.size()] = i;
  for (auto it = ro.begin(); it != ro.end(); i++, it++)
    opermutation[(*it)+lro.size()+lo.size()] = i;
  }

  left = left.permute(lpermutation).reshape({lro_size, lo_size, sum_size});
  right = right.permute(rpermutation).reshape({lro_size, sum_size, ro_size});
  Tensor result = at::bmm(left, right);
  result = result.view(out_size).reshape(opermutation);
  if (keepdim) {
    for (int i = 0; i < dim; i++)
      if (sum_dims[i])
	result.unsqueeze_(i);
  }
  return result;
}

    //static inline std::bitset<dim_bitset_size> dim_list_to_vector(IntList dims, int64_t ndims, bool wrap_scalar=true) {

Tensor trilinear(const Tensor& i1_, const Tensor& i2_, const Tensor& i3_,
		 IntList expand1_, IntList expand2_, IntList expand3_,
		 IntList sumdim_) {
  int64_t unroll_dim  = 1;
  int64_t total_dim = i1_.dim()+expand1_.size();
  std::cout << "totdim" << total_dim << std::endl;
  auto expand1 = dim_list_to_vector(expand1_, total_dim);
  auto expand2 = dim_list_to_vector(expand2_, total_dim);
  auto expand3 = dim_list_to_vector(expand3_, total_dim);
  auto sumdim  = dim_list_to_vector(sumdim_,  total_dim);
  Tensor i1 = i1_;
  Tensor i2 = i2_;
  Tensor i3 = i3_;
  std::vector<int64_t> output_size;
  std::vector<int64_t> sum_dims_12, sum_dims_23;
  int64_t unroll_size = -1;
  // asserts...
  for (int64_t i = 0; i < total_dim; i++) {
    int64_t s = 0;
    if (expand1[i]) {
      i1 = i1.unsqueeze(i);
    } else  {
      s = i1.size(i);
    }
    if (expand2[i]) {
      i2 = i2.unsqueeze(i);
    } else  {
      s = i2.size(i);
    }
    if (expand3[i]) {
      i3 = i3.unsqueeze(i);
      if (sumdim[i] && (i != unroll_dim))
	sum_dims_12.push_back(i);
    } else  {
      s = i3.size(i);
      if (sumdim[i] && (i != unroll_dim))
	sum_dims_23.push_back(i);
    }
    if (! sumdim[i])
      output_size.push_back(s);
    if (i == unroll_dim)
      unroll_size = s;
  }
  
  auto output = i1.type().tensor(output_size).zero_();
  if (! sumdim[unroll_dim]) {
    for (int64_t i = 0; i < unroll_size; i++) {
      Tensor buf = sumproduct_pair(i1, i2, sum_dims_12, true);
      buf = sumproduct_pair(buf, i3, sum_dims_23, true);
      output.add_(buf);
    }
  }
  else {
    std::cout << "hello4" << std::endl;
    for (int64_t i = 0; i < unroll_size; i++) {
      Tensor buf = sumproduct_pair(i1, i2, sum_dims_12, true);
      buf = sumproduct_pair(buf, i3, sum_dims_23, true);
      output.add_(buf);
    }
  }
  return output;
}

Tensor bilinear(const Tensor& input1, const Tensor& input2, const Tensor& weight, const Tensor& bias) {
  AT_ASSERT(input1.dim() == input2.dim(), "bilinear(): input dimensions do not match: got %lld and %lld",
            (long long)input1.dim(), (long long)input2.dim());
  for (int64_t i = 0; i < input1.dim() - 1; i++) {
    AT_ASSERT(input1.size(i) == input2.size(i),
              "bilinear(): input batch dimensions do not match at dim %lld: got %lld and %lld",
              (long long)i, (long long)input1.size(i), (long long)input2.size(i));
  }
  AT_ASSERT(input1.size(input1.dim() - 1) == weight.size(1),
            "bilinear(): input1 size does not match weight size: got %lld but expected %lld",
            (long long)input1.size(input1.dim() - 1), (long long)weight.size(1));
  AT_ASSERT(input2.size(input2.dim() - 1) == weight.size(2),
            "bilinear(): input2 size does not match weight size: got %lld but expected %lld",
            (long long)input2.size(input2.dim() - 1), (long long)weight.size(2));
  AT_ASSERT(!bias.defined() || bias.size(0) == weight.size(0),
            "bilinear(): bias size does not match weight size: got %lld but expected %lld",
            (long long)bias.size(0), (long long)weight.size(0));

  Tensor output = trilinear(input1, weight, input2, {1,3},{0},{1,2},{2,3});
  if (bias.defined()) {
    output = output + bias;
  }
  return output;
  
}

Tensor bilinear2(const Tensor& input1, const Tensor& input2, const Tensor& weight, const Tensor& bias) {
  AT_ASSERT(input1.dim() == input2.dim(), "bilinear(): input dimensions do not match: got %lld and %lld",
            (long long)input1.dim(), (long long)input2.dim());
  for (int64_t i = 0; i < input1.dim() - 1; i++) {
    AT_ASSERT(input1.size(i) == input2.size(i),
              "bilinear(): input batch dimensions do not match at dim %lld: got %lld and %lld",
              (long long)i, (long long)input1.size(i), (long long)input2.size(i));
  }
  AT_ASSERT(input1.size(input1.dim() - 1) == weight.size(1),
            "bilinear(): input1 size does not match weight size: got %lld but expected %lld",
            (long long)input1.size(input1.dim() - 1), (long long)weight.size(1));
  AT_ASSERT(input2.size(input2.dim() - 1) == weight.size(2),
            "bilinear(): input2 size does not match weight size: got %lld but expected %lld",
            (long long)input2.size(input2.dim() - 1), (long long)weight.size(2));
  AT_ASSERT(!bias.defined() || bias.size(0) == weight.size(0),
            "bilinear(): bias size does not match weight size: got %lld but expected %lld",
            (long long)bias.size(0), (long long)weight.size(0));

  std::vector<int64_t> output_size;
  auto size1 = input1.sizes();
  output_size.insert(output_size.end(), size1.begin(), size1.end() - 1);
  output_size.push_back(weight.size(0));

  auto output = input1.type().tensor(output_size);
  auto buf = input1.type().tensor(input2.sizes());

  size_t output_features = weight.size(0);
  auto input1_flattened = input1.view({-1, input1.size(-1)});
  auto buf_flattened = buf.view({-1, buf.size(-1)});
  for (size_t k = 0; k < output_features; k++) {
    at::mm_out(buf_flattened, input1_flattened, weight[k]);
    buf.mul_(input2);
    auto output_col = output.narrow(-1, k, 1);
    sum_out(output_col, buf, -1, true);
  }
  if (bias.defined()) {
    output = output + bias;
  }
  return output;
}

std::tuple<Tensor, Tensor, Tensor, Tensor> bilinear_backward(const Tensor& grad_out, const Tensor& input1, const Tensor& input2,
							     const Tensor& weight, std::array<bool, 4> grad_mask)
{
  Tensor grad_input1, grad_input2, grad_weight, grad_bias;

  size_t output_features = weight.size(0);
  auto input1_flattened = input1.view({-1, input1.size(-1)});
  auto input2_flattened = input2.view({-1, input2.size(-1)});
  auto grad_out_flattened = grad_out.view({-1, grad_out.size(-1)});

  if (grad_mask[0]) {
    grad_input1 = at::mm(input2_flattened, weight[0].t());
    grad_input1.mul_(grad_out_flattened.narrow(1, 0, 1));
    for (size_t k = 1; k < output_features; k++) {
      auto buf = input2_flattened.mm(weight[k].t());
      buf.mul_(grad_out_flattened.narrow(1, k, 1));
      grad_input1 += buf;
    }
    grad_input1 = grad_input1.view_as(input1);
  }
  if (grad_mask[1]) {
    grad_input2 = at::mm(input1_flattened, weight[0]);
    grad_input2.mul_(grad_out_flattened.narrow(1, 0, 1));
    for (size_t k = 1; k < output_features; k++) {
      auto buf = input1_flattened.mm(weight[k]);
      buf.mul_(grad_out_flattened.narrow(1, k, 1));
      grad_input2 += buf;
    }
    grad_input2 = grad_input2.view_as(input2);
  }
  if (grad_mask[2]) {
    grad_weight = weight.type().tensor(weight.sizes());
    for (size_t k = 0; k < output_features; k++) {
      auto buf = input1_flattened.mul(grad_out_flattened.narrow(1, k, 1));
      auto weight_row = grad_weight[k];
      at::mm_out(weight_row, buf.t(), input2_flattened);
    }
  }
  if (grad_mask[3]) {
    grad_bias = grad_out_flattened.sum(0, false);
  }
  return std::tuple<Tensor, Tensor, Tensor, Tensor>(grad_input1, grad_input2, grad_weight, grad_bias);
}

}}  // namespace at::native
