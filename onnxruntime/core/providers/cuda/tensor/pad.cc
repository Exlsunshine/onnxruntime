// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pad.h"
#include "pad_impl.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/cpu/tensor/pad.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(                        \
      Pad,                                                        \
      kOnnxDomain,                                                \
      2, 10,                                                      \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Pad<T>);                                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      Pad,                                                        \
      kOnnxDomain,                                                \
      11,                                                         \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .InputMemoryType<OrtMemTypeCPUInput>(1)                 \
          .InputMemoryType<OrtMemTypeCPUInput>(2)                 \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Pad<T>);

template <typename T>
typename ToCudaType<T>::MappedType ToCudaValue(const T& value) {
  return value;
}

template<>
typename ToCudaType<MLFloat16>::MappedType ToCudaValue<MLFloat16>(const MLFloat16& value) {
  return *reinterpret_cast<const typename ToCudaType<MLFloat16>::MappedType *>(&value.val);
}

template <typename T>
Status Pad<T>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const auto& input_tensor = *ctx->Input<Tensor>(0);
  auto const& input_shape = input_tensor.Shape();
  auto dimension_count = input_shape.NumDimensions();

  const std::vector<int64_t>* p_pads = &pads_;
  const std::vector<int64_t>* p_slices = &slices_;
  CudaT value = ToCudaType<T>::FromFloat(value_);

  // kOnnxDomain Pad opset >= 11 (Or) kMsDomain opset == 1
  std::vector<int64_t> pads;
  std::vector<int64_t> slices;
  if (is_dynamic_) {
    const Tensor& pads_tensor = *ctx->Input<Tensor>(1);
    const std::vector<int64_t>& pads_tensor_dims = pads_tensor.Shape().GetDims();
    ORT_ENFORCE(utils::IsPrimitiveDataType<int64_t>(pads_tensor.DataType()),
                "Pads tensor should be an INT64 tensor");
    ORT_ENFORCE(pads_tensor_dims.size() == 1 || (pads_tensor_dims.size() == 2 && pads_tensor_dims[0] == 1),
                "Pads tensor should be a 1D tensor of shape [2 * input_rank] or a 2D tensor of shape [1, 2 * input_rank]");

    const int64_t* pads_tensor_raw_data = pads_tensor.template Data<int64_t>();
    size_t pads_size = static_cast<size_t>(pads_tensor.Shape().Size());
    ORT_ENFORCE(pads_size == 2 * dimension_count,
                "Pads tensor size should be equal to twice the input dimension count ");

    pads.reserve(2 * dimension_count);
    for (size_t i = 0; i < pads_size; ++i) {
      pads.push_back(pads_tensor_raw_data[i]);
    }
    // Separate out any negative pads into the slices array
    slices.resize(pads.size(), 0);
    for (size_t index = 0; index < pads.size(); index++) {
      if (pads[index] < 0) {
        slices[index] = pads[index];
        pads[index] = 0;
      }
    }

    T raw_value(0);
    const Tensor* value_tensor = ctx->Input<Tensor>(2);
    if (nullptr != value_tensor) {
      ORT_ENFORCE(utils::IsPrimitiveDataType<T>(value_tensor->DataType()) &&
                      value_tensor->Shape().Size() == 1,
                  "Value tensor should be a 1D tensor of size 1 with the same type as that of the input tensor");
      raw_value = value_tensor->template Data<T>()[0];
      value = ToCudaValue<T>(raw_value);
    }
    p_pads = &pads;
    p_slices = &slices;
  }

  CudaAsyncBuffer<int64_t> input_dims(this, input_shape.GetDims());
  CudaAsyncBuffer<int64_t> input_strides(this, dimension_count);
  CudaAsyncBuffer<int64_t> lower_pads(this, dimension_count);
  CudaAsyncBuffer<int64_t> upper_pads(this, dimension_count);
  CudaAsyncBuffer<fast_divmod> fdm_output_strides(this, dimension_count);

  TensorPitches::Calculate(input_strides.CpuSpan(), input_shape.GetDims());
  std::vector<int64_t> output_dims(input_shape.GetDims());
  ORT_ENFORCE(dimension_count * 2 == p_pads->size(), "'pads' attribute has wrong number of values");

  // Calculate output dimensions, and handle any negative padding
  auto lower_pads_span = lower_pads.CpuSpan();
  auto upper_pads_span = upper_pads.CpuSpan();
  for (size_t i = 0; i < dimension_count; i++) {
    lower_pads_span[i] = (*p_pads)[i] + (*p_slices)[i];
    upper_pads_span[i] = (*p_pads)[i + dimension_count] + (*p_slices)[i + dimension_count];
    output_dims[i] += lower_pads_span[i] + upper_pads_span[i];
  }
  TensorShape output_shape(output_dims);

  // special case when there is a dim value of 0 in the shape. behavior depends on mode
  if (input_shape.Size() == 0) {
    ORT_RETURN_IF_ERROR(PadBase::HandleDimValueZero(mode_, input_shape, output_shape));
  }

  auto& output_tensor = *ctx->Output(0, output_shape);
  ORT_ENFORCE(CalculateFdmStrides(fdm_output_strides.CpuSpan(), output_dims));
  ORT_RETURN_IF_ERROR(input_dims.CopyToGpu());
  ORT_RETURN_IF_ERROR(input_strides.CopyToGpu());
  ORT_RETURN_IF_ERROR(lower_pads.CopyToGpu());
  ORT_RETURN_IF_ERROR(upper_pads.CopyToGpu());
  ORT_RETURN_IF_ERROR(fdm_output_strides.CopyToGpu());

  PadImpl(
      dimension_count,
      input_dims.GpuPtr(),
      input_strides.GpuPtr(),
      lower_pads.GpuPtr(),
      upper_pads.GpuPtr(),
      value,
      static_cast<int>(mode_),
      reinterpret_cast<const typename ToCudaType<T>::MappedType*>(input_tensor.template Data<T>()),
      fdm_output_strides.GpuPtr(),
      reinterpret_cast<typename ToCudaType<T>::MappedType*>(output_tensor.template MutableData<T>()),
      output_tensor.Shape().Size());

  return Status::OK();
}

#define SPECIALIZED_COMPUTE(T) \
  REGISTER_KERNEL_TYPED(T)     \
  template Status Pad<T>::ComputeInternal(OpKernelContext* ctx) const;

SPECIALIZED_COMPUTE(float)
SPECIALIZED_COMPUTE(double)
SPECIALIZED_COMPUTE(MLFloat16)

}  // namespace cuda
};  // namespace onnxruntime
