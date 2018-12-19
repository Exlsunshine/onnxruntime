// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace contrib {

class SequenceConvEmbedding : public OpKernel
{
public:
   explicit SequenceConvEmbedding( const OpKernelInfo& info ) : OpKernel( info )
   {
   }

   Status Compute( OpKernelContext* context ) const override;

private:
   void CharEmbeddingLookup( const int* sequence_p, const float* char_embedding_weight_p, size_t batch_size, size_t max_sequence_length, size_t max_word_length, size_t char_embedding_size, const int* sequence_length_p, float* dst ) const;
   void SequenceConvEmbedding::ComputeConvMaxPoolWithActivationAndMask(
      AllocatorPtr allocator,
      const float* input,
      const float* weights,
      const float* bias,
      const int* sequence_length_p, // [batch_size]
      const int* words_length_p,    // [batch_size, max_sequence_length]
      int64_t batch_size,
      int64_t max_sequence_length,
      int64_t max_word_length,
      int64_t char_embedding_size,
      int64_t filter_width,
      int64_t num_filters,
      float * output ) const;
   void CalculateSuquenceAndWordLength (
      const int* sequence_p,
      int* sequence_length_p,
      int* words_length_p,
      size_t batch_size,
      size_t max_sequence_length,
      size_t max_word_length ) const;
};

}  // namespace contrib
}  // namespace onnxruntime