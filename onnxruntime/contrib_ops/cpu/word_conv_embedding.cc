#include "word_conv_embedding.h"

#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace contrib {

void SequenceConvEmbedding::CharEmbeddingLookup(
   const int* sequence_p,
   const float* char_embedding_weight_p,
   size_t batch_size,
   size_t max_sequence_length,
   size_t max_word_length,
   size_t char_embedding_size,
   const int* sequence_length_p,
   float* dst ) const
{
   const int* current_sequence_p;
   float* current_dst_p;
   for( size_t batch_index = 0; batch_index < batch_size; batch_index++ )
   {
      current_sequence_p = sequence_p + batch_index * max_sequence_length * max_word_length;
      current_dst_p = dst + batch_index * max_sequence_length * max_word_length * char_embedding_size;
      for( size_t word_inx = 0; word_inx < sequence_length_p[ batch_index ]; word_inx++ )
      {
         for( size_t char_inx = 0; char_inx < max_word_length; char_inx++ )
         {
            memcpy( current_dst_p, char_embedding_weight_p + ( *current_sequence_p ) * char_embedding_size, sizeof( float )* char_embedding_size );
            current_dst_p += char_embedding_size;
            current_sequence_p++;
         }
      }
   }
}

//input : [batch, sequence_length, word_length, char_embedding_size] 
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
   float * output ) const
{
   int64_t input_word_size = max_word_length * char_embedding_size;
   int64_t input_sequence_size = max_sequence_length* input_word_size;
   int64_t unfolded_width = max_word_length - filter_width + 1;
   int64_t unfolded_kernal_size = filter_width * char_embedding_size;
   int64_t unfolded_segment_size = unfolded_width * unfolded_kernal_size;
   //int conv_res_segment_size = _filter_width * _num_filters; // TODO: it imay be unfolded_width * _num_filters
   int64_t conv_res_segment_size = unfolded_width * num_filters;
   int64_t memcpy_size = unfolded_kernal_size * sizeof( float );

   IAllocatorUniquePtr<float> input_unfolded_buffer_p = IAllocator::MakeUniquePtr<float>( allocator, batch_size * max_sequence_length * unfolded_segment_size );
   IAllocatorUniquePtr<float> conv_result_p = IAllocator::MakeUniquePtr<float>( allocator, batch_size * max_sequence_length * conv_res_segment_size );
   IAllocatorUniquePtr<float> conv_activation_result_p = IAllocator::MakeUniquePtr<float>( allocator, batch_size * max_sequence_length * conv_res_segment_size );


   //   omp_set_nested( true );
   //   omp_set_dynamic( false );
   //   omp_set_num_threads( 24 );
   //
   //#pragma omp parallel for num_threads(nt)
   for( int batch_inx = 0; batch_inx < batch_size; batch_inx++ )
   {
      const float* current_seq_input = input + batch_inx * input_sequence_size;
      float* current_input_unfolded_buffer_p = input_unfolded_buffer_p.get() + batch_inx * max_sequence_length * unfolded_segment_size;
      float* current_conv_result_p = conv_result_p.get() + batch_inx * max_sequence_length * conv_res_segment_size;
      float* current_conv_activation_result_p = conv_activation_result_p.get() + batch_inx * max_sequence_length * conv_res_segment_size;
      float* current_output = output + batch_inx * max_sequence_length * num_filters;
      for( int64_t word_inx=0; word_inx < sequence_length_p[ batch_inx ]; word_inx++ )
      {
         const float * current_word_input = current_seq_input + word_inx * input_word_size;
         float * current_word_unfolded_buffer_p = current_input_unfolded_buffer_p + word_inx * unfolded_segment_size;
         float * conv_buf_p = current_conv_result_p + word_inx * conv_res_segment_size;
         float * pactivationbuf = current_conv_activation_result_p + word_inx * conv_res_segment_size;
         float * pres = current_output + word_inx * num_filters;

         // Unfolding from pin to pufbuf.
         float * pufbuf1 = current_word_unfolded_buffer_p;
         for( int64_t j = 0; j < unfolded_width; j++ )
         {
            memcpy( pufbuf1, current_word_input, memcpy_size );
            current_word_input += char_embedding_size;
            pufbuf1 += unfolded_kernal_size;
         }

         // Matrix multiplication.
         //mkl_set_num_threads_local( 1 );
         math::GemmEx<float, CPUMathUtil>(
            CblasNoTrans, CblasTrans,
            static_cast<int>( unfolded_width), static_cast<int>(num_filters), static_cast<int>( unfolded_kernal_size), 1.0f,
            current_word_unfolded_buffer_p, static_cast<int>( unfolded_kernal_size),
            weights, static_cast<int>( unfolded_kernal_size), 0.0f,
            conv_buf_p, static_cast<int>( num_filters), &CPUMathUtil::Instance() );
         //MatrixMult( current_word_unfolded_buffer_p, weights, conv_buf_p, unfolded_width, num_filters, unfolded_kernal_size );

         for( int64_t x = 0; x < unfolded_width; x++ )
            for( int64_t k = 0; k < num_filters; k++ )
            {
               conv_buf_p[ x * num_filters + k ] += bias[ k ];
            }

         MlasComputeTanh( conv_buf_p, pactivationbuf, unfolded_width * num_filters );

         // Max pooling.
         for( int64_t k = 0; k < num_filters; k++ )
         {
            pres[ k ] = -1.0f * 1e12f;
         }

         for( int64_t j = 0; j < unfolded_width; j++ )
         {
            if( j > 0 && j > ( words_length_p[ batch_inx * max_sequence_length + word_inx ] - filter_width ) ) break;
            float * pcur = ( float * ) pactivationbuf + j * num_filters;
#if USE_AVX
            int kLim8 = _num_filters - 7;
            float * ps = pcur;
            float * pd = pres;
            for(int64_t k=0 ; k < kLim8; k += 8, ps += 8, pd += 8 )
            {
               __m256 input = _mm256_loadu_ps( ps );
               __m256 res = _mm256_loadu_ps( pd );
               res = _mm256_max_ps( input, res );
               _mm256_storeu_ps( pd, res );
            }
#endif
            for( int64_t k=0; k < num_filters; k++ )
            {
               pres[ k ] = std::max( pcur[ k ], pres[ k ] );
            }
         }
      }
   }
}
void SequenceConvEmbedding::CalculateSuquenceAndWordLength  (
   const int* sequence_p,
   int* sequence_length_p,
   int* words_length_p,
   size_t batch_size,
   size_t max_sequence_length,
   size_t max_word_length) const
{
   for( size_t batch_inx = 0; batch_inx < batch_size; batch_inx++ )
   {
      int seq_length = 0;
      size_t b_off = batch_inx * max_sequence_length*max_word_length;
      for( size_t seq_inx = 0; seq_inx < max_sequence_length; seq_inx++ )
      {
         int word_length = 0;
         size_t w_off = seq_inx * max_word_length;
         if( sequence_p[ b_off + w_off ] > 0 ) seq_length++;
         for( size_t word_inx = 0; word_inx < max_word_length; word_inx++ )
         {
            if( sequence_p[ b_off + w_off + word_inx ] > 0 )word_length++;
         }
         words_length_p[ batch_inx* max_sequence_length + seq_inx ] = word_length;
      }
      sequence_length_p[ batch_inx ] = seq_length;
   }
}

Status SequenceConvEmbedding::Compute( OpKernelContext* ctx ) const
{
   // original lstm processing
   const Tensor& sequence = *(ctx->Input<Tensor>( 0 ));  // sequence: [batch_size, sequence_length, word_length]
   const Tensor& w_conv = *( ctx->Input<Tensor>( 1 ) );  // conv weight: [M, C/group, kH, kW]
   const Tensor& b_conv = *( ctx->Input<Tensor>( 2 ) );  // conv bias: [M]
   const Tensor& w_char_embedding = *( ctx->Input<Tensor>( 3 ) );  // conv weights. [index, char_embedding_size]

   const TensorShape& sequence_shape = sequence.Shape();

   int64_t batch_size = sequence_shape[ 0 ];;
   int64_t max_sequence_length = sequence_shape[ 1 ];
   int64_t max_word_length = sequence_shape[ 2 ];
   int64_t char_embedding_size = w_char_embedding.Shape()[1];
   int64_t filter_size = w_conv.Shape()[ 0 ];
   int64_t filter_width = w_conv.Shape()[ 2 ];

   TensorShape Y_dims { batch_size* max_sequence_length,  filter_size };
   Tensor* Y = ctx->Output(/*index*/ 0, Y_dims );

   const int* sequence_p = sequence.Data<int>(); 



   AllocatorPtr alloc;
   ORT_RETURN_IF_ERROR( ctx->GetTempSpaceAllocator( &alloc ));

   // allocate memery for char look up
   // batch_size * max_sequence_length * max_word_length * char_embedding_size 
   size_t chars_embeddings_size = batch_size * max_sequence_length * max_word_length * char_embedding_size;
   IAllocatorUniquePtr<float> chars_embeddings_p = IAllocator::MakeUniquePtr<float>( alloc, chars_embeddings_size );
   IAllocatorUniquePtr<int> words_length_p = IAllocator::MakeUniquePtr<int>( alloc, batch_size * max_sequence_length );
   IAllocatorUniquePtr<int> sequence_length_p = IAllocator::MakeUniquePtr<int>( alloc, batch_size );
   std::memset( chars_embeddings_p.get(), 0, chars_embeddings_size * sizeof( float ) );
   std::memset( words_length_p.get(), 0, batch_size * max_sequence_length * sizeof( int ) );
   std::memset( sequence_length_p.get(), 0, batch_size * sizeof( int ) );

   CalculateSuquenceAndWordLength( 
      sequence_p, 
      sequence_length_p.get(),
      words_length_p.get(),
      batch_size, 
      max_sequence_length,
      max_word_length );

   CharEmbeddingLookup( sequence_p,
      w_char_embedding.Data<float>(),
      batch_size,
      max_sequence_length,
      max_word_length,
      char_embedding_size,
      sequence_length_p.get(),
      chars_embeddings_p.get() );

   ComputeConvMaxPoolWithActivationAndMask(
      alloc,
      chars_embeddings_p.get(),
      w_conv.Data<float>(),
      b_conv.Data<float>(),
      sequence_length_p.get(),
      words_length_p.get(),
      batch_size,
      max_sequence_length,
      max_word_length,
      char_embedding_size,
      filter_width,
      filter_size,
      Y->MutableData<float>() );

   return Status::OK();
}

/* Range operator */
ONNX_OPERATOR_KERNEL_EX(
   SequenceConvEmbedding,  //name
   kMSDomain,
   1,
   kCpuExecutionProvider,
   KernelDefBuilder().TypeConstraint( "T", {
      DataTypeImpl::GetTensorType<int32_t>()} ).TypeConstraint("T1", {
         DataTypeImpl::GetTensorType<float>() }),
      SequenceConvEmbedding );

}
}