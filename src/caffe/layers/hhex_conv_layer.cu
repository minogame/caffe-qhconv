#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void HHexConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output) {
  const Dtype* col_buff = input;

  conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
  col_buff = col_buffer_.gpu_data();

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_,
      conv_out_spatial_dim_, kernel_dim_,
      (Dtype)1., weights, col_buff,
      (Dtype)0., output);
}

template <typename Dtype>
void BaseHexConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      height_out_ * width_out_, 1, (Dtype)1., bias, bias_multiplier_.gpu_data(),
      (Dtype)1., output);
}

template <typename Dtype>
void HHexConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
          top_data + top[i]->offset(n));
      // if (this->bias_term_) {
      //   const Dtype* bias = this->blobs_[1]->gpu_data();
      //   this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
      // }
    }
  }
}

template <typename Dtype>
void HHexConvolutionLayer<Dtype>::backward_cpu_gemm(const Dtype* output, const Dtype* weights, Dtype* input) {
  Dtype* col_buff = col_buffer_.mutable_gpu_data();
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, kernel_dim_,
      conv_out_spatial_dim_, conv_out_channels_,
      (Dtype)1., weights, output,
      (Dtype)0., col_buff);
  conv_col2im_gpu(col_buff, input);
}

template <typename Dtype>
void HHexConvolutionLayer<Dtype>::weight_cpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights) {
  const Dtype* col_buff = input;
  conv_im2col_gpu(input, col_buffer_.mutable_gpu_data());
  col_buff = col_buffer_.gpu_data();
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, conv_out_channels_,
      kernel_dim_, conv_out_spatial_dim_,
      (Dtype)1., output, col_buff,
      (Dtype)1., weights);
}

template <typename Dtype>
void BaseHexConvolutionLayer<Dtype>::backward_cpu_bias(Dtype* bias,
    const Dtype* input) {
  caffe_gpu_gemv<Dtype>(CblasNoTrans, num_output_, height_out_ * width_out_, 1.,
      input, bias_multiplier_.gpu_data(), 1., bias);
}

template <typename Dtype>
void HHexConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();

    // if (this->bias_term_ && this->param_propagate_down_[1]) {
    //   Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
    //   for (int n = 0; n < this->num_; ++n) {
    //     this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));
    //   }
    // }

    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + bottom[i]->offset(n),
              top_diff + top[i]->offset(n), weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
        //if (1) {
          this->backward_cpu_gemm(top_diff + top[i]->offset(n), weight,
              bottom_diff + bottom[i]->offset(n));
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(HHexConvolutionLayer);

}  // namespace caffe
