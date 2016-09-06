#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void MinoConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  channels_ = bottom[0]->channels();
  num_output_ = this->layer_param_.mino_convolution_param().num_output();
  // bias_term_ = this->layer_param_.mino_convolution_param().bias_term();

  conv_out_channels_ = num_output_;
  conv_in_channels_ = channels_;

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    // if (bias_term_) {
    //   this->blobs_.resize(2);
    // } else {
      this->blobs_.resize(1);
    }
    // Initialize and fill the weights:
    // output channels x input channels per-group x kernel height x kernel width
    this->blobs_[0].reset(new Blob<Dtype>(4, conv_out_channels_/4, conv_in_channels_, 7));
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.mino_convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, initialize and fill the biases.
    // if (bias_term_) {
    //   vector<int> bias_shape(1, num_output_);
    //   this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
    //   shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
    //       this->layer_param_.mino_convolution_param().bias_filler()));
    //   bias_filler->Fill(this->blobs_[1].get());
    // }
  // }

  // Propagate gradients to the parameters (as directed by backward pass).
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void MinoConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();

  // Shape the tops.
  for (int top_id = 0; top_id < top.size(); ++top_id) {
    top[top_id]->Reshape(num_, num_output_, height_, width_);
  }

  conv_in_height_ = height_;
  conv_in_width_ = width_;
  conv_out_spatial_dim_ = height_ * width_;

  kernel_dim_ = conv_in_channels_ * 7;

  col_buffer_.Reshape(1, kernel_dim_, height_, width_);
  col_buffer_buffer_.Reshape(1, conv_in_channels_, height_, width_);

  // if (bias_term_) {
  //   vector<int> bias_multiplier_shape(1, height_ * width_);
  //   bias_multiplier_.Reshape(bias_multiplier_shape);
  //   caffe_set(bias_multiplier_.count(), Dtype(1),
  //       bias_multiplier_.mutable_cpu_data());
  // }
}


template <typename Dtype>
void MinoConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

}

template <typename Dtype>
void MinoConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
}

#ifdef CPU_ONLY
STUB_GPU(MinoConvolutionLayer);
#endif

INSTANTIATE_CLASS(MinoConvolutionLayer);
REGISTER_LAYER_CLASS(MinoConvolution);

}  // namespace caffe
