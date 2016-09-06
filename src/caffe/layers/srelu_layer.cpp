#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  SReLUParameter srelu_param = this->layer_param().srelu_param();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    //mean & sqrt of variance
    this->blobs_[0].reset(new Blob<Dtype>(1,1,1,2));
    Dtype* hyperparam = this->blobs_[0]->mutable_cpu_data();
    hyperparam[0] = this->layer_param_.srelu_param().initial_mean();
    hyperparam[1] = this->layer_param_.srelu_param().initial_var();
    threshold_lr = this->layer_param_.srelu_param().threshold_lr();
    showvalue = this->layer_param_.srelu_param().show_value();
    variance_recover = this->layer_param_.srelu_param().variance_recover();
    stop_updata = this->layer_param_.srelu_param().stop_updata();
    this->iter = 0;
  }
}

template <typename Dtype>
void SReLULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_GE(bottom[0]->num_axes(), 2)
      << "Number of axes of bottom blob must be >=2.";
  top[0]->ReshapeLike(*bottom[0]);
  // For in-place computation
  if (bottom[0] == top[0]) {
    bottom_memory_.ReshapeLike(*bottom[0]);
  }
  threshold_buffer_.ReshapeLike(*bottom[0]);
  bp_diff_buffer_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void SReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype* th_data = threshold_buffer_.mutable_cpu_data();
  const Dtype* hyperparam = this->blobs_[0]->cpu_data();

  // For in-place computation
  if (bottom[0] == top[0]) {
    caffe_copy(count, bottom_data, bottom_memory_.mutable_cpu_data());
  }

  // Create threshold buffer vector
  caffe_rng_gaussian(count, hyperparam[0], hyperparam[1], th_data);

  // if channel_shared, channel index in the following computation becomes
  // always zero.
  for (int i = 0; i < count; ++i) {
    top_data[i] = (bottom_data[i] >= th_data[i]) ? bottom_data[i] : 0;
  }
}

template <typename Dtype>
void SReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* top_diff = top[0]->cpu_diff();
  const int count = bottom[0]->count();
  Dtype* th_data = threshold_buffer_.mutable_cpu_data();
  Dtype* hyperparam = this->blobs_[0]->mutable_cpu_data();

  // For in-place computation
  if (top[0] == bottom[0]) {
    bottom_data = bottom_memory_.cpu_data();
  }

  // Propagte to param
  if (!stop_updata){
    if (!(this->iter%this->showvalue)) {
  for (int i = 0; i < count; ++i) {
    Dtype th_diff;
    if (bottom_data[i] <= 0) {
      th_diff = 0;
    }
    else {
      th_diff = (1 - exp(bottom_data[i])/(exp(bottom_data[i]) + exp(th_data[i]))) * top_diff[i];
    }
    th_data[i] -= threshold_lr * th_diff;
  }

  Dtype ave = 0;
  for (int i = 0; i < count; ++i)  {
    ave += th_data[i];
  }
  ave /= bottom[0]->count();
  hyperparam[0] = ave; 

  Dtype var = 0;
  for (int i = 0; i < count; ++i)  {
    var += (ave - th_data[i]) * (ave - th_data[i]);
  }
  hyperparam[1] = sqrt(var / bottom[0]->count()) * variance_recover;

  
    LOG(INFO)<<"MEAN: "<<hyperparam[0]<<"    VAR:"<<hyperparam[1];}
  this->iter++;}


  // Propagate to bottom
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * (bottom_data[i] > 0);
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(SReLULayer);
#endif

INSTANTIATE_CLASS(SReLULayer);
REGISTER_LAYER_CLASS(SReLU);

}  // namespace caffe
