#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
void DetSoftmaxLossLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	// CHECK_EQ(bottom.size(),3) << "Needs 3 input blobs, bottom, mask, label.";
	// LossLayer<Dtype>::LayerSetUp(bottom, top);
	LayerParameter softmax_param(this->layer_param_);
	softmax_param.set_type("Softmax");
	buffer_.Reshape(bottom[1]->num(),bottom[0]->channels(),1,1);
	prob_.Reshape(bottom[1]->num(),bottom[0]->channels(),1,1);

	softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
	softmax_bottom_vec_.clear();
	softmax_bottom_vec_.push_back(&buffer_);
	softmax_top_vec_.clear();
	softmax_top_vec_.push_back(&prob_);
	softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
}

template <typename Dtype>
void DetSoftmaxLossLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	// LossLayer<Dtype>::Reshape(bottom, top);
	buffer_.Reshape(bottom[1]->num(),bottom[0]->channels(),1,1);
	prob_.Reshape(bottom[1]->num(),bottom[0]->channels(),1,1);
	// loss_buffer_.Reshape(bottom[1]->num(),bottom[0]->channels(),1,1);
	softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);

	outer_num_ = bottom[1]->num();
	inner_num_ = 1;
	vector<int> loss_shape(0);  // Loss layers output a scalar; 0 axes.
  top[0]->Reshape(loss_shape);
}

template <typename Dtype>
void DetSoftmaxLossLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void DetSoftmaxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(DetSoftmaxLossLayer);
#endif

INSTANTIATE_CLASS(DetSoftmaxLossLayer);
REGISTER_LAYER_CLASS(DetSoftmaxLoss);

}  // namespace caffe
