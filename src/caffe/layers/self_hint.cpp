#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SelfHintLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
	iteration_ = 1;
	max_step_ = this->layer_param_.self_hint_param().max_step();
	smoothness_ = this->layer_param_.self_hint_param().smoothness();
	limit_ = this->layer_param_.self_hint_param().limit();

	// if (this->blobs_.size() > 0) {
	// 	LOG(INFO) << "Skipping parameter initialization";
	// } else {
	// 	this->blobs_.resize(1);}
	// this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void SelfHintLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
	CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
			<< "corresponding to (num, channels, height, width)";
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();

	top[0]->Reshape(1,1,1,1);
}

template <typename Dtype>
void SelfHintLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void SelfHintLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}


#ifdef CPU_ONLY
STUB_GPU(SelfHintLayer);
#endif

INSTANTIATE_CLASS(SelfHintLayer);
REGISTER_LAYER_CLASS(SelfHint);

}  // namespace caffe
