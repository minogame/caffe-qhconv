// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SDropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
	NeuronLayer<Dtype>::LayerSetUp(bottom, top);
	lamda_ = this->layer_param_.stupid_dropout_param().lamda();
}

template <typename Dtype>
void SDropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  if (bottom[0] == top[0]) {
    bottom_memory_.ReshapeLike(*bottom[0]);
  }
  theta_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void SDropoutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int count = bottom[0]->count();
	Dtype* theta_data = theta_.mutable_cpu_data();

	if (this->phase_ == TRAIN) {
		// Create theta & epsilon
		caffe_rng_uniform(count, static_cast<Dtype>(0.00001), static_cast<Dtype>(0.999), theta_data);
		for (int i = 0; i < count; ++i)	{
			theta_data[i] = log (1 - theta_data[i]) / (-lamda_);
		}

		for (int i = 0; i < count; ++i)	{
			top_data[i] = bottom_data[i] * theta_data[i];
		}
	}	else {
		caffe_copy(bottom[0]->count(), bottom_data, top_data);
	}
}

template <typename Dtype>
void SDropoutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

	const int count = bottom[0]->count();
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* theta_data = theta_.cpu_data();

	if (propagate_down[0]) {
		if (this->phase_ == TRAIN) {
			for (int i = 0; i < count; ++i) {
				bottom_diff[i] = top_diff[i] * theta_data[i];
			}
		} else {
			caffe_copy(top[0]->count(), top_diff, bottom_diff);
		}
	}
}


#ifdef CPU_ONLY
STUB_GPU(SDropoutLayer);
#endif

INSTANTIATE_CLASS(SDropoutLayer);
REGISTER_LAYER_CLASS(SDropout);

}  // namespace caffe
