#include <algorithm>
#include <limits>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DropoutForward(const int n, const Dtype* bottom_data, const Dtype lamda,
		Dtype* theta_data, Dtype* top_data) {
	CUDA_KERNEL_LOOP(index, n) {
		//theta_data[index] = log (1 - theta_data[index]) / (-lamda);
		top_data[index] = bottom_data[index] * theta_data[index];
	}
}

template <typename Dtype>
__global__ void DropoutBackward(const int n,Dtype* bottom_diff, 
		const Dtype* top_diff, const Dtype* theta_data) {
	CUDA_KERNEL_LOOP(index, n) {
		bottom_diff[index] = top_diff[index] * theta_data[index];;
	}
}

template <typename Dtype>
void SDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const int count = bottom[0]->count();
	Dtype* theta_data = theta_.mutable_gpu_data();

	if (this->phase_ == TRAIN) {
		// Create theta & epsilon
		caffe_gpu_rng_gaussian(count, static_cast<Dtype>(1), static_cast<Dtype>(lamda_), theta_data);
		DropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			count, bottom_data, lamda_, theta_data, top_data);
	} else {
		caffe_copy(bottom[0]->count(), bottom_data, top_data);
	}
}

template <typename Dtype>
void SDropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

	const int count = bottom[0]->count();
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const Dtype* theta_data = theta_.gpu_data();

	if (propagate_down[0]) {
		if (this->phase_ == TRAIN) {
			DropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),	CAFFE_CUDA_NUM_THREADS>>>(
					count, bottom_diff, top_diff, theta_data);
		}
		else {
			caffe_copy(top[0]->count(), top_diff, bottom_diff);
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(SDropoutLayer);


}  // namespace caffe
