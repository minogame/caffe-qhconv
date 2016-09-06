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
__global__ void DropoutForward(const int n, const int K, const Dtype* bottom_data, const Dtype* p_data, const Dtype* lamda_data,
		Dtype* theta_data, Dtype* epsilon_data, Dtype* top_data) {
	CUDA_KERNEL_LOOP(index, n) {
		const int pos = index % K;
		theta_data[index] = log (1 - theta_data[index]) / (-lamda_data[pos]);
		epsilon_data[index] = asinh(p_data[pos]/(2 * exp(-lamda_data[pos] * bottom_data[index]))) / lamda_data[pos];
		if ((bottom_data[index] - epsilon_data[index]) < theta_data[index] &&
			 (bottom_data[index] + epsilon_data[index]) > theta_data[index])
			top_data[pos] = bottom_data[index] / ( 1 - p_data[pos]);
		else top_data[pos] = 0;
	}
}

template <typename Dtype>
__global__ void DropoutBackward(const int n, const int K, const Dtype* bottom_data, Dtype* p_data, Dtype* bottom_diff, 
		const Dtype* top_diff, const Dtype* epsilon_data, const Dtype* theta_data) {
	CUDA_KERNEL_LOOP(index, n) {
		const int pos = index % K;
		if ((bottom_data[index] - epsilon_data[index]) < theta_data[index] && (bottom_data[index] + epsilon_data[index]) > theta_data[index])
			bottom_diff[index] = top_diff[index] / ( 1 - p_data[pos]);
		else
			bottom_diff[index] = 0;
	}
}

template <typename Dtype>
__global__ void Updata_p(const int n, const int K, const Dtype* bottom_data, Dtype* p_data, Dtype* lamda_data, 
		const Dtype* epsilon_data, const Dtype* theta_data, const Dtype* top_diff, const Dtype p_lr) {
		CUDA_KERNEL_LOOP(index, n) {
		const int pos = index % K;
			if ((bottom_data[index] - epsilon_data[index]) < theta_data[index] && (bottom_data[index] + epsilon_data[index]) > theta_data[index])
				p_data[pos] += (1 / (lamda_data[pos] * p_data[pos])) * top_diff[index] * p_lr;
	}
}

template <typename Dtype>
__global__ void Updata_lamda(const int n, const int K, const Dtype* bottom_data, Dtype* p_data, Dtype* lamda_data, 
		const Dtype* epsilon_data, const Dtype* theta_data, const Dtype* top_diff, const Dtype lamda_lr) {
		CUDA_KERNEL_LOOP(index, n) {
		const int pos = index % K;
			if ((bottom_data[index] - epsilon_data[index]) < theta_data[index] && (bottom_data[index] + epsilon_data[index]) > theta_data[index])
				lamda_data[pos] -= ((log(p_data[pos]/epsilon_data[index]/2) - log(lamda_data[pos]) + 1)
					/ lamda_data[pos] / lamda_data[pos] + epsilon_data[index] * epsilon_data[index] / 6) * top_diff[index] * lamda_lr;
	}
}

template <typename Dtype>
void BDropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const int count = bottom[0]->count();
	Dtype* theta_data = theta_.mutable_gpu_data();
	const Dtype* p_data = this->blobs_[0]->gpu_data();
	const Dtype* lamda_data = this->blobs_[1]->gpu_data();
	Dtype* epsilon_data = epsilon_.mutable_gpu_data();

	const int channels = bottom[0]->channels();
	const int height = bottom[0]->height();
	const int width = bottom[0]->width();
	const int K = channels * height * width;
	
	if (bottom[0] == top[0]) {
		caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
	}

	if (this->phase_ == TRAIN) {
		// Create theta & epsilon
		caffe_gpu_rng_uniform(count, static_cast<Dtype>(0), static_cast<Dtype>(1), theta_data);
		DropoutForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			count, K, bottom_data, p_data, lamda_data, theta_data, epsilon_data, top_data);
	} else {
		caffe_copy(bottom[0]->count(), bottom_data, top_data);
	}
}

template <typename Dtype>
void BDropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

	const Dtype* bottom_data = bottom[0]->gpu_data();
	const int count = bottom[0]->count();
	Dtype* p_data = this->blobs_[0]->mutable_gpu_data();
	Dtype* lamda_data = this->blobs_[1]->mutable_gpu_data();
	const Dtype* epsilon_data = epsilon_.gpu_data();
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const Dtype* theta_data = theta_.gpu_data();
	if (top[0] == bottom[0]) { bottom_data = bottom_memory_.gpu_data();}

	const int channels = bottom[0]->channels();
	const int height = bottom[0]->height();
	const int width = bottom[0]->width();
	const int K = channels * height * width;

	iteration_++;

	if (!(iteration_ % updata_stride_)) {
		if (!fixed_p_) {
			Updata_p<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
					count, K, bottom_data, p_data, lamda_data, epsilon_data, theta_data, top_diff, p_lr_);
		}
		if (!fixed_lamda_) {
			Updata_lamda<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
					count, K, bottom_data, p_data, lamda_data, epsilon_data, theta_data, top_diff, lamda_lr_);
		}
	}

	if (propagate_down[0]) {
		if (this->phase_ == TRAIN) {
			DropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(count),	CAFFE_CUDA_NUM_THREADS>>>(
					count, K, bottom_data, p_data, bottom_diff, top_diff, epsilon_data, theta_data);
		}
		else {
			caffe_copy(top[0]->count(), top_diff, bottom_diff);
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(BDropoutLayer);


}  // namespace caffe
