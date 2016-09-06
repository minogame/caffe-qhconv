#include <algorithm>
#include <vector>
#include <iostream>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SelfHintLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	sh_coe_ = iteration_ >= max_step_ ? limit_ : (exp(iteration_ * smoothness_ / max_step_) - 1) / (exp(smoothness_) - 1) * limit_ ;
	top[0]->mutable_cpu_data()[0] = sh_coe_;
	iteration_ += 1;
}

template <typename Dtype>
__global__ void CreateBuffer(const int n, const Dtype* label_data, const Dtype* bottom_data, Dtype* buffer,
		const int channels, const int height, const int width) {
	CUDA_KERNEL_LOOP(index, n) {
		const int size = height * width;
		const int c = index / size;
		const int cr = index % size;
		const int label = static_cast<int>(label_data[0]);
		const Dtype t = bottom_data[index] - bottom_data[label*size + cr];

		if (c == label) buffer[index] = 0;
		else buffer[index] = (t >= 0) ? t : -t;
	}
}

template <typename Dtype>
__global__ void SelfHintBackward(const int n, const Dtype* prob, const Dtype* label_data, const Dtype* bottom_data, Dtype* bottom_diff,
		const int channels, const int height, const int width, const Dtype sh_coe, const Dtype* buffer, const int num) {
	CUDA_KERNEL_LOOP(index, n) {
		const int size = height * width;
		const int c = index / size;
		const int label = static_cast<int>(label_data[0]);

		/***************** VERSION 0 *****************/
		// const int cr = index % size;
		// Dtype sum = 0;

		// if (c == label)
		// 	bottom_diff[index] = (prob[c] - 1) / size / num *3;
		// else
		// 	bottom_diff[index] = prob[c] / size / num * 3;

		/***************** VERSION 1 *****************/
		// Dtype sum = 0;
		// for (int i = 0; i < size; ++i)
		// 	sum += bottom_data[c*size+i];

		// if (c == label) {
		// 	if (sum == 0)
		// 		bottom_diff[index] = (prob[c] - 1) / size / num;
		// 	else 
		// 		bottom_diff[index] = ((1 - sh_coe) * (prob[c] - 1) / size + sh_coe * (prob[c] - 1) * bottom_data[index] / sum) / num;
		// }
		// else {
		// 	if (sum == 0)
		// 		bottom_diff[index] = prob[c] / size / num;
		// 	else
		// 		bottom_diff[index] = ((1 - sh_coe) * prob[c] / size + sh_coe * prob[c] * bottom_data[index] / sum) / num;
		// }

		/***************** VERSION 2 *****************/
		// const int cr = index % size;
		// Dtype sum = 0;
		// for (int i = 0; i < size; ++i)
		// 	sum += bottom_data[label * size + i];

		// if (c == label) {
		// 	if (sum == 0)
		// 		bottom_diff[index] = (prob[c] - 1) / size / num;
		// 	else 
		// 		bottom_diff[index] = ((1 - sh_coe) * (prob[c] - 1) / size + sh_coe * (prob[c] - 1) * bottom_data[index] / sum) / num;
		// }
		// else {
		// 	if (sum == 0)
		// 		bottom_diff[index] = prob[c] / size / num;
		// 	else
		// 		bottom_diff[index] = ((1 - sh_coe) * prob[c] / size + sh_coe * prob[c] * bottom_data[label * size + cr] / sum) / num;
		// }


		/***************** VERSION 3 *****************/
		const int cr = index % size;
		Dtype sum = 0;

		if (c == label) {
			for (int i = 0; i < size; ++i)
				sum += bottom_data[label * size + i];
			if (sum == 0)
				bottom_diff[index] = (prob[c] - 1) / size / num;
			else 
				bottom_diff[index] = ((1 - sh_coe) * (prob[c] - 1) / size + sh_coe * (prob[c] - 1) * bottom_data[index] / sum) / num;
		}
		else {
			for (int i = 0; i < size; ++i)
				sum += bottom_data[label * size + i] * bottom_data[c*size+i];
			if (sum == 0)
				bottom_diff[index] = prob[c] / size / num;
			else
				bottom_diff[index] = ((1 - sh_coe) * prob[c] / size + sh_coe * prob[c] * bottom_data[label * size + cr] * bottom_data[index] / sum) / num;
		}

		/***************** VERSION 4 *****************/ //TEST FOR ONLY GIVE LABEL MAP
		// if (c == label) {
		// 	Dtype sum = 0;
		// 	for (int i = 0; i < size; ++i)
		// 		sum += bottom_data[label * size + i];
		// 	if (sum == 0)
		// 		bottom_diff[index] = (prob[c] - 1) / size / num;
		// 	else 
		// 		bottom_diff[index] = ((1 - sh_coe) * (prob[c] - 1) / size + sh_coe * (prob[c] - 1) * bottom_data[index] / sum) / num;
		// }
		// else {
		// 		bottom_diff[index] = prob[c] / size / num;
		// }

		/***************** VERSION 5 *****************/
		// const int cr = index % size;
		// Dtype sum = 0;

		// if (c == label) {
		// 	for (int i = 0; i < size; ++i)
		// 		sum += bottom_data[label * size + i];
		// 	if (sum == 0)
		// 		bottom_diff[index] = (prob[c] - 1) / size / num;
		// 	else 
		// 		bottom_diff[index] = ((1 - sh_coe) * (prob[c] - 1) / size + sh_coe * (prob[c] - 1) * bottom_data[index] / sum) / num;
		// }
		// else {
		// 	for (int i = 0; i < size; ++i)
		// 		sum += bottom_data[label * size + i] * bottom_data[c*size+i];
		// 	Dtype sumB = 0;
		// 	for (int i = 0; i < size; ++i)
		// 		sumB += bottom_data[c*size+i];
		// 	if (sum == 0)
		// 		bottom_diff[index] = prob[c] / size / num;
		// 	else
		// 		bottom_diff[index] = ((1 - 2 * sh_coe) * prob[c] / size
		// 											 + sh_coe * prob[c] * bottom_data[label * size + cr] * bottom_data[index] / sum
		// 											 + sh_coe * prob[c] * bottom_data[index] / sumB) / num;
		// }
	}
}

template <typename Dtype>
void SelfHintLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	const int count = height_ * width_ * channels_;
	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* prob_data = bottom[1]->gpu_data();
	const Dtype* label = bottom[2]->gpu_data();
	const int num = bottom[0]->num();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	Dtype* buffer;
	cudaMalloc((void**) &buffer, count*sizeof(Dtype));

	for (int n = 0; n < num; ++n) {
		// CreateBuffer<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, label + n, bottom_data + n * count, buffer, channels_, height_, width_);
		SelfHintBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, prob_data + n * channels_, label + n, 
																								bottom_data + n * count, bottom_diff + n * count, channels_, height_, width_, sh_coe_, buffer, num);
	}
	cudaFree(buffer);
}


INSTANTIATE_LAYER_GPU_FUNCS(SelfHintLayer);


}  // namespace caffe
