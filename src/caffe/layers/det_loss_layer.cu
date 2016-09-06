#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DetLossKernel(const Dtype* bottom_data, const Dtype* target_data,
														Dtype* bottom_diff, const int width) {
	const int index = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.x * width) + threadIdx.y;
	bottom_diff[index] = ((1 - target_data[index]) * tanhf(bottom_data[index]) * 0.1 +
												target_data[index] * (tanhf(bottom_data[index]) - 1)) / (blockDim.x * blockDim.y) * 10;
}

template <typename Dtype>
void DetLossLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const Dtype* target_data = bottom[1]->gpu_data();
	const int channels = bottom[1]->channels();
	const int width = bottom[1]->width();
	const int height = bottom[1]->height();
	// const int count = bottom[1]->count();

	dim3 theradsPerBlock(height,width);
	DetLossKernel<<<channels, theradsPerBlock>>>
									(bottom_data, target_data, bottom_diff, width);

	// Dtype* loss = top[0]->mutable_gpu_data();
	Dtype loss;
	// cudaMalloc((void**) &loss, sizeof(Dtype));
	caffe_gpu_asum(channels*width*height, bottom_diff, &loss);
	top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void DetLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[1]) {
		LOG(FATAL) << this->type()
							 << " Layer cannot backpropagate to label inputs.";
	}
	if (propagate_down[0]) {
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		const Dtype loss_weight = top[0]->cpu_diff()[0];
		caffe_gpu_scal(bottom[0]->count(), loss_weight, bottom_diff);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(DetLossLayer);

}  // namespace caffe
