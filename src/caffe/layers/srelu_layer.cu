#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

// CUDA kernele for forward
template <typename Dtype>
__global__ void SReLUForward(const int n, const Dtype* in, Dtype* out, const Dtype* th_data) {
	CUDA_KERNEL_LOOP(index, n) {
		out[index] = in[index] > th_data[index] ? in[index] : 0;
		if (out[index] < 0) out[index] = 0;
	}
}

template <typename Dtype>
__global__ void SReLUParamBackwardAve(const int count, const Dtype* bottom_data, const Dtype* top_diff, Dtype* th_data, Dtype threshold_lr) {
	CUDA_KERNEL_LOOP(index, count) {
		Dtype th_diff;
		th_diff = (bottom_data[index] <= th_data[index]) ? 0 : (1 - expf(bottom_data[index])/(expf(bottom_data[index]) + expf(th_data[index]))) * top_diff[index];
		th_data[index] = (th_data[index] - threshold_lr * th_diff);
		//if (!(index%100000)) printf("th_diff[%d] = %f  bd:%f   thd:%f   td:%f\n",index, th_diff, bottom_data[index], th_data[index], top_diff[index] );
	}
}

template <typename Dtype>
__global__ void SReLUParamBackwardVar(const int count, Dtype* th_data, const Dtype* hyperparam) {
	Dtype ave = hyperparam[0];
	CUDA_KERNEL_LOOP(index, count) {
		th_data[index] = (ave - th_data[index]) * (ave - th_data[index]);
	}
}

template <typename Dtype>
__global__ void SReLUParamForwardVar(const int count, Dtype* th_data, const Dtype* bottom_data) {
	CUDA_KERNEL_LOOP(index, count) {
		th_data[index] = bottom_data[index] * bottom_data[index];
	}
}

template <typename Dtype>
__global__ void SReLUVarSqrt(int count, Dtype* hyperparam, float variance_recover) {
	hyperparam[1] = sqrt(hyperparam[1] / count) * variance_recover;
}

// CUDA kernel for bottom backward
template <typename Dtype>
__global__ void SReLUBackward(const int n, const Dtype* top_diff, const Dtype* bottom_data, Dtype* bottom_diff, const Dtype* th_data) {
	CUDA_KERNEL_LOOP(index, n) {
		bottom_diff[index] = top_diff[index] * (bottom_data[index] > th_data[index]);
	}
}

template <typename Dtype>
__global__ void show(const Dtype* data_from, int total, int stride) {
	CUDA_KERNEL_LOOP(index, total)  {
		if (!(index%stride)) printf("th_data[%d] = %f\n",index, data_from[index] );
	}
}

template <typename Dtype>
void SReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const int count = bottom[0]->count();
	Dtype* th_data = threshold_buffer_.mutable_gpu_data();
	Dtype* hyperparam = this->blobs_[0]->mutable_gpu_data();
	Dtype hpcpu[2];

	cudaMemcpy(hpcpu, hyperparam, sizeof(Dtype) * 2, cudaMemcpyDeviceToHost);

	// For in-place computation
	if (top[0] == bottom[0]) {
		caffe_copy(count, bottom_data, bottom_memory_.mutable_gpu_data());
	}

	if (!(this->iter%this->showvalue)) {
		SReLUParamForwardVar<Dtype><<<CAFFE_GET_BLOCKS(count),
		 					CAFFE_CUDA_NUM_THREADS>>>(count, th_data, bottom_data);
		caffe_gpu_asum<Dtype>(count, th_data, hpcpu + 1);

		hpcpu[1] = sqrt (hpcpu[1]/count) * variance_recover;
		LOG(INFO)<<"MEAN: "<<hpcpu[0]<<"    VAR:"<<hpcpu[1];
	}
	this->iter++;

	// Create threshold buffer vector
	caffe_gpu_rng_gaussian(count, hpcpu[0], hpcpu[1], th_data);

	// NOLINT_NEXT_LINE(whitespace/operators)
	SReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
			count, bottom_data, top_data, th_data);
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void SReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
	const Dtype* bottom_data = bottom[0]->gpu_data();
	const Dtype* top_diff = top[0]->gpu_diff();
	const int count = bottom[0]->count();
	Dtype* th_data = threshold_buffer_.mutable_gpu_data();
	Dtype* hyperparam = this->blobs_[0]->mutable_gpu_data();

	// For in-place computation
	if (top[0] == bottom[0]) {
		bottom_data = bottom_memory_.gpu_data();
	}

	// // Propagte to param
	// if (!stop_updata){
	// 	if (!(this->iter%this->showvalue)) {
	// 		SReLUParamBackwardAve<Dtype><<<CAFFE_GET_BLOCKS(count),
	// 					CAFFE_CUDA_NUM_THREADS>>>(count, bottom_data, top_diff, th_data, threshold_lr);

	// 		//show<Dtype><<<1,1>>>(th_data,100,1);

	// 		Dtype* ave = new Dtype;
	// 		caffe_gpu_asum<Dtype>(count, th_data, ave);
	// 		*ave /= count;
	// 		cudaMemcpy(hyperparam, ave, sizeof(Dtype), cudaMemcpyHostToDevice);

	// 		SReLUParamBackwardVar<Dtype><<<CAFFE_GET_BLOCKS(count),
	// 					CAFFE_CUDA_NUM_THREADS>>>(count, th_data, hyperparam);

	// 		Dtype* var = new Dtype;
	// 		caffe_gpu_asum<Dtype>(count, th_data, var);
	// 		cudaMemcpy(hyperparam + 1, var, sizeof(Dtype), cudaMemcpyHostToDevice);

	// 		SReLUVarSqrt<Dtype><<<1,1>>>(count, hyperparam, variance_recover);

			
	// 		LOG(INFO)<<"MEAN: "<<*ave<<"    VAR:"<<*var/count;
	// 	}
	// 	this->iter++;
	// }

	// Propagate to bottom
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	SReLUBackward<<<CAFFE_GET_BLOCKS(count),
		CAFFE_CUDA_NUM_THREADS>>>(count, top_diff, bottom_data, bottom_diff, th_data);
}


INSTANTIATE_LAYER_GPU_FUNCS(SReLULayer);


}  // namespace caffe
