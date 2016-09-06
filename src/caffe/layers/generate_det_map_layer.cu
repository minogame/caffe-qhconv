#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include <cstdio>

namespace caffe {

template <typename Dtype>
__global__ void generaterKernel(Dtype* top_data, const Dtype* anno_data, const int objects, 
																const int width, const Dtype hs, const Dtype ws, 
																const int hskip, const int wskip, const Dtype hgs, const Dtype wgs, const Dtype mirror) {
	const int h = threadIdx.x;
	const int w = threadIdx.y;
	const int c = blockIdx.x;

	const int index = (mirror)? (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.x * width) + width - threadIdx.y -1: 
															(blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.x * width) + threadIdx.y;
	top_data[index] = 0;
	for (int i = 0; i < objects; ++i) {
		if (anno_data[i*5] != 0) {
			const Dtype rh = (h * hs + hskip)/hgs;
			const Dtype rw = (w * ws + wskip)/wgs;

			if (c == (anno_data[i*5]-1) && rh>=anno_data[i*5+3] && rw>=anno_data[i*5+1] && rh<anno_data[i*5+4] && rw<anno_data[i*5+2] ) {
				top_data[index] = 1;
			}
		}
	}
}

template <typename Dtype>
__global__ void backgroundKernel(Dtype* top_data, const int c) {
	const int pos = (threadIdx.x * blockDim.y) + threadIdx.y;
	const int size = blockDim.x * blockDim.y;
	const int index = ((c -1) * size) + pos;

	Dtype tmp;
	for (int i = 0; i < c-1 ; ++i)
		tmp += top_data[i*size+pos];

	top_data[index] = (tmp>0.1)? 0:1;
}

template <typename Dtype>
__global__ void maskKernel(Dtype* top_data, const Dtype* anno_data, const int objects, 
																const int width, const int channels, const Dtype hs, const Dtype ws, 
																const int hskip, const int wskip, const Dtype hgs, const Dtype wgs, const Dtype mirror) {
	const int h = threadIdx.x;
	const int w = threadIdx.y;
	const int block_size = channels * blockDim.x * blockDim.y;

	for (int i = 0; i < objects; ++i) {
		const int index = (mirror)? (block_size * i) + (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.x * width) + width - threadIdx.y -1: 
																(block_size * i) + (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.x * width) + threadIdx.y;

		if (anno_data[i*5] != 0) {
			const Dtype rh = (h * hs + hskip)/hgs;
			const Dtype rw = (w * ws + wskip)/wgs;

			if (rh>=anno_data[i*5+3] && rw>=anno_data[i*5+1] && rh<anno_data[i*5+4] && rw<anno_data[i*5+2] ) {
				top_data[index] = 1;
				// for (int j = 0; j <objects; ++j)
				// 	if (blockIdx.x == anno_data[i*5] - 1 && i != j) {
				// 		top_data[index] = 0; break;
				// 	}
			}	else
				top_data[index] = 0;
		}
	}
}

template <typename Dtype>
__global__ void backgroundmaskKernel(Dtype* top_data, const Dtype* background_data, const int c) {
	const int pos = (threadIdx.x * blockDim.y) + threadIdx.y;
	const int size = blockDim.x * blockDim.y;

	for (int i = 0; i < c ; ++i) {
		const int index = (i * size) + pos;
		top_data[index] = background_data[pos];
	}
}

template <typename Dtype>
void GenerateDetMapLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	Dtype* top_data = top[0]->mutable_gpu_data();
	const Dtype* anno_data = bottom[2]->gpu_data();
	const int objects = bottom[2]->channels();
	height_scale_ = static_cast<Dtype>(in_height_) / static_cast<Dtype>(out_height_);
	width_scale_ = static_cast<Dtype>(in_width_) / static_cast<Dtype>(out_width_);

	dim3 theradsPerBlock(out_height_,out_width_);
	if (cropped_) {
		const Dtype* crop_data = bottom[3]->cpu_data();
		generaterKernel<<<out_channels_, theradsPerBlock>>>
										(top_data, anno_data, objects, out_width_, height_scale_, width_scale_, 
										crop_data[0], crop_data[2], crop_data[4], crop_data[5], crop_data[6]);
	} else {
		generaterKernel<<<out_channels_, theradsPerBlock>>>
										(top_data, anno_data, objects, out_width_, height_scale_, width_scale_, 0, 0, (Dtype)1, (Dtype)1, (Dtype)0);
	}

	if (this->layer_param_.generate_det_map_param().background_class())
		backgroundKernel<<<1, theradsPerBlock>>>(top_data, out_channels_);

	if (top.size() == 3) {
		Dtype* mask_data = top[1]->mutable_gpu_data();
		Dtype* label_data = top[2]->mutable_cpu_data();
		Dtype* anno_cpu_data = new Dtype[bottom[2]->count()];
		cudaMemcpy(anno_cpu_data, anno_data, bottom[2]->count()*sizeof(Dtype), cudaMemcpyDeviceToHost);
		for (std::pair<int,int> iter(0, 0); iter.first < bottom[2]->count(); iter.first += 5, iter.second++)
			label_data[iter.second] = anno_cpu_data[iter.first] - 1;
		delete[] anno_cpu_data;
		if (this->layer_param_.generate_det_map_param().background_class()) label_data[bottom[2]->channels()] = out_channels_ - 1;

		if (cropped_) {
			const Dtype* crop_data = bottom[3]->cpu_data();
			maskKernel<<<out_channels_, theradsPerBlock>>>
											(mask_data, anno_data, objects, out_width_, out_channels_, height_scale_, width_scale_, 
											crop_data[0], crop_data[2], crop_data[4], crop_data[5], crop_data[6]);
		} else maskKernel<<<out_channels_, theradsPerBlock>>>
										(mask_data, anno_data, objects, out_width_, out_channels_, height_scale_, width_scale_, 0, 0, (Dtype)1, (Dtype)1, (Dtype)0);

		if (this->layer_param_.generate_det_map_param().background_class())
			backgroundmaskKernel<<<1, theradsPerBlock>>>
										(mask_data+objects*out_channels_*out_width_*out_height_, top_data+(out_channels_-1)*out_width_*out_height_ ,out_channels_);
	}
}

template <typename Dtype>
void GenerateDetMapLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_LAYER_GPU_FUNCS(GenerateDetMapLayer);


}  // namespace caffe
