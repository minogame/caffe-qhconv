#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/loss_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DetSoftmaxLossForwardKernel(const int nthreads, const Dtype* prob_data, 
					const Dtype* label, Dtype* loss, const int dim, const int spatial_dim) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int n = index / spatial_dim;
		const int s = index % spatial_dim;
		const int label_value = static_cast<int>(label[n * spatial_dim + s]);
		loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
										Dtype(FLT_MIN)));
	}
}

template <typename Dtype>
__global__ void GeffBufferKernel(const Dtype* bottom_data, const Dtype* mask, Dtype* buffer,
																	const int num, const int size) {
	extern __shared__ float sdata[];
	const int pos = (threadIdx.x * blockDim.y) + threadIdx.y;
	const int channel = blockIdx.x;
	const int idx = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.x * blockDim.y) + threadIdx.y;

	for (int i = 0; i < num; ++i)	{
		//read data and mutliply by mask
		sdata[pos] = static_cast<float>(bottom_data[idx] * mask[size*i+idx]);
		__syncthreads();

		//reduction
		if (threadIdx.y == 0)
			for (int j = 1; j < blockDim.y; ++j)
				sdata[pos] += sdata[pos+j];
		__syncthreads();

		if ( threadIdx.x == 0 && threadIdx.y == 0) {
			for (int k = blockDim.y; k < blockDim.x * blockDim.y; k+=blockDim.y)
				sdata[0] += sdata[k];

			//write to buffer
			const int idbu = gridDim.x * i + channel;
			buffer[idbu] = sdata[0];
		}
	}
}

template <typename Dtype>
void DetSoftmaxLossLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	const int channels = bottom[0]->channels();
	const int width = bottom[0]->width();
	const int height = bottom[0]->height();
	const int size = height * width;
	const int num = bottom[1]->num();

	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* buffer_data = buffer_.mutable_gpu_data();
	const Dtype* mask_data = bottom[1]->gpu_data();

	dim3 theradsPerBlock(height,width);
	GeffBufferKernel<<<channels,theradsPerBlock,size*sizeof(float)>>>(bottom_data,mask_data,buffer_data,num,size);

	softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

	const Dtype* prob_data = prob_.gpu_data();
	const Dtype* label = bottom[2]->gpu_data();
	Dtype* loss_buffer_data = buffer_.mutable_gpu_diff();
	const int nthreads = outer_num_ * inner_num_;
	const int dim = prob_.count() / outer_num_;
  DetSoftmaxLossForwardKernel<Dtype><<<CAFFE_GET_BLOCKS(nthreads),CAFFE_CUDA_NUM_THREADS>>>
  			(nthreads, prob_data, label, loss_buffer_data, dim, inner_num_);

  Dtype loss;
  caffe_gpu_asum(nthreads, loss_buffer_data, &loss);
  loss /= outer_num_;
  top[0]->mutable_cpu_data()[0] = loss;

  if (top.size() == 2) {
  	top[1]->Reshape(bottom[1]->num(),bottom[0]->channels(),1,1);
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void DetSoftmaxLossBackwardKernel(Dtype* bottom_diff, const Dtype* mask_data,
					const Dtype* prob_data, const Dtype* label, const int objects, const int dim) {
	const int block_size = gridDim.x * blockDim.x * blockDim.y;
	const int idx = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.x * blockDim.y) + threadIdx.y;
	for (int i = 0; i < objects; ++i) {
		const int idx_msk = idx + block_size * i;
		bottom_diff[idx] += prob_data[blockIdx.x + dim*i] * mask_data[idx_msk] / static_cast<Dtype>(objects);
		if (blockIdx.x == label[i]) bottom_diff[idx] -= 0.02 * mask_data[idx_msk];
		// if (idx == 0) printf("bottom_diff[%d] += prob_data[%d] * mask_data[%d] / objects\n",idx, blockIdx.x + dim*i, idx_msk);
		// if (idx == 0) printf("%f += %f * %f / %d\n",bottom_diff[idx], prob_data[blockIdx.x + dim*i], mask_data[idx_msk], objects);
	}
	if (bottom_diff[idx] < -0.02) bottom_diff[idx] = (Dtype)-0.0199999;
	// if (blockIdx.x == 0 && threadIdx.y<6 && threadIdx.x<6) {bottom_diff[idx]/=2; printf("%d === %f\n", idx, bottom_diff[idx]);}
}

// template <typename Dtype>
// __global__ void GeffBufferCountKernel(const Dtype* bottom_data, const Dtype* mask, Dtype* buffer,
// 																	const int num, const int size) {
// 	extern __shared__ float sdata[];
// 	const int pos = (threadIdx.x * blockDim.y) + threadIdx.y;
// 	const int channel = blockIdx.x;
// 	const int idx = (blockIdx.x * blockDim.x * blockDim.y) + (threadIdx.x * blockDim.y) + threadIdx.y;

// 	for (int i = 0; i < num; ++i)	{
// 		//read data and mutliply by mask
// 		sdata[pos] = static_cast<float>(bottom_data[idx] * mask[size*i+idx]) == 0 ? 0 : 1;
// 		__syncthreads();

// 		//reduction
// 		if (threadIdx.y == 0)
// 			for (int j = 1; j < blockDim.y; ++j)
// 				sdata[pos] += sdata[pos+j];
// 		__syncthreads();

// 		if ( threadIdx.x == 0 && threadIdx.y == 0) {
// 			for (int k = blockDim.y; k < blockDim.x * blockDim.y; k+=blockDim.y)
// 				sdata[0] += sdata[k];

// 			//write to buffer
// 			const int idbu = gridDim.x * i + channel;
// 			buffer[idbu] = sdata[0];
// 		}
// 	}
// }

template <typename Dtype>
void DetSoftmaxLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
	if (propagate_down[1] or propagate_down[2]) {
		LOG(FATAL) << this->type()
							 << " Layer cannot backpropagate to label or mask inputs.";
	}
	if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* label = bottom[2]->gpu_data();
		const Dtype* mask_data = bottom[1]->gpu_data();
		const int num = bottom[1]->num();
		const int channels = bottom[0]->channels();
		const int width = bottom[0]->width();
		const int height = bottom[0]->height();
		const int dim = prob_.count() / outer_num_;

    dim3 theradsPerBlock(height,width);

		// GeffBufferCountKernel<<<channels,theradsPerBlock,size*sizeof(float)>>>(bottom_data,mask_data,buffer_data,num,size);
    DetSoftmaxLossBackwardKernel<<<channels,theradsPerBlock>>>(bottom_diff, mask_data, prob_data, label, num, dim);

		const Dtype loss_weight = top[0]->cpu_diff()[0];
		caffe_gpu_scal(bottom[0]->count(), loss_weight, bottom_diff);
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(DetSoftmaxLossLayer);

}  // namespace caffe
