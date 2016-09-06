#include <vector>
// #include <iostream>
// #include <cstdio>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void create_buffer_kernel(const Dtype* data_from, Dtype* data_to,
										int K, int N, int L, int W, int total, Dtype RNK, Dtype RWK, Dtype RK, Dtype RL) {
	CUDA_KERNEL_LOOP(index, total)  {
		int _m = floorf(index * RNK);
		int _mr = index - _m * N * K;
		int _h = floorf(_mr * RWK);
		int _hr = _mr - _h * W * K;
		int _w = floorf(_hr * RK);
		int _k = _hr - _w * K;
		int _hk = _h % L;
		int _wk = _w % L;
		// int _m = index / (N * K);
		// int _mr = index % (N * K);
		// int _h = _mr / (W * K);
		// int _hr = _mr % (W * K);
		// int _w = _hr / K;
		// int _k = _hr % K;
		// int _hk = _h % L;
		// int _wk = _w % L;
		data_to[index] = data_from[_m * L * L * K + _hk * L * K + _wk * K + _k];
	}
}

// template <typename Dtype>
// __global__ void reduce(const Dtype* col, const Dtype* weight, int _bk, int N, int _n, Dtype* g_odata) {
// 	extern __shared__ Dtype sdata[];
// 	// each thread loads one element from global to shared mem
// 	unsigned int tid = threadIdx.x;
// 	unsigned int _k = blockIdx.x*blockDim.x + threadIdx.x;
// 	sdata[tid] = weight[_bk + _k] * col[_k * N + _n];
// 	__syncthreads();
// 	// do reduction in shared mem
// 	for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
// 		if (tid < s) {
// 			sdata[tid] += sdata[tid + s];
// 		}
// 		__syncthreads();
// 	}
// 	// write result for this block to global mem
// 	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
// }

// template <typename Dtype>
// __global__ void forward_wrt_top(const Dtype* col, const Dtype* weight, Dtype* top,
// 										int K, int N, int L, int W, int H, Dtype RN, int total) {
// 	CUDA_KERNEL_LOOP(index, total) {
// 		int _m = floorf(index * RN);
// 		int _n = index - _m * N;
// 		//int _bk = _m * N * K + _n * K;
// 		int _bk = index * K;

// 		for (int _k = 0; _k < K; _k++) {
// 			top[index] += (weight[_bk + _k] * col[_k * N + _n]);
// 		}
// 	}
// }

template <typename Dtype>
__global__ void forward_wrt_top_inter(const Dtype* col, const Dtype* weight, Dtype* top,
										int K, int N, Dtype RNK, Dtype RK, int total) {
	CUDA_KERNEL_LOOP(index, total) {
		// int _m = floorf(index * RNK);
		// int _mr = index - _m * N * K;
		// int _n = floorf(_mr * RN);
		// int _k = _mr - _n * K;

		int _mr = index - floorf(index * RNK) * N * K;
		int _n = floorf(_mr * RK);
		int _k = _mr - _n * K;

		top[index] = (weight[index] * col[_k * N + _n]);
	}
}

// template <typename Dtype>
// __global__ void forward_wrt_top_reduce(const Dtype* col, const Dtype* weight, Dtype* top, int K, int N, int I) {
// 	__shared__ Dtype sdata[CAFFE_CUDA_NUM_THREADS];
// 	int _m = blockIdx.x;
// 	int _n = blockIdx.y;
// 	int _k = threadIdx.x + I * CAFFE_CUDA_NUM_THREADS;
// 	int idx = threadIdx.x;
// 	int index = _m * N + _n;
// 	sdata[idx] = weight[index * K + _k] * col[_k * N + _n];
// 	__syncthreads();
// 	for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
// 		if (idx < s) {
// 			sdata[idx] += sdata[idx + s];
// 		}
// 		__syncthreads();
// 	}
// 	if (idx == 0) top[index] += sdata[0];
// }

template <typename Dtype>
__global__ void gradient_wrt_weight(const Dtype* top_diff, const Dtype* col_buffer, Dtype* weight_diff,
										int S, int K, int N, int L, int W, int H, Dtype RSK, Dtype RK, Dtype RL, int total) {
	CUDA_KERNEL_LOOP(index, total)  {

		int _m = floorf(index * RSK);
		int _mr = index - _m * S * K;
		int _s = floorf(_mr * RK);
		int _k = _mr - _s * K;

		int _hk = floorf(_s * RL);
		int _wk = _s - _hk * L;

		// int _m = index / (L * L * K);
		// int _mr = index % (L * L * K);
		// int _s = _mr / K;
		// int _k = _mr % K;

		// int _hk = _s / L;
		// int _wk = _s % L;

		for (int _h = _hk; _h < H; _h += L) {
			for (int _w = _wk; _w < W; _w += L) {
				int _n = _h * W + _w;
				weight_diff[index] += (top_diff[_m * N + _n] * col_buffer[_k * N + _n]);
			}
		}
	}
}

template <typename Dtype>
__global__ void gradient_wrt_bottom(const Dtype* top_diff, const Dtype* weight, Dtype* x_diff,
										int M, int K, int N, int L, Dtype RK, int total) {
	CUDA_KERNEL_LOOP(index, total) {
		int _n = floorf(index * RK);
		//int _k = index - _n * K;
		for (int _mn = 0; _mn < M * N; _mn += N) {
			x_diff[index] += (weight[_mn * K + index] * top_diff[_mn + _n]);  //_n * K + _k
		}
	}
}

// template <typename Dtype>
// __global__ void transpose(const Dtype* data_from, Dtype* data_to,
// 										int M, int K, int N, int L, int W, int H) {
// 	int total = N * K;
// 	CUDA_KERNEL_LOOP(index, total)  {
// 		int _n = index / K;
// 		int _k = index % K;
// 		data_to[index] = data_from[_k*N + _n];
// 	}
// }

// template <typename Dtype>
// __global__ void show(const Dtype* data_from, int total, int stride) {
// 	CUDA_KERNEL_LOOP(index, total)  {
// 		if (!(index%stride)) printf("top[%d] = %f\n",index, data_from[index] );
// 	}
// }

template <typename Dtype>
void SharedWeightsLocalLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {

	Dtype* x_data = col_buffer_.mutable_gpu_data();
	const Dtype* weight = this->blobs_[0]->gpu_data();
	//const Dtype* bias = this->blobs_[1]->gpu_data();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();

	Dtype* w_data = weight_buffer_.mutable_gpu_data();
	Dtype* b_data = bias_buffer_.mutable_gpu_data();

	//CUDA_CHECK(cudaMemset(top_data, 0, sizeof(Dtype) * top[0]->count()));

	create_buffer_kernel<Dtype><<<CAFFE_GET_BLOCKS(N_*M_*K_),CAFFE_CUDA_NUM_THREADS>>>
											(weight, w_data, K_, N_, num_kernel_, width_out_, N_*M_*K_, RNK_, RWK_, RK_, RL_);
	//CUDA_POST_KERNEL_CHECK;

	// show<Dtype><<<1,1>>>(w_data);
	// CUDA_POST_KERNEL_CHECK;

	// if (bias_term_){
	//  for (int n = 0; n < num_output_; n++){
	//    for (int r = 0; r < N_; r++){
	//      int _h_ = r / width_out_;
	//      int _w_ = r % width_out_;
	//      int _hk_ = _h_ % num_kernel_;
	//      int _wk_ = _w_ % num_kernel_;
	//      caffe_copy(1,bias + this->blobs_[1]->offset(n,0,_hk_*num_kernel_+_wk_),
	//                  b_data + bias_buffer_.offset(0,0,n,_h_*width_out_+_w_));
	//    }
	//  }
	// }

	Blob<Dtype> E;
	E.Reshape(1, 1, 1, K_);
	FillerParameter filler_param;
	filler_param.set_value(1);
	ConstantFiller<Dtype> filler(filler_param);
	filler.Fill(&E);

	Blob<Dtype> intermediate;
	intermediate.Reshape(M_, 1, N_, K_);
	Dtype* inter_data = intermediate.mutable_gpu_data();

	// Blob<Dtype> slow;
	// slow.Reshape(1, 1, N_, K_);
	// Dtype* slow_data = slow.mutable_gpu_data();
	
	for (int n=0; n<num_; n++) {
		im2col_gpu(bottom_data + bottom[0]->offset(n), channels_, height_,
							 width_, kernel_size_, kernel_size_, pad_, pad_, stride_, stride_, x_data);

		//cudaMemcpy(inter_data, w_data, sizeof(Dtype) * N_*M_*K_, cudaMemcpyDeviceToDevice);

		// forward_wrt_top<Dtype><<<CAFFE_GET_BLOCKS(N_*M_), CAFFE_CUDA_NUM_THREADS>>>
		// 							(x_data, w_data, top_data + top[0]->offset(n), K_, N_, num_kernel_, width_out_, height_out_, RN_, N_ * M_);
		// CUDA_POST_KERNEL_CHECK;

		// dim3 NUM_BLOCKS(M_,N_);
		// for (int I = 0; I < K_; I += CAFFE_CUDA_NUM_THREADS)	{
		// 	forward_wrt_top_reduce<Dtype><<<NUM_BLOCKS, CAFFE_CUDA_NUM_THREADS>>>
		// 								(x_data, w_data, top_data + top[0]->offset(n), K_, N_, I);
		// }

		// int RESTK = K_ % CAFFE_CUDA_NUM_THREADS;
		// forward_wrt_top_reduce<Dtype><<<NUM_BLOCKS, RESTK>>>
		// 								(x_data, w_data, top_data + top[0]->offset(n), K_, N_, K_/CAFFE_CUDA_NUM_THREADS);

		// transpose<Dtype><<<CAFFE_GET_BLOCKS(N_*K_),
		// 					CAFFE_CUDA_NUM_THREADS>>>(x_data, slow_data, M_, K_, N_, num_kernel_, width_out_, height_out_);
		// CUDA_POST_KERNEL_CHECK;
	
		// // for (int m=0; m<num_output_; m++) { 
		// caffe_gpu_mul(K_*N_, slow_data, w_data + this->weight_buffer_.offset(m),
		// 					intermediate.mutable_gpu_data());

		forward_wrt_top_inter<Dtype><<<CAFFE_GET_BLOCKS(N_*M_*K_), CAFFE_CUDA_NUM_THREADS>>>
									(x_data, w_data, inter_data,	K_, N_, RNK_, RK_, N_*M_*K_);

		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, N_*M_, 1, K_,
													(Dtype)1., inter_data, E.gpu_data(), 
													(Dtype)0., top_data + top[0]->offset(n));

		// show<Dtype><<<1,1>>>(inter_data,K_*180,180);
		// show<Dtype><<<1,1>>>(top_data + top[0]->offset(n),2,1);

		// if (bias_term_) {
		//  caffe_gpu_add(M_ * N_, b_data,
		//            top_data + top[0]->offset(n),
		//            top_data + top[0]->offset(n));}
	}
}

template <typename Dtype>
void SharedWeightsLocalLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	const Dtype* top_diff = top[0]->gpu_diff();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	Dtype* x_data = col_buffer_.mutable_gpu_data();
	Dtype* x_diff = col_buffer_.mutable_gpu_diff();
	const Dtype* weight = this->blobs_[0]->gpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
	// Dtype* bias_diff = NULL;
	Dtype* w_data = weight_buffer_.mutable_gpu_data();

	// if (bias_term_) {
	//  bias_diff = this->blobs_[1]->mutable_gpu_diff();
	//  CUDA_CHECK(cudaMemset(bias_diff, 0, sizeof(Dtype) * this->blobs_[1]->count()));
	//  for (int n = 0; n < num_; ++n) {
	//    for (int m = 0; m < M_; m++){
	//      for (int r = 0; r < N_; r++){
	//        int _h_ = r / width_out_;
	//        int _w_ = r % width_out_;
	//        int _hk_ = _h_ % num_kernel_;
	//        int _wk_ = _w_ % num_kernel_;     
	//        caffe_gpu_add(1, bias_diff + this->blobs_[1]->offset(m,0,_hk_*num_kernel_+_wk_),
	//                top_diff + top[0]->offset(n,m) + _h_*width_out_+_w_,
	//                bias_diff + this->blobs_[1]->offset(m,0,_hk_*num_kernel_+_wk_));
	//      }
	//    }
	//  }
	// }

	create_buffer_kernel<Dtype><<<CAFFE_GET_BLOCKS(N_*M_*K_),CAFFE_CUDA_NUM_THREADS>>>
											(weight, w_data, K_, N_, num_kernel_, width_out_, N_*M_*K_, RNK_, RWK_, RK_, RL_);

	CUDA_CHECK(cudaMemset(weight_diff, 0, sizeof(Dtype) * this->blobs_[0]->count()));
	for (int n=0; n<num_; n++) {
		im2col_gpu(bottom_data + bottom[0]->offset(n), channels_, height_,
							 width_, kernel_size_, kernel_size_, pad_, pad_, stride_, stride_, x_data);

		// gradient wrt weight
		gradient_wrt_weight<Dtype><<<CAFFE_GET_BLOCKS(S_*M_*K_), CAFFE_CUDA_NUM_THREADS>>>
										(top_diff + top[0]->offset(n), x_data, weight_diff, S_, K_, N_, num_kernel_, width_out_, height_out_, RSK_, RK_, RL_, S_*M_*K_);
		//CUDA_POST_KERNEL_CHECK;

		// gradient wrt bottom data
		if (propagate_down[0]) {
			CUDA_CHECK(cudaMemset(x_diff, 0, col_buffer_.count() * sizeof(Dtype))); 
			gradient_wrt_bottom<Dtype><<<CAFFE_GET_BLOCKS(N_*K_), CAFFE_CUDA_NUM_THREADS>>>
										(top_diff + top[0]->offset(n), w_data, x_diff, M_, K_, N_, num_kernel_, RK_, N_*K_);
			//CUDA_POST_KERNEL_CHECK;

			// col2im back to the data
			col2im_gpu(x_diff, channels_, height_, width_, kernel_size_, kernel_size_,
								 pad_, pad_, stride_, stride_, bottom_diff + bottom[0]->offset(n));
		}
	}
}
INSTANTIATE_LAYER_GPU_FUNCS(SharedWeightsLocalLayer);

}  // namespace caffe