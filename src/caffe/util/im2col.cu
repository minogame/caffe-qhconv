#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/im2col.hpp"

#define DATA_COL_PTR_CONSTRAIN(pos) *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ? data_im_ptr[(pos)] : 0
#define DATA_COL_PTR_NEXT data_col_ptr += N
#define ADD_DATA(pos) val += data_col[(pos)]

namespace caffe {

template <typename Dtype>
__global__ void im2col_gpu_kernel(const int n, const Dtype* data_im,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int height_col, const int width_col,
		Dtype* data_col) {
	CUDA_KERNEL_LOOP(index, n) {
		int w_out = index % width_col;
		int h_index = index / width_col;
		int h_out = h_index % height_col;
		int channel_in = h_index / height_col;
		int channel_out = channel_in * kernel_h * kernel_w;
		int h_in = h_out * stride_h - pad_h;
		int w_in = w_out * stride_w - pad_w;
		Dtype* data_col_ptr = data_col;
		data_col_ptr += (channel_out * height_col + h_out) * width_col + w_out;
		const Dtype* data_im_ptr = data_im;
		data_im_ptr += (channel_in * height + h_in) * width + w_in;
		for (int i = 0; i < kernel_h; ++i) {
			for (int j = 0; j < kernel_w; ++j) {
				int h = h_in + i;
				int w = w_in + j;
				*data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
						data_im_ptr[i * width + j] : 0;
				data_col_ptr += height_col * width_col;
			}
		}
	}
}

template <typename Dtype>
void im2col_gpu(const Dtype* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		Dtype* data_col) {
	// We are going to launch channels * height_col * width_col kernels, each
	// kernel responsible for copying a single-channel grid.
	int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
	int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
	int num_kernels = channels * height_col * width_col;
	// NOLINT_NEXT_LINE(whitespace/operators)
	im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
														 CAFFE_CUDA_NUM_THREADS>>>(
			num_kernels, data_im, height, width, kernel_h, kernel_w, pad_h,
			pad_w, stride_h, stride_w, height_col,
			width_col, data_col);
	CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void im2col_gpu<float>(const float* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h, const int stride_w,
		float* data_col);
template void im2col_gpu<double>(const double* data_im, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w, const int stride_h, const int stride_w,
		double* data_col);

template <typename Dtype>
__global__ void col2im_gpu_kernel(const int n, const Dtype* data_col,
		const int height, const int width, const int channels,
		const int patch_h, const int patch_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int height_col, const int width_col,
		Dtype* data_im) {
	CUDA_KERNEL_LOOP(index, n) {
		Dtype val = 0;
		int w = index % width + pad_w;
		int h = (index / width) % height + pad_h;
		int c = index / (width * height);
		// compute the start and end of the output
		int w_col_start = (w < patch_w) ? 0 : (w - patch_w) / stride_w + 1;
		int w_col_end = min(w / stride_w + 1, width_col);
		int h_col_start = (h < patch_h) ? 0 : (h - patch_h) / stride_h + 1;
		int h_col_end = min(h / stride_h + 1, height_col);

		// equivalent implementation
		int offset =
				(c * patch_h * patch_w + h * patch_w + w) * height_col * width_col;
		int coeff_h_col = (1 - stride_h * patch_w * height_col) * width_col;
		int coeff_w_col = (1 - stride_w * height_col * width_col);
		for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
			for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
				ADD_DATA(offset + h_col * coeff_h_col + w_col * coeff_w_col);
			}
		}
		data_im[index] = val;
	}
}

template <typename Dtype>
void col2im_gpu(const Dtype* data_col, const int channels,
		const int height, const int width, const int patch_h, const int patch_w,
		const int pad_h, const int pad_w, const int stride_h,
		const int stride_w, Dtype* data_im) {
	int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
	int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
	int num_kernels = channels * height * width;
	// To avoid involving atomic operations, we will launch one kernel per
	// bottom dimension, and then in the kernel add up the top dimensions.
	// NOLINT_NEXT_LINE(whitespace/operators)
	col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),
														 CAFFE_CUDA_NUM_THREADS>>>(
			num_kernels, data_col, height, width, channels, patch_h, patch_w,
			pad_h, pad_w, stride_h, stride_w,
			height_col, width_col, data_im);
	CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void col2im_gpu<float>(const float* data_col, const int channels,
		const int height, const int width, const int patch_h, const int patch_w,
		const int pad_h, const int pad_w, const int stride_h,
		const int stride_w, float* data_im);
template void col2im_gpu<double>(const double* data_col, const int channels,
		const int height, const int width, const int patch_h, const int patch_w,
		const int pad_h, const int pad_w, const int stride_h,
		const int stride_w, double* data_im);

/**************************************4 DIRECTIONS HEX*******************************************/
template <typename Dtype>
__global__ void hex_im2col_gpu_kernel(const int n, const Dtype* data_im,
		const int height, const int width, Dtype* data_col, const int direction) {
	CUDA_KERNEL_LOOP(index, n) {
		int w_out = index % width;
		int h_index = index / width;
		int h_out = h_index % height;
		int channel_in = h_index / height;
		int channel_out = channel_in * 7;
		int N = height * width;

		const Dtype* data_im_ptr = data_im;
		Dtype* data_col_ptr = data_col;
		data_col_ptr += channel_out * N + h_out * width + w_out;


		int h = h_out;
		int w = w_out;

		switch (direction) {
		case 0:
			data_im_ptr += (channel_in * height + h_out - 1) * width + w_out - 1;

																										h--; w--;
			DATA_COL_PTR_CONSTRAIN(0); DATA_COL_PTR_NEXT; w++;
			DATA_COL_PTR_CONSTRAIN(1); DATA_COL_PTR_NEXT; h++; w--;
			DATA_COL_PTR_CONSTRAIN(width); DATA_COL_PTR_NEXT; w++;
			DATA_COL_PTR_CONSTRAIN(width+1); DATA_COL_PTR_NEXT; w++;
			DATA_COL_PTR_CONSTRAIN(width+2); DATA_COL_PTR_NEXT; h++; w-=2;
			DATA_COL_PTR_CONSTRAIN(width*2); DATA_COL_PTR_NEXT; w++;
			DATA_COL_PTR_CONSTRAIN(width*2+1);
			break;

		case 1:
			data_im_ptr += (channel_in * height + h_out - 1) * width + w_out;//delete[-1]
			
																										h--;
			DATA_COL_PTR_CONSTRAIN(0); DATA_COL_PTR_NEXT; w++;
			DATA_COL_PTR_CONSTRAIN(1); DATA_COL_PTR_NEXT; h++; w-=2;
			DATA_COL_PTR_CONSTRAIN(width-1); DATA_COL_PTR_NEXT; w++;
			DATA_COL_PTR_CONSTRAIN(width); DATA_COL_PTR_NEXT; w++;
			DATA_COL_PTR_CONSTRAIN(width+1); DATA_COL_PTR_NEXT; h++; w--;
			DATA_COL_PTR_CONSTRAIN(width*2); DATA_COL_PTR_NEXT; w++;
			DATA_COL_PTR_CONSTRAIN(width*2+1);
			break;

		case 2:
			data_im_ptr += (channel_in * height + h - 1) * width + w - 1;

			DATA_COL_PTR_CONSTRAIN(width+1); DATA_COL_PTR_NEXT; w--;
			DATA_COL_PTR_CONSTRAIN(width); DATA_COL_PTR_NEXT; h--; w++;
			DATA_COL_PTR_CONSTRAIN(1); DATA_COL_PTR_NEXT; h++; w++;
			DATA_COL_PTR_CONSTRAIN(width+2); DATA_COL_PTR_NEXT; h++;
			DATA_COL_PTR_CONSTRAIN(width*2+2); DATA_COL_PTR_NEXT; w--;
			DATA_COL_PTR_CONSTRAIN(width*2+1); DATA_COL_PTR_NEXT; w--;
			DATA_COL_PTR_CONSTRAIN(width*2);
			break;

		case 3:
			data_im_ptr += (channel_in * height + h - 1) * width + w - 1;

			DATA_COL_PTR_CONSTRAIN(width+1); DATA_COL_PTR_NEXT; w--; h--;
			DATA_COL_PTR_CONSTRAIN(0); DATA_COL_PTR_NEXT; w++;
			DATA_COL_PTR_CONSTRAIN(1); DATA_COL_PTR_NEXT; w++;
			DATA_COL_PTR_CONSTRAIN(2); DATA_COL_PTR_NEXT; h++;
			DATA_COL_PTR_CONSTRAIN(width+2); DATA_COL_PTR_NEXT; h++; w--;
			DATA_COL_PTR_CONSTRAIN(width*2+1); DATA_COL_PTR_NEXT; h--; w--;
			DATA_COL_PTR_CONSTRAIN(width);
			break;

		case 4:
			data_im_ptr += (channel_in * height + h - 1) * width + w - 1;

			DATA_COL_PTR_CONSTRAIN(width+1); DATA_COL_PTR_NEXT; w--; h--;
			DATA_COL_PTR_CONSTRAIN(0); DATA_COL_PTR_NEXT; w+=2; h++;
			DATA_COL_PTR_CONSTRAIN(width+2); DATA_COL_PTR_NEXT; h++;
			DATA_COL_PTR_CONSTRAIN(width*2+2); DATA_COL_PTR_NEXT; w--;
			DATA_COL_PTR_CONSTRAIN(width*2+1); DATA_COL_PTR_NEXT; w--;
			DATA_COL_PTR_CONSTRAIN(width*2); DATA_COL_PTR_NEXT; h--;
			DATA_COL_PTR_CONSTRAIN(width);
			break;

		case 5:
			data_im_ptr += (channel_in * height + h - 1) * width + w - 1;

			DATA_COL_PTR_CONSTRAIN(width+1); DATA_COL_PTR_NEXT; w++; h--;
			DATA_COL_PTR_CONSTRAIN(2); DATA_COL_PTR_NEXT; h++;
			DATA_COL_PTR_CONSTRAIN(width+2); DATA_COL_PTR_NEXT; h++;
			DATA_COL_PTR_CONSTRAIN(width*2+2); DATA_COL_PTR_NEXT; w--;
			DATA_COL_PTR_CONSTRAIN(width*2+1); DATA_COL_PTR_NEXT; w--;
			DATA_COL_PTR_CONSTRAIN(width*2); DATA_COL_PTR_NEXT; h--;
			DATA_COL_PTR_CONSTRAIN(width);
			break;

		case 6:
			data_im_ptr += (channel_in * height + h - 1) * width + w - 1;

			DATA_COL_PTR_CONSTRAIN(width+1); DATA_COL_PTR_NEXT; w--; h--;
			DATA_COL_PTR_CONSTRAIN(0); DATA_COL_PTR_NEXT; w++;
			DATA_COL_PTR_CONSTRAIN(1); DATA_COL_PTR_NEXT; w++;
			DATA_COL_PTR_CONSTRAIN(2); DATA_COL_PTR_NEXT; w--; h+=2;
			DATA_COL_PTR_CONSTRAIN(width*2+1); DATA_COL_PTR_NEXT; w--;
			DATA_COL_PTR_CONSTRAIN(width*2); DATA_COL_PTR_NEXT; h--;
			DATA_COL_PTR_CONSTRAIN(width);
			break;

		case 7:
			data_im_ptr += (channel_in * height + h - 1) * width + w - 1;

			DATA_COL_PTR_CONSTRAIN(width+1); DATA_COL_PTR_NEXT; w--; h--;
			DATA_COL_PTR_CONSTRAIN(0); DATA_COL_PTR_NEXT; w++;
			DATA_COL_PTR_CONSTRAIN(1); DATA_COL_PTR_NEXT; w++; h+=2;
			DATA_COL_PTR_CONSTRAIN(width*2); DATA_COL_PTR_NEXT; w--;
			DATA_COL_PTR_CONSTRAIN(width*2+1); DATA_COL_PTR_NEXT; w--;
			DATA_COL_PTR_CONSTRAIN(width*2); DATA_COL_PTR_NEXT; h--;
			DATA_COL_PTR_CONSTRAIN(width);
			break;

		case 8:
			data_im_ptr += (channel_in * height + h - 1) * width + w - 1;

			DATA_COL_PTR_CONSTRAIN(width+1); DATA_COL_PTR_NEXT; w--; h--;
			DATA_COL_PTR_CONSTRAIN(0); DATA_COL_PTR_NEXT; w++;
			DATA_COL_PTR_CONSTRAIN(1); DATA_COL_PTR_NEXT; w++;
			DATA_COL_PTR_CONSTRAIN(2); DATA_COL_PTR_NEXT; h++;
			DATA_COL_PTR_CONSTRAIN(width+2); DATA_COL_PTR_NEXT; h++;
			DATA_COL_PTR_CONSTRAIN(width*2+2); DATA_COL_PTR_NEXT; h--; w-=2;
			DATA_COL_PTR_CONSTRAIN(width);
			break;

		case 9:
			data_im_ptr += (channel_in * height + h - 1) * width + w - 1;

			DATA_COL_PTR_CONSTRAIN(width+1); DATA_COL_PTR_NEXT; w--; h--;
			DATA_COL_PTR_CONSTRAIN(0); DATA_COL_PTR_NEXT; w++;
			DATA_COL_PTR_CONSTRAIN(1); DATA_COL_PTR_NEXT; w++;
			DATA_COL_PTR_CONSTRAIN(2); DATA_COL_PTR_NEXT; h++;
			DATA_COL_PTR_CONSTRAIN(width+2); DATA_COL_PTR_NEXT; h++; w-=2;
			DATA_COL_PTR_CONSTRAIN(width*2+1); DATA_COL_PTR_NEXT; h--;
			DATA_COL_PTR_CONSTRAIN(width);
			break;

		case 10:
			data_im_ptr += (channel_in * height + h - 1) * width + w - 1;

			DATA_COL_PTR_CONSTRAIN(width+1); DATA_COL_PTR_NEXT; h--;
			DATA_COL_PTR_CONSTRAIN(1); DATA_COL_PTR_NEXT; w++;
			DATA_COL_PTR_CONSTRAIN(2); DATA_COL_PTR_NEXT; h++;
			DATA_COL_PTR_CONSTRAIN(width+2); DATA_COL_PTR_NEXT; h++;
			DATA_COL_PTR_CONSTRAIN(width*2+2); DATA_COL_PTR_NEXT; w--;
			DATA_COL_PTR_CONSTRAIN(width*2+1); DATA_COL_PTR_NEXT; w--;
			DATA_COL_PTR_CONSTRAIN(width*2);
			break;

		case 11:
			data_im_ptr += (channel_in * height + h - 1) * width + w - 1;

			DATA_COL_PTR_CONSTRAIN(width+1); DATA_COL_PTR_NEXT; w--; h--;
			DATA_COL_PTR_CONSTRAIN(0); DATA_COL_PTR_NEXT; w++;
			DATA_COL_PTR_CONSTRAIN(1); DATA_COL_PTR_NEXT; w++;
			DATA_COL_PTR_CONSTRAIN(2); DATA_COL_PTR_NEXT; h++;
			DATA_COL_PTR_CONSTRAIN(width+2); DATA_COL_PTR_NEXT; h++;
			DATA_COL_PTR_CONSTRAIN(width*2+2); DATA_COL_PTR_NEXT; w--;
			DATA_COL_PTR_CONSTRAIN(width*2+1);
			break;

		case 12:
			data_im_ptr += (channel_in * height + h - 1) * width + w - 1;

			DATA_COL_PTR_CONSTRAIN(width+1); DATA_COL_PTR_NEXT; h--;
			DATA_COL_PTR_CONSTRAIN(1); DATA_COL_PTR_NEXT; w++;
			DATA_COL_PTR_CONSTRAIN(2); DATA_COL_PTR_NEXT; h++;
			DATA_COL_PTR_CONSTRAIN(width+2); DATA_COL_PTR_NEXT; h++; w--;
			DATA_COL_PTR_CONSTRAIN(width*2+1); DATA_COL_PTR_NEXT; w--;
			DATA_COL_PTR_CONSTRAIN(width*2); DATA_COL_PTR_NEXT; h--;
			DATA_COL_PTR_CONSTRAIN(width);
			break;

		case 13:
			data_im_ptr += (channel_in * height + h - 1) * width + w - 1;

			DATA_COL_PTR_CONSTRAIN(width+1); DATA_COL_PTR_NEXT; w--; h--;
			DATA_COL_PTR_CONSTRAIN(0); DATA_COL_PTR_NEXT; w++;
			DATA_COL_PTR_CONSTRAIN(1); DATA_COL_PTR_NEXT; h++; w++;
			DATA_COL_PTR_CONSTRAIN(width+2); DATA_COL_PTR_NEXT; h++;
			DATA_COL_PTR_CONSTRAIN(width*2+2); DATA_COL_PTR_NEXT; w--;
			DATA_COL_PTR_CONSTRAIN(width*2+1); DATA_COL_PTR_NEXT; h--; w--;
			DATA_COL_PTR_CONSTRAIN(width);
			break;
		}
	}
}

template <typename Dtype>
void hex_im2col_gpu(const Dtype* data_im, const int channels,
		const int height, const int width, Dtype* data_col, const int direction) {
	// We are going to launch channels * height_col * width_col kernels, each
	// kernel responsible for copying a single-channel grid.
	int num_kernels = channels * height * width;
	// NOLINT_NEXT_LINE(whitespace/operators)
	hex_im2col_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels), CAFFE_CUDA_NUM_THREADS>>>
			(num_kernels, data_im, height, width, data_col, direction);
	CUDA_POST_KERNEL_CHECK;
}


// Explicit instantiation
template void hex_im2col_gpu<float>(const float* data_im, const int channels,
		const int height, const int width, float* data_col, const int direction);
template void hex_im2col_gpu<double>(const double* data_im, const int channels,
		const int height, const int width, double* data_col, const int direction);


template <typename Dtype>
__global__ void hex_col2im_gpu_kernel(const int n, const Dtype* data_col,
		const int height, const int width, const int channels, Dtype* data_im, const int direction) {
	CUDA_KERNEL_LOOP(index, n) {
		Dtype val = 0;
		int w = index % width;
		int h = (index / width) % height;
		int c = index / (width * height);
		int N = height * width;
		int offset = c * 7 * N;
		int hww = h * width + w;

		switch (direction) {
		case 0:
			if (h + 1 < height && w + 1 < width)  ADD_DATA(offset + 0 * N + hww + width + 1);
			if (h + 1 < height) 									ADD_DATA(offset + 1 * N + hww + width);
			if (w + 1 < width) 										ADD_DATA(offset + 2 * N + hww + 1);
																						ADD_DATA(offset + 3 * N + hww);
			if (w > 0)														ADD_DATA(offset + 4 * N + hww - 1);
			if (h > 0 && w + 1 < width) 					ADD_DATA(offset + 5 * N + hww - width + 1);
			if (h > 0) 														ADD_DATA(offset + 6 * N + hww - width);
			break;

		case 1:
			if (h + 1 < height) 									ADD_DATA(offset + 0 * N + hww + width);
			if (h + 1 < height && w > 0)					ADD_DATA(offset + 1 * N + hww + width - 1);
			if (w + 1 < width) 										ADD_DATA(offset + 2 * N + hww + 1);
																						ADD_DATA(offset + 3 * N + hww);
			if (w > 0) 														ADD_DATA(offset + 4 * N + hww - 1);
			if (h > 0) 														ADD_DATA(offset + 5 * N + hww - width);
			if (h > 0 && w > 0 ) 									ADD_DATA(offset + 6 * N + hww - width - 1);
			break;

		case 2:
																						ADD_DATA(offset + 0 * N + hww);
			if (h + 1 < height) 									ADD_DATA(offset + 2 * N + hww + width);
			if (h > 0) 														ADD_DATA(offset + 5 * N + hww - width);
			if (w + 1 < width) 										ADD_DATA(offset + 1 * N + hww + 1);
			if (w > 0) 														ADD_DATA(offset + 3 * N + hww - 1);
			if (h > 0 && w > 0) 									ADD_DATA(offset + 4 * N + hww - width - 1);
			if (h > 0 && w + 1 < width) 					ADD_DATA(offset + 6 * N + hww - width + 1);
			break;

		case 3:
																						ADD_DATA(offset + 0 * N + hww);
			if (h + 1 < height) 									ADD_DATA(offset + 2 * N + hww + width);
			if (h > 0) 														ADD_DATA(offset + 5 * N + hww - width);
			if (h + 1 < height && w + 1 < width) 	ADD_DATA(offset + 1 * N + hww + width + 1);
			if (h + 1 < height && w > 0) 					ADD_DATA(offset + 3 * N + hww + width - 1);
			if (w > 0) 														ADD_DATA(offset + 4 * N + hww - 1);
			if (w + 1 < width) 										ADD_DATA(offset + 6 * N + hww + 1);
			break;

		case 4:
																						ADD_DATA(offset + 0 * N + hww);
			if (w + 1 < width && h + 1 < height) 	ADD_DATA(offset + 1 * N + hww + width + 1);
			if (w > 0)														ADD_DATA(offset + 2 * N + hww - 1);					
			if (h > 0 && w > 0)										ADD_DATA(offset + 3 * N + hww - width - 1);
			if (h > 0)														ADD_DATA(offset + 4 * N + hww - width);
			if (h > 0 && w + 1 < width)						ADD_DATA(offset + 5 * N + hww - width + 1);
			if (w + 1 < width)										ADD_DATA(offset + 6 * N + hww + 1);
			break;

		case 5:
																						ADD_DATA(offset + 0 * N + hww);
			if (w > 0 && h + 1 < height)				 	ADD_DATA(offset + 1 * N + hww + width - 1);
			if (w > 0)														ADD_DATA(offset + 2 * N + hww - 1);					
			if (h > 0 && w > 0)										ADD_DATA(offset + 3 * N + hww - width - 1);
			if (h > 0)														ADD_DATA(offset + 4 * N + hww - width);
			if (h > 0 && w + 1 < width)						ADD_DATA(offset + 5 * N + hww - width + 1);
			if (w + 1 < width)										ADD_DATA(offset + 6 * N + hww + 1);
			break;

		case 6:
																						ADD_DATA(offset + 0 * N + hww);
			if (h + 1 < height && w + 1 < width) 	ADD_DATA(offset + 1 * N + hww + width + 1);
			if (h + 1 < height) 									ADD_DATA(offset + 2 * N + hww + width);
			if (h + 1 < height && w > 0) 					ADD_DATA(offset + 3 * N + hww + width - 1);
			if (h > 0) 														ADD_DATA(offset + 4 * N + hww - width);
			if (h > 0 && w + 1 < width) 					ADD_DATA(offset + 5 * N + hww - width + 1);
			if (w + 1 < width) 										ADD_DATA(offset + 6 * N + hww + 1);
			break;

		case 7:
																						ADD_DATA(offset + 0 * N + hww);
			if (h + 1 < height && w + 1 < width) 	ADD_DATA(offset + 1 * N + hww + width + 1);
			if (h + 1 < height) 									ADD_DATA(offset + 2 * N + hww + width);
			if (h > 0 && w > 0) 									ADD_DATA(offset + 3 * N + hww - width - 1);
			if (h > 0) 														ADD_DATA(offset + 4 * N + hww - width);
			if (h > 0 && w + 1 < width) 					ADD_DATA(offset + 5 * N + hww - width + 1);
			if (w + 1 < width) 										ADD_DATA(offset + 6 * N + hww + 1);
			break;

		case 8:
																						ADD_DATA(offset + 0 * N + hww);
			if (h + 1 < height && w + 1 < width) 	ADD_DATA(offset + 1 * N + hww + width + 1);
			if (h + 1 < height) 									ADD_DATA(offset + 2 * N + hww + width);
			if (h + 1 < height && w > 0) 					ADD_DATA(offset + 3 * N + hww + width - 1);
			if (w > 0) 														ADD_DATA(offset + 4 * N + hww - 1);
			if (h > 0 && w > 0) 									ADD_DATA(offset + 5 * N + hww - width - 1);
			if (w + 1 < width) 										ADD_DATA(offset + 6 * N + hww + 1);
			break;

		case 9:
																						ADD_DATA(offset + 0 * N + hww);
			if (h + 1 < height && w + 1 < width) 	ADD_DATA(offset + 1 * N + hww + width + 1);
			if (h + 1 < height) 									ADD_DATA(offset + 2 * N + hww + width);
			if (h + 1 < height && w > 0) 					ADD_DATA(offset + 3 * N + hww + width - 1);
			if (w > 0) 														ADD_DATA(offset + 4 * N + hww - 1);
			if (h > 0 && w + 1 < width) 					ADD_DATA(offset + 5 * N + hww - width + 1);
			if (w + 1 < width) 										ADD_DATA(offset + 6 * N + hww + 1);
			break;

		case 10:
																						ADD_DATA(offset + 0 * N + hww);
			if (h + 1 < height) 									ADD_DATA(offset + 1 * N + hww + width);
			if (h + 1 < height && w > 0) 					ADD_DATA(offset + 2 * N + hww + width - 1);
			if (w > 0) 														ADD_DATA(offset + 3 * N + hww - 1);
			if (h > 0 && w > 0) 									ADD_DATA(offset + 4 * N + hww - width - 1);
			if (h > 0) 														ADD_DATA(offset + 5 * N + hww - width);
			if (h > 0 && w + 1 < width) 					ADD_DATA(offset + 6 * N + hww - width + 1);
			break;

		case 11:
																						ADD_DATA(offset + 0 * N + hww);
			if (h + 1 < height && w + 1 < width) 	ADD_DATA(offset + 1 * N + hww + width + 1);
			if (h + 1 < height) 									ADD_DATA(offset + 2 * N + hww + width);
			if (h + 1 < height && w > 0) 					ADD_DATA(offset + 3 * N + hww + width - 1);
			if (w > 0) 														ADD_DATA(offset + 4 * N + hww - 1);
			if (h > 0 && w > 0) 									ADD_DATA(offset + 5 * N + hww - width - 1);
			if (h > 0) 														ADD_DATA(offset + 6 * N + hww - width);
			break;

		case 12:
																						ADD_DATA(offset + 0 * N + hww);
			if (h + 1 < height) 									ADD_DATA(offset + 1 * N + hww + width);
			if (h + 1 < height && w > 0) 					ADD_DATA(offset + 2 * N + hww + width - 1);
			if (w > 0) 														ADD_DATA(offset + 3 * N + hww - 1);
			if (h > 0) 														ADD_DATA(offset + 4 * N + hww - width);
			if (h > 0 && w + 1 < width) 					ADD_DATA(offset + 5 * N + hww - width + 1);
			if (w + 1 < width) 										ADD_DATA(offset + 6 * N + hww + 1);
			break;

		case 13:
																						ADD_DATA(offset + 0 * N + hww);
			if (h + 1 < height && w + 1 < width) 	ADD_DATA(offset + 1 * N + hww + width + 1);
			if (h + 1 < height) 									ADD_DATA(offset + 2 * N + hww + width);
			if (w > 0) 														ADD_DATA(offset + 3 * N + hww - 1);
			if (h > 0 && w > 0) 									ADD_DATA(offset + 4 * N + hww - width - 1);
			if (h > 0) 														ADD_DATA(offset + 5 * N + hww - width);
			if (w + 1 < width) 										ADD_DATA(offset + 6 * N + hww + 1);
			break;
		}
		data_im[index] = val;
	}
}

template <typename Dtype>
void hex_col2im_gpu(const Dtype* data_col, const int channels,
		const int height, const int width, Dtype* data_im, const int direction) {
	int num_kernels = channels * height * width;
	// To avoid involving atomic operations, we will launch one kernel per
	// bottom dimension, and then in the kernel add up the top dimensions.
	// NOLINT_NEXT_LINE(whitespace/operators)
	hex_col2im_gpu_kernel<Dtype><<<CAFFE_GET_BLOCKS(num_kernels),CAFFE_CUDA_NUM_THREADS>>>
			(num_kernels, data_col, height, width, channels, data_im, direction);
	CUDA_POST_KERNEL_CHECK;
}

// Explicit instantiation
template void hex_col2im_gpu<float>(const float* data_col, const int channels,
		const int height, const int width, float* data_im, const int direction);
template void hex_col2im_gpu<double>(const double* data_col, const int channels,
		const int height, const int width, double* data_im, const int direction);

}  // namespace caffe