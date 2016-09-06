#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
__device__ void resample(const Dtype* bottom_data, Dtype* top_data, Dtype W, Dtype H, const int index, int width, int height) {
	const int size = width * height;
	const Dtype w_ = floorf(W);
	const Dtype h_ = floorf(H);
	const Dtype _w = fminf(w_+1,width-1);
	const Dtype _h = fminf(h_+1,height-1);
	const Dtype wTL = (W-w_)*(H-h_);
	const Dtype wTR = (_w-W)*(H-h_);
	const Dtype wBL = (W-w_)*(_h-H);
	const Dtype wBR = (_w-W)*(_h-H);
	const int pTL = w_ + h_ * width;
	const int pTR = w_ + _h * width;
	const int pBL = _w + h_ * width;
	const int pBR = _w + _h * width;

	top_data[index] = 
			wTL * bottom_data[pTL] + wTR * bottom_data[pTR] + wBR * bottom_data[pBR] + wBL * bottom_data[pBL];
	top_data[index+size] = 
			wTL * bottom_data[pTL+size] + wTR * bottom_data[pTR+size] + wBR * bottom_data[pBR+size] + wBL * bottom_data[pBL+size];
	top_data[index+size+size] = 
			wTL * bottom_data[pTL+size+size] + wTR * bottom_data[pTR+size+size] + wBR * bottom_data[pBR+size+size] + wBL * bottom_data[pBL+size+size];
}

/***********************************COLOR SHIFT***********************************/
template <typename Dtype>
__global__ void color_shift_kernel(const Dtype* bottom_data, Dtype* top_data,
																		const Dtype* eig_value, const Dtype* eig_matrix, Dtype* random_vector, int total) {
	CUDA_KERNEL_LOOP(index, total) {
		int channel = index * 3 / total;
		int pos = 3 * channel;
		top_data[index] = bottom_data[index] + eig_value[0] * random_vector[0] * eig_matrix[pos]
																					+ eig_value[1] * random_vector[1] * eig_matrix[pos+1]
																					+ eig_value[2] * random_vector[2] * eig_matrix[pos+2];
	}
}

template <typename Dtype>
void AugmentationLayer<Dtype>::do_color_shift_gpu(const Dtype* bottom_data, Dtype* top_data) {
	const Dtype* eig_value = color_shift_eig_value_.gpu_data();
	const Dtype* eig_matrix = color_shift_eig_matrix_.gpu_data();
	Dtype* random_vector_ = color_shift_random_vector_.mutable_gpu_data();
	for (int n = 0; n < num_; ++n) {
		caffe_gpu_rng_gaussian(4, color_shift_mu_, color_shift_sigma_, random_vector_);
		color_shift_kernel<Dtype><<<CAFFE_GET_BLOCKS(total_),CAFFE_CUDA_NUM_THREADS>>>
											(bottom_data + n * total_, top_data + n * total_, eig_value, eig_matrix, random_vector_, total_);
	}
}

/***********************************GAUSSIAN NOISE***********************************/
template <typename Dtype>
__global__ void gaussian_noise_kernel(const Dtype* bottom_data, Dtype* top_data,
																		const unsigned int* mask, const Dtype* random_vector, const int thershold, const int count) {
	CUDA_KERNEL_LOOP(index, count) {
		top_data[index] = (mask[index] < thershold)? bottom_data[index] + random_vector[index] : bottom_data[index];
	}
}

template <typename Dtype>
void AugmentationLayer<Dtype>::do_gaussian_noise_gpu(const Dtype* bottom_data, Dtype* top_data) {
	unsigned int* mask = static_cast<unsigned int*>(gaussian_noise_mask_.mutable_gpu_data());
	Dtype* random_vector = gaussian_noise_buffer_.mutable_gpu_data();
	caffe_gpu_rng_uniform(count_, mask);
	caffe_gpu_rng_uniform(count_, gaussian_noise_mu_, gaussian_noise_sigma_, random_vector);
	gaussian_noise_kernel<Dtype><<<CAFFE_GET_BLOCKS(count_),CAFFE_CUDA_NUM_THREADS>>>
											(bottom_data, top_data, mask, random_vector, uint_thres_, count_);
}

/***********************************CONTRAST***********************************/
template <typename Dtype>
__global__ void contrast_kernel(const Dtype* bottom_data, Dtype* top_data, const Dtype factor, const int count) {
	CUDA_KERNEL_LOOP(index, count) {
		top_data[index] = factor * (bottom_data[index] - 128) + 128;
		if (top_data[index] > 255) top_data[index] = 255.0;
		if (top_data[index] < 0) top_data[index] = 0.0;
	}
}

template <typename Dtype>
void AugmentationLayer<Dtype>::do_contrast_gpu(const Dtype* bottom_data, Dtype* top_data) {
	Dtype C = Dtype(Rand(contrast_max_thershold_*2) - contrast_max_thershold_);
	Dtype F = 1.0156868 * (C + 255) / ( 259 - C);
	contrast_kernel<Dtype><<<CAFFE_GET_BLOCKS(count_),CAFFE_CUDA_NUM_THREADS>>>
											(bottom_data, top_data, F, count_);
}

/***********************************FLARE***********************************/
template <typename Dtype>
__global__ void flare_kernel(const Dtype* bottom_data, Dtype* top_data, const int radius, const int lumi, const int centerH, const int centerW, 
											const int height, const int width, const int total) {
	CUDA_KERNEL_LOOP(index, total) {
		const int size = width * height;
		const int CR = index % size;
		const int H = CR / width;
		const int W = CR % width;
		const int dH = H - centerH;
		const int dW = W - centerW;
		const Dtype dis = sqrtf(dH * dH + dW * dW) + 0.00001;

		top_data[index] = bottom_data[index] + lumi * fmaxf(1 - dis / radius, 0);
		if (top_data[index] > 255) top_data[index] = 255.0;
		if (top_data[index] < 0) top_data[index] = 0.0;
	}
}

template <typename Dtype>
void AugmentationLayer<Dtype>::do_flare_gpu(const Dtype* bottom_data, Dtype* top_data) {
	if (flare_num_ == 1) {
		for (int n = 0; n < num_; ++n) {
			int centerH = Rand(height_);
			int centerW = Rand(width_);
			int radius = flare_min_radius_ + Rand(flare_max_radius_ - flare_min_radius_);
			int lumi = flare_min_lumi_ + Rand(flare_max_lumi_ - flare_min_lumi_);
			flare_kernel<Dtype><<<CAFFE_GET_BLOCKS(count_),CAFFE_CUDA_NUM_THREADS>>>
												(bottom_data + n * total_, top_data + n * total_, radius, lumi, centerH, centerW, height_, width_ ,total_);
		}
	} else {
		Dtype* buffer_ = flare_buffer_.mutable_gpu_data();
		caffe_copy(count_, bottom_data, buffer_);
		for (int i = 0; i < flare_num_; ++i) {
			for (int n = 0; n < num_; ++n) {
				int centerH = Rand(height_);
				int centerW = Rand(width_);
				int radius = flare_min_radius_ + Rand(flare_max_radius_ - flare_min_radius_);
				int lumi = flare_min_lumi_ + Rand(flare_max_lumi_ - flare_min_lumi_);
				flare_kernel<Dtype><<<CAFFE_GET_BLOCKS(count_),CAFFE_CUDA_NUM_THREADS>>>
													(buffer_ + n * total_, top_data + n * total_, radius, lumi, centerH, centerW, height_, width_ ,total_);
			}
			if (i != flare_num_ - 1) caffe_copy(count_, top_data, buffer_);
		}
	}
}

/***********************************BLUR***********************************/
template <typename Dtype>
__global__ void blur_kernel(const Dtype* bottom_data, Dtype* top_data, const Dtype t, const Dtype sum, 
											const int height, const int width, const int total) {
	CUDA_KERNEL_LOOP(index, total) {
		const int size = width * height;
		const int CR = index % size;
		const int H = CR / width;
		const int W = CR % width;

		if (H == 0 || H == (height -1) || W == 0 || W == (width - 1))
			top_data[index] = bottom_data[index];
		else {
			top_data[index] = (bottom_data[index] + t * (bottom_data[index+1] + bottom_data[index-1] + bottom_data[index+width] + bottom_data[index-width])
					+ t * t * (bottom_data[index+width+1] + bottom_data[index-width-1] + bottom_data[index+width-1] + bottom_data[index-width+1])) / sum;
		}
	}
}

template <typename Dtype>
void AugmentationLayer<Dtype>::do_blur_gpu(const Dtype* bottom_data, Dtype* top_data) {
	for (int n = 0; n < num_; ++n) {
		caffe_rng_uniform(1, blur_min_sigma_, blur_max_sigma_, &blur_sigma_);
		const Dtype t = std::exp(- (1/blur_sigma_) * (1/blur_sigma_) / 2);
		const Dtype sum = 4 * t * t + 4 * t + 1;
		blur_kernel<Dtype><<<CAFFE_GET_BLOCKS(count_),CAFFE_CUDA_NUM_THREADS>>>
											(bottom_data + n * total_, top_data + n * total_, t, sum, height_, width_ ,total_);
	}
}

/***********************************COLOR BIAS***********************************/
template <typename Dtype>
__global__ void color_bias_kernel(const Dtype* bottom_data, Dtype* top_data, const int size, const int bias, const int bias_ch) {
	CUDA_KERNEL_LOOP(index, size) {
		const int pos = index + bias_ch * size;
		top_data[pos] = bottom_data[pos] + bias;
		if (top_data[pos] > 255) top_data[pos] = 255.0;
		if (top_data[pos] < 0) top_data[pos] = 0.0;
	}
}

template <typename Dtype>
void AugmentationLayer<Dtype>::do_color_bias_gpu(const Dtype* bottom_data, Dtype* top_data) {
	for (int n = 0; n < num_; ++n) {
		const int size = height_ * width_;
		const int bias = Rand(color_max_bias_);
		const int bias_ch = Rand(3);
		caffe_copy(count_, bottom_data, top_data);
		color_bias_kernel<Dtype><<<CAFFE_GET_BLOCKS(size),CAFFE_CUDA_NUM_THREADS>>>
											(bottom_data + n * total_, top_data + n * total_, size, bias, bias_ch);
	}
}

/***********************************MIRROR***********************************/
template <typename Dtype>
__global__ void mirror_kernel(const Dtype* bottom_data, Dtype* top_data, const int size, const int width) {
	CUDA_KERNEL_LOOP(index, size) {
		const int W = index % width;
		top_data[index] = bottom_data[index - W + width - 1 - W];
		top_data[index+size] = bottom_data[index+size - W + width - 1 - W];
		top_data[index+size+size] = bottom_data[index+size+size - W + width - 1 - W];
	}
}

template <typename Dtype>
void AugmentationLayer<Dtype>::do_mirror_gpu(const Dtype* bottom_data, Dtype* top_data) {
	const int size = height_ * width_;
	for (int n = 0; n < num_; ++n) {
		mirror_kernel<Dtype><<<CAFFE_GET_BLOCKS(size),CAFFE_CUDA_NUM_THREADS>>>
											(bottom_data + n * total_, top_data + n * total_, size, width_);
	}
}

/***********************************SATURATION***********************************/
template <typename Dtype>
__global__ void saturation_kernel(const Dtype* bottom_data, Dtype* top_data, const int size, const int width, Dtype bias) {
	CUDA_KERNEL_LOOP(index, size) {
		//Weights are ordered by B R G.
		const Dtype param = sqrtf(bottom_data[index] * bottom_data[index] * 0.114 + 
															bottom_data[index+size] * bottom_data[index+size] * 0.587 +
															bottom_data[index+size+size] * bottom_data[index+size+size] * 0.299);

		top_data[index] = param + (bottom_data[index] - param) * bias;
		top_data[index+size] = param + (bottom_data[index+size] - param) * bias;
		top_data[index+size+size] = param + (bottom_data[index+size+size] - param) * bias;
		if (top_data[index] > 255) top_data[index] = 255.0;
		if (top_data[index] < 0) top_data[index] = 0.0;
		if (top_data[index+size] > 255) top_data[index+size] = 255.0;
		if (top_data[index+size] < 0) top_data[index+size] = 0.0;
		if (top_data[index+size+size] > 255) top_data[index+size+size] = 255.0;
		if (top_data[index+size+size] < 0) top_data[index+size+size] = 0.0;
	}
}

template <typename Dtype>
void AugmentationLayer<Dtype>::do_saturation_gpu(const Dtype* bottom_data, Dtype* top_data) {
	const int size = height_ * width_;
	for (int n = 0; n < num_; ++n) {
		caffe_rng_uniform(1, 1-saturation_max_bias_, 1+saturation_max_bias_, &saturation_bias_);
		saturation_kernel<Dtype><<<CAFFE_GET_BLOCKS(size),CAFFE_CUDA_NUM_THREADS>>>
											(bottom_data + n * total_, top_data + n * total_, size, width_, saturation_bias_);
	}
}

/***********************************MEAN***********************************/
template <typename Dtype>
__global__ void mean_file_kernel(const Dtype* bottom_data, Dtype* top_data, const int total, const Dtype* mean) {
	CUDA_KERNEL_LOOP(index, total) {
		top_data[index] = bottom_data[index] - mean[index];
	}
}

template <typename Dtype>
__global__ void mean_value_kernel(const Dtype* bottom_data, Dtype* top_data, const int size, 
										const Dtype value_B, const Dtype value_G, const Dtype value_R) {
	CUDA_KERNEL_LOOP(index, size) {
		top_data[index] = bottom_data[index] - value_B;
		top_data[index+size] = bottom_data[index+size] - value_G;
		top_data[index+size+size] = bottom_data[index+size+size] - value_R;
	}
}

template <typename Dtype>
void AugmentationLayer<Dtype>::do_mean_gpu(const Dtype* bottom_data, Dtype* top_data) {
	if (has_mean_file_) {
		const string& mean_file = this->layer_param_.augmentation_param().mean().mean_file();
		CHECK_EQ(channels_, mean_data_.channels());
		CHECK_EQ(height_, mean_data_.height());
		CHECK_EQ(width_, mean_data_.width());
		// const Dtype* mean = mean_data_.gpu_data();
		const Dtype* mean_cpu = mean_data_.cpu_data();
		Dtype* mean;
		cudaMalloc((void**) &mean, total_*sizeof(Dtype));
		cudaMemcpy(mean, mean_cpu, total_*sizeof(Dtype), cudaMemcpyHostToDevice);
		for (int n = 0; n < num_; ++n) {
			mean_file_kernel<Dtype><<<CAFFE_GET_BLOCKS(total_),CAFFE_CUDA_NUM_THREADS>>>
												(bottom_data + n * total_, top_data + n * total_, total_, mean);
		}
		cudaFree(mean);
	} else {
		const int size = height_ * width_;
		for (int n = 0; n < num_; ++n) {
			mean_value_kernel<Dtype><<<CAFFE_GET_BLOCKS(size),CAFFE_CUDA_NUM_THREADS>>>
												(bottom_data + n * total_, top_data + n * total_, size, mean_values_[0], mean_values_[1], mean_values_[2]);
		}
	}
}

/***********************************SIN WARP***********************************/
template <typename Dtype>
__global__ void sin_warp_kernel(const Dtype* bottom_data, Dtype* top_data, 
															const Dtype am, const Dtype hz, const Dtype ph, const int size, const int width, const int height) {
	CUDA_KERNEL_LOOP(index, size) {
		const int H = index / width;
		const int W = index % width;

		const Dtype xo = (am * sinf(3.1416 * H / hz + ph));
		const Dtype yo = (am * sinf(3.1416 * W / hz + ph));

		Dtype nH = (Dtype)H + yo;
		Dtype nW = (Dtype)W + xo;

		if (nW > width - 1 || nW < 0 || nH > height - 1 || nH < 0) {
			top_data[index] = 0;
			top_data[index+size] = 0;
			top_data[index+size+size] = 0;
		} else {
			resample(bottom_data, top_data, nW, nH, index, width, height);
		}

		// const Dtype p = fminf(height-1, fmaxf((H + yo),0));
		// const Dtype q = fminf(width-1, fmaxf((W + xo),0));

		// resample(bottom_data, top_data, q, p, index, width, height);
	}
}

template <typename Dtype>
void AugmentationLayer<Dtype>::do_sin_warp_gpu(const Dtype* bottom_data, Dtype* top_data) {
	const int size = height_ * width_;
	for (int n = 0; n < num_; ++n) {
		caffe_rng_uniform(1, sin_warp_min_am_, sin_warp_max_am_, &random_am_);
		caffe_rng_uniform(1, sin_warp_min_hz_, sin_warp_max_hz_, &random_hz_);
		caffe_rng_uniform(1, (Dtype)0., (Dtype)6.2832, &random_ph_);
		sin_warp_kernel<Dtype><<<CAFFE_GET_BLOCKS(size),CAFFE_CUDA_NUM_THREADS>>>
									(bottom_data + n * total_, top_data + n * total_, random_am_, random_hz_, random_ph_, size, width_, height_);
	}
}

/***********************************ROTATION***********************************/
template <typename Dtype>
__global__ void rotation_kernel(const Dtype* bottom_data, Dtype* top_data, const Dtype cH, const Dtype cW,
															const Dtype degree, const int size, const int width, const int height) {
	CUDA_KERNEL_LOOP(index, size) {
		const int H = index / width;
		const int W = index % width;
		const Dtype dW = (Dtype)W - cW;
		const Dtype dH = (Dtype)H - cH;
		const Dtype ca = cosf(degree);
		const Dtype sa = sinf(degree);

		// const Dtype ncW = ca * cW + sa * cH;
		// const Dtype ncH = -sa * cH + ca * cW;	
		
		// const Dtype dW = ncW - cW;
		// const Dtype dH = ncH - cH;

		// Dtype nW = ca * (W+dW) + sa * (H+dH);
		// Dtype nH = -sa * (W+dW) + ca * (H+dH);

		Dtype nW = ca * dW + sa * dH + cW;
		Dtype nH = -sa * dW + ca * dH + cH;

		// const Dtype dis = sqrtf(dW*dW+dH*dH) + 0.000001;
		// const Dtype theta = (cH > H)? 6.2831853 - acosf(dW/dis) : acosf(dW/dis);
		// Dtype nW = cW + dis * cosf(theta - degree);
		// Dtype nH = cH + dis * sinf(theta - degree);

		if (nW > width - 1 || nW < 0 || nH > height - 1 || nH < 0) {
			top_data[index] = 0;
			top_data[index+size] = 0;
			top_data[index+size+size] = 0;
		} else {
			resample(bottom_data, top_data, nW, nH, index, width, height);
		}

		// const Dtype nW = fminf(width-1, fmaxf((cW + dis * cosf(theta - degree)),0));
		// const Dtype nH = fminf(height-1, fmaxf((cH + dis * sinf(theta - degree)),0));

	}
}

template <typename Dtype>
void AugmentationLayer<Dtype>::do_rotation_gpu(const Dtype* bottom_data, Dtype* top_data) {
	const int size = height_ * width_;
	const Dtype cH = (height_ - 1) / 2;
	const Dtype cW = (width_ - 1) / 2;
	for (int n = 0; n < num_; ++n) {
		caffe_rng_uniform(1, -rotation_max_degree_, rotation_max_degree_, &rotation_degree_);
		rotation_kernel<Dtype><<<CAFFE_GET_BLOCKS(size),CAFFE_CUDA_NUM_THREADS>>>
									(bottom_data + n * total_, top_data + n * total_, cH, cW, rotation_degree_, size, width_, height_);
	}
}

/***********************************LENS DISTORTION***********************************/
template <typename Dtype>
__global__ void lens_distortion_kernel(const Dtype* bottom_data, Dtype* top_data, const Dtype cH, const Dtype cW, const Dtype paramA, 
															const Dtype paramB, const Dtype paramC, const Dtype paramD, const int size, const int width, const int height) {
	CUDA_KERNEL_LOOP(index, size) {
		const int H = index / width;
		const int W = index % width;
		const Dtype dW = ((Dtype)W - cW)/width;
		const Dtype dH = ((Dtype)H - cH)/height;
		const Dtype dstR = sqrtf(dW*dW+dH*dH) + 0.000001;
		const Dtype srcR = (paramA * dstR * dstR * dstR + paramB * dstR * dstR + paramC * dstR + paramD) * dstR;
		const Dtype factor = fabsf(dstR / srcR);
		Dtype nW = cW + dW * factor * width;
		Dtype nH = cH + dH * factor * height;

		if (nW > width - 1 || nW < 0 || nH > height - 1 || nH < 0) {
			top_data[index] = 0;
			top_data[index+size] = 0;
			top_data[index+size+size] = 0;
		} else {
			resample(bottom_data, top_data, nW, nH, index, width, height);
		}

		// const Dtype nW = fminf(width-1, fmaxf((cW + dW * factor * width),0));
		// const Dtype nH = fminf(height-1, fmaxf((cH + dH * factor * height),0));

		// resample(bottom_data, top_data, nW, nH, index, width, height);
	}
}

template <typename Dtype>
void AugmentationLayer<Dtype>::do_lens_distortion_gpu(const Dtype* bottom_data, Dtype* top_data) {
	const int size = height_ * width_;
	const Dtype cH = (height_ - 1) / 2;
	const Dtype cW = (width_ - 1) / 2;
	for (int n = 0; n < num_; ++n) {
		caffe_rng_uniform(1, -lens_max_parama_, lens_max_parama_, &lens_parama_);
		caffe_rng_uniform(1, -lens_max_paramb_, lens_max_paramb_, &lens_paramb_);
		lens_distortion_kernel<Dtype><<<CAFFE_GET_BLOCKS(size),CAFFE_CUDA_NUM_THREADS>>>(bottom_data + n * total_, top_data + n * total_, 
									cH, cW, lens_parama_, lens_paramb_, lens_paramc_, lens_paramd_, size, width_, height_);
	}
}

/***********************************RESCALE + CROP***********************************/
template <typename Dtype>
__global__ void rescale_kernel(const Dtype* bottom_data, Dtype* top_data, const Dtype width_scale, const Dtype height_scale,
														 const int size, const int width, const int height) {
	CUDA_KERNEL_LOOP(index, size) {
		const int H = index / width;
		const int W = index % width;

		Dtype nW = W / width_scale;
		Dtype nH = H / height_scale;

		if (nW > width - 1 || nW < 0 || nH > height - 1 || nH < 0) {
			top_data[index] = 255;
			top_data[index+size] = 255;
			top_data[index+size+size] = 255;
		} else {
			resample(bottom_data, top_data, nW, nH, index, width, height);
		}
	}
}

template <typename Dtype>
__global__ void crop_kernel(const Dtype* bottom_data, Dtype* top_data, const int wskip, const int hskip,
														 const int size, const int width, const int out_size, const int out_width) {
	CUDA_KERNEL_LOOP(index, out_size) {
		const int H = index / out_width;
		const int W = index % out_width;
		const int nH = H + hskip;
		const int nW = W + wskip;
		const int pos = nH * width + nW;
		top_data[index] = bottom_data[pos];
		top_data[index+out_size] = bottom_data[pos+size];
		top_data[index+out_size+out_size] = bottom_data[pos+size+size];
	}
}

template <typename Dtype>
void AugmentationLayer<Dtype>::do_crop_gpu(const Dtype* bottom_data, Dtype* top_data) {
	const int out_size = out_height_ * out_width_;
	const int size = height_ * width_;
	const int out_total = out_height_ * out_width_ * channels_;
	for (int n = 0; n < num_; ++n) {
		const int wskip = Rand(width_ - crop_size_);
		const int hskip = Rand(height_ - crop_size_);

		crop_kernel<Dtype><<<CAFFE_GET_BLOCKS(size),CAFFE_CUDA_NUM_THREADS>>>(bottom_data + n * total_, top_data + n * out_total, 
									wskip, hskip, size, width_ , out_size, out_width_);
	}
}

template <typename Dtype>
void AugmentationLayer<Dtype>::do_rescale_crop_gpu(const Dtype* bottom_data, Dtype* buffer, Dtype* top_data) {
	const int out_size = out_height_ * out_width_;
	const int out_total = out_height_ * out_width_ * channels_;
	const int size = height_ * width_;
	for (int n = 0; n < num_; ++n) {
		caffe_rng_uniform(1, width_min_scale_, width_max_scale_, &width_scale_);
		if (fixed_ratio_)
			height_scale_ = width_scale_;
		else
			caffe_rng_uniform(1, height_min_scale_, height_max_scale_, &height_scale_);

		// LOG(INFO)<<height_scale_<<","<<width_scale_;
		rescale_kernel<Dtype><<<CAFFE_GET_BLOCKS(size),CAFFE_CUDA_NUM_THREADS>>>(bottom_data + n * total_, buffer + n * total_, 
									width_scale_, height_scale_, size, width_, height_);

		const int nwidth = (int)(width_ * width_scale_);
		const int nheight = (int)(height_ * height_scale_);
		const int wskip = Rand(max(1,nwidth - crop_size_));
		const int hskip = Rand(max(1,nheight - crop_size_));
		crop_kernel<Dtype><<<CAFFE_GET_BLOCKS(size),CAFFE_CUDA_NUM_THREADS>>>(buffer + n * total_, top_data + n * out_total, 
									wskip, hskip, size, width_ , out_size, out_width_);
		// LOG(INFO)<<"=================";
		// LOG(INFO)<<nheight<<","<<nwidth;
		// LOG(INFO)<<crop_size_;
		// LOG(INFO)<<nheight - crop_size_<<","<<nwidth - crop_size_;
		// LOG(INFO)<<hskip<<","<<wskip;
		// LOG(INFO)<<hskip+crop_size_<<","<<wskip+crop_size_;
		// LOG(INFO)<<"=================";
	}
}

template <typename Dtype>
void AugmentationLayer<Dtype>::do_variable_crop_gpu(const Dtype* bottom_data, Dtype* buffer, Dtype* top_data, Dtype* crop_data, 
																const vector<Blob<Dtype>*>& top) {//DEVICE         DEVICE           DEVICE            HOST

	const int size = height_ * width_;

	// LOG(INFO) << ":::::::::::::::::::::::::::Starting:::::::::::::::::::::::::::";
	//do some rescale staffs
	if (size_protection_ == false) {
		caffe_rng_uniform(1, width_min_scale_, width_max_scale_, &width_scale_);
		if (fixed_ratio_)
			height_scale_ = width_scale_;
		else
			caffe_rng_uniform(1, height_min_scale_, height_max_scale_, &height_scale_);

	} else {
		Dtype tmpratio = std::min(static_cast<Dtype>(max_size_)/static_cast<Dtype>(height_),
													static_cast<Dtype>(max_size_)/static_cast<Dtype>(width_));
		Dtype ratio = std::min(tmpratio, (Dtype)1.0);

		caffe_rng_uniform(1, width_min_scale_, width_max_scale_, &width_scale_);
		if (fixed_ratio_)
			height_scale_ = width_scale_;
		else
			caffe_rng_uniform(1, height_min_scale_, height_max_scale_, &height_scale_);

		width_scale_ *= ratio;
		height_scale_ *= ratio;
	// LOG(INFO) << "size_protection: " <<size_protection_;
	// LOG(INFO) << "tmpRatio: " <<tmpratio;
	// LOG(INFO) << "Ratio: " <<ratio;
	}

	// LOG(INFO) << ":::::::::::::::::::::::::::Rescale:::::::::::::::::::::::::::";
	// LOG(INFO) << "Original size: " <<height_ <<" "<< width_;
	// LOG(INFO) << "Rescale scale: " <<height_scale_ <<" "<< width_scale_;
	const int nwidth = (int)(width_ * width_scale_);
	const int nheight = (int)(height_ * height_scale_);

	// LOG(INFO) << "Rescale size: " <<nheight <<" "<< nwidth;

	rescale_kernel<Dtype><<<CAFFE_GET_BLOCKS(size),CAFFE_CUDA_NUM_THREADS>>>(bottom_data, buffer, 
										width_scale_, height_scale_, size, width_, height_);

	// LOG(INFO) << ":::::::::::::::::::::::::::Crop:::::::::::::::::::::::::::";
	// LOG(INFO) << "Crop scale: " <<crop_scale_;
	const int crop_size_width = (int)nwidth * crop_scale_;
	const int crop_size_height = (int)nheight * crop_scale_;
	const int wskip = Rand(max(1,nwidth - crop_size_width));
	const int hskip = Rand(max(1,nheight - crop_size_height));
	const int out_size = crop_size_height * crop_size_width;

	// LOG(INFO) << "Crop size: " <<crop_size_height <<" "<< crop_size_width;

	top[0]->Reshape(num_,channels_,crop_size_height,crop_size_width);
	crop_kernel<Dtype><<<CAFFE_GET_BLOCKS(size),CAFFE_CUDA_NUM_THREADS>>>(buffer, top_data, 
								wskip, hskip, size, width_ , out_size, crop_size_width);
	// LOG(INFO) << ":::::::::::::::::::::::::::Ending:::::::::::::::::::::::::::";

	const int wrest = nwidth - wskip - crop_size_width;
	const int hrest = nheight - hskip - crop_size_height;

	crop_data[4] = height_scale_; crop_data[5] = width_scale_;
	crop_data[2] = wskip; crop_data[3] = wrest;
	crop_data[0] = hskip; crop_data[1] = hrest;
}


/***********************************FORWARD FUNCTION***********************************/
template <typename Dtype>
void AugmentationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
	const Dtype* bottom_data = bottom[0]->cpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	Dtype* top_buffer;
	Dtype* bottom_buffer;

	// LOG(INFO)<<top[t]->channels()<<","<<top[t]->height()<<","<<top[t]->width();
	cudaMalloc((void**) &bottom_buffer, count_*sizeof(Dtype));
	cudaMalloc((void**) &top_buffer, count_*sizeof(Dtype));
	cudaMemcpy(bottom_buffer, bottom_data, count_*sizeof(Dtype), cudaMemcpyHostToDevice);

	//==========LV1==========//
	//COLOR SHIFT
	if ((this->layer_param_.augmentation_param().color_shift().execute() ==
		AugmentationColorShiftParameter_Execute_ALWAYS) ||
			((this->layer_param_.augmentation_param().color_shift().execute() ==
		AugmentationColorShiftParameter_Execute_RANDOM) && (Rand(2)) ) ) {
		do_color_shift_gpu(bottom_buffer, top_buffer);
		cudaMemcpy(bottom_buffer, top_buffer, count_*sizeof(Dtype), cudaMemcpyDeviceToDevice);}

	//GAUSSIAN NOISE
	if ((this->layer_param_.augmentation_param().gaussian_noise().execute() ==
		AugmentationGaussianNoiseParameter_Execute_ALWAYS) ||
			((this->layer_param_.augmentation_param().gaussian_noise().execute() ==
		AugmentationGaussianNoiseParameter_Execute_RANDOM) && (Rand(2)) ) ) {
		do_gaussian_noise_gpu(bottom_buffer, top_buffer);
		cudaMemcpy(bottom_buffer, top_buffer, count_*sizeof(Dtype), cudaMemcpyDeviceToDevice);}

	//CONTRAST
	if ((this->layer_param_.augmentation_param().contrast().execute() ==
		AugmentationContrastParameter_Execute_ALWAYS) ||
			((this->layer_param_.augmentation_param().contrast().execute() ==
		AugmentationContrastParameter_Execute_RANDOM) && (Rand(2)) ) ) {
		do_contrast_gpu(bottom_buffer, top_buffer);
		cudaMemcpy(bottom_buffer, top_buffer, count_*sizeof(Dtype), cudaMemcpyDeviceToDevice);}

	//FLARE
	if ((this->layer_param_.augmentation_param().flare().execute() ==
		AugmentationFlareParameter_Execute_ALWAYS) ||
			((this->layer_param_.augmentation_param().flare().execute() ==
		AugmentationFlareParameter_Execute_RANDOM) && (Rand(2)) ) ) {
		do_flare_gpu(bottom_buffer, top_buffer);
		cudaMemcpy(bottom_buffer, top_buffer, count_*sizeof(Dtype), cudaMemcpyDeviceToDevice);}

	//BLUR
	if ((this->layer_param_.augmentation_param().blur().execute() ==
		AugmentationBlurParameter_Execute_ALWAYS) ||
			((this->layer_param_.augmentation_param().blur().execute() ==
		AugmentationBlurParameter_Execute_RANDOM) && (Rand(2)) ) ) {
		do_blur_gpu(bottom_buffer, top_buffer);
		cudaMemcpy(bottom_buffer, top_buffer, count_*sizeof(Dtype), cudaMemcpyDeviceToDevice);}

	//COLOR_BIAS
	if ((this->layer_param_.augmentation_param().color_bias().execute() ==
		AugmentationColorBiasParameter_Execute_ALWAYS) ||
			((this->layer_param_.augmentation_param().color_bias().execute() ==
		AugmentationColorBiasParameter_Execute_RANDOM) && (Rand(2)) ) ) {
		do_color_bias_gpu(bottom_buffer, top_buffer);
		cudaMemcpy(bottom_buffer, top_buffer, count_*sizeof(Dtype), cudaMemcpyDeviceToDevice);}

	//MIRROR
	if ((this->layer_param_.augmentation_param().mirror().execute() ==
		AugmentationMirrorParameter_Execute_ALWAYS) ||
			((this->layer_param_.augmentation_param().mirror().execute() ==
		AugmentationMirrorParameter_Execute_RANDOM) && (Rand(2)) ) ) {
		do_mirror_gpu(bottom_buffer, top_buffer);
		cudaMemcpy(bottom_buffer, top_buffer, count_*sizeof(Dtype), cudaMemcpyDeviceToDevice);
		if (output_crop_) { (top[1]->mutable_cpu_data())[6] = 1;} }

	//SATURATION
	if ((this->layer_param_.augmentation_param().saturation().execute() ==
		AugmentationSaturationParameter_Execute_ALWAYS) ||
			((this->layer_param_.augmentation_param().saturation().execute() ==
		AugmentationSaturationParameter_Execute_RANDOM) && (Rand(2)) ) ) {
		do_saturation_gpu(bottom_buffer, top_buffer);
		cudaMemcpy(bottom_buffer, top_buffer, count_*sizeof(Dtype), cudaMemcpyDeviceToDevice);}

	//MEAN
	if (this->layer_param_.augmentation_param().mean().execute() ==
		AugmentationMeanParameter_Execute_ALWAYS) {
		do_mean_gpu(bottom_buffer, top_buffer);
		cudaMemcpy(bottom_buffer, top_buffer, count_*sizeof(Dtype), cudaMemcpyDeviceToDevice);}

	//==========LV2==========//
	//LV2 LASSO
	vector<AUGFUNC> lv2_lasso;
	if (this->layer_param_.augmentation_param().sin_warp().execute() ==
		AugmentationSinWarpParameter_Execute_LASSO)
		lv2_lasso.push_back(&AugmentationLayer<Dtype>::do_sin_warp_gpu);
	if (this->layer_param_.augmentation_param().rotation().execute() ==
		AugmentationRotationParameter_Execute_LASSO)
		lv2_lasso.push_back(&AugmentationLayer<Dtype>::do_rotation_gpu);
	if (this->layer_param_.augmentation_param().lens_distortion().execute() ==
		AugmentationLensDistortionParameter_Execute_LASSO)
		lv2_lasso.push_back(&AugmentationLayer<Dtype>::do_lens_distortion_gpu);

	if (lv2_lasso.size()!=0 && Rand(2)) {
		int run_lasso = Rand(lv2_lasso.size());
		(this->*lv2_lasso[run_lasso])(bottom_buffer, top_buffer);
		cudaMemcpy(bottom_buffer, top_buffer, count_*sizeof(Dtype), cudaMemcpyDeviceToDevice);
	}

	//SIN WARP
	if ((this->layer_param_.augmentation_param().sin_warp().execute() ==
		AugmentationSinWarpParameter_Execute_ALWAYS) ||
			((this->layer_param_.augmentation_param().sin_warp().execute() ==
		AugmentationSinWarpParameter_Execute_RANDOM) && (Rand(2)) ) ) {
		do_sin_warp_gpu(bottom_buffer, top_buffer);
		cudaMemcpy(bottom_buffer, top_buffer, count_*sizeof(Dtype), cudaMemcpyDeviceToDevice);}

	//ROTATION
	if ((this->layer_param_.augmentation_param().rotation().execute() ==
		AugmentationRotationParameter_Execute_ALWAYS) ||
			((this->layer_param_.augmentation_param().rotation().execute() ==
		AugmentationRotationParameter_Execute_RANDOM) && (Rand(2)) ) ) {
		do_rotation_gpu(bottom_buffer, top_buffer);
		cudaMemcpy(bottom_buffer, top_buffer, count_*sizeof(Dtype), cudaMemcpyDeviceToDevice);}

	//LENSDISTORTION
	if ((this->layer_param_.augmentation_param().lens_distortion().execute() ==
		AugmentationLensDistortionParameter_Execute_ALWAYS) ||
			((this->layer_param_.augmentation_param().lens_distortion().execute() ==
		AugmentationLensDistortionParameter_Execute_RANDOM) && (Rand(2)) ) ) {
		do_lens_distortion_gpu(bottom_buffer, top_buffer);
		cudaMemcpy(bottom_buffer, top_buffer, count_*sizeof(Dtype), cudaMemcpyDeviceToDevice);}

	//==========LV3==========//
	//LV3
	if ( (this->layer_param_.augmentation_param().rescale().execute() == AugmentationRescaleParameter_Execute_NEVER) && 
		(this->layer_param_.augmentation_param().crop().execute() == AugmentationCropParameter_Execute_NEVER) ) {
		if (this->layer_param_.augmentation_param().variable_crop().execute() == AugmentationVariableCropParameter_Execute_NEVER)
			cudaMemcpy(top_data, bottom_buffer, count_*sizeof(Dtype), cudaMemcpyDeviceToDevice);
		else {
			if (output_crop_) {
				Dtype* crop_data = top[1]->mutable_cpu_data();
				do_variable_crop_gpu(bottom_buffer, top_buffer, top_data, crop_data, top);
			} else {
				Dtype* crop_data;
				crop_data = new Dtype[4];
				do_variable_crop_gpu(bottom_buffer, top_buffer, top_data, crop_data, top);
				delete[] crop_data;
			}
		}
	}
	else {
		if (this->layer_param_.augmentation_param().rescale().execute() == AugmentationRescaleParameter_Execute_NEVER)
			do_crop_gpu(bottom_buffer, top_data);
		else
			do_rescale_crop_gpu(bottom_buffer, top_buffer, top_data);
	}

	cudaFree(bottom_buffer);
	cudaFree(top_buffer);
}

template <typename Dtype>
void AugmentationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

}

INSTANTIATE_LAYER_GPU_FUNCS(AugmentationLayer);

}  // namespace caffe
