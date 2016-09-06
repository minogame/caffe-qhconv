#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void AugmentationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
	this->InitRand();
	num_ = bottom[0]->num();
	channels_ = bottom[0]->channels();
	height_ = bottom[0]->height();
	width_ = bottom[0]->width();
	total_ = channels_ * width_ * height_;
	count_ = bottom[0]->count();

	////////////////////////////COLOR SHIFT//////////////////////////
	if (this->layer_param_.augmentation_param().color_shift().execute() != AugmentationColorShiftParameter_Execute_NEVER) {
		color_shift_mu_ = this->layer_param_.augmentation_param().color_shift().mu();
		color_shift_sigma_ = this->layer_param_.augmentation_param().color_shift().sigma();
		color_shift_eig_value_.Reshape(1,1,1,3);
		color_shift_eig_matrix_.Reshape(1,1,3,3);
		color_shift_random_vector_.Reshape(1,1,1,4);
		Dtype* eig_value = color_shift_eig_value_.mutable_cpu_data();
		Dtype* eig_matrix = color_shift_eig_matrix_.mutable_cpu_data();
		//Dtype* random_vector = color_shift_random_vector_.mutable_cpu_data();
		for (int i = 0; i < 3; ++i) {
			eig_value[i] = this->layer_param_.augmentation_param().color_shift().eig_value(i); }
		for (int i = 0; i < 9; ++i) {
			eig_matrix[i] = this->layer_param_.augmentation_param().color_shift().eig_matrix(i); }
	}

	////////////////////////////GAUSSIAN NOISE//////////////////////////
	if (this->layer_param_.augmentation_param().gaussian_noise().execute() != AugmentationGaussianNoiseParameter_Execute_NEVER) {
		gaussian_noise_mu_ = this->layer_param_.augmentation_param().gaussian_noise().mu();
		gaussian_noise_sigma_ = this->layer_param_.augmentation_param().gaussian_noise().sigma(); 
		gaussian_noise_density_ = this->layer_param_.augmentation_param().gaussian_noise().density();
		gaussian_noise_mask_.Reshape(num_,channels_,height_,width_);
		gaussian_noise_buffer_.Reshape(num_,channels_,height_,width_);
		uint_thres_ = static_cast<unsigned int>(UINT_MAX * gaussian_noise_density_);
	}

	////////////////////////////CONTRAST//////////////////////////
	if (this->layer_param_.augmentation_param().contrast().execute() != AugmentationContrastParameter_Execute_NEVER) {
		contrast_max_thershold_ = this->layer_param_.augmentation_param().contrast().max_thershold();
	}

	////////////////////////////FLARE//////////////////////////
	if (this->layer_param_.augmentation_param().flare().execute() != AugmentationFlareParameter_Execute_NEVER) {
		flare_max_radius_ = this->layer_param_.augmentation_param().flare().max_radius();
		flare_min_radius_ = this->layer_param_.augmentation_param().flare().min_radius();
		flare_max_lumi_ = this->layer_param_.augmentation_param().flare().max_lumi();
		flare_min_lumi_ = this->layer_param_.augmentation_param().flare().min_lumi();
		flare_num_ = this->layer_param_.augmentation_param().flare().num_flare();
		flare_buffer_.Reshape(num_,channels_,height_,width_);
	}

	////////////////////////////BLUR//////////////////////////
	if (this->layer_param_.augmentation_param().blur().execute() != AugmentationBlurParameter_Execute_NEVER) {
		blur_max_sigma_ = this->layer_param_.augmentation_param().blur().max_sigma();
		blur_min_sigma_ = this->layer_param_.augmentation_param().blur().min_sigma();
	}

	////////////////////////////COLOR BIAS//////////////////////////
	if (this->layer_param_.augmentation_param().color_bias().execute() != AugmentationColorBiasParameter_Execute_NEVER) {
		color_max_bias_ = this->layer_param_.augmentation_param().color_bias().max_bias();
	}

	////////////////////////////MIRROR//////////////////////////

	////////////////////////////SATURATION//////////////////////////
	if (this->layer_param_.augmentation_param().saturation().execute() != AugmentationSaturationParameter_Execute_NEVER) {
		saturation_max_bias_ = this->layer_param_.augmentation_param().saturation().max_bias();
	}

	////////////////////////////MEAN//////////////////////////
	if (this->layer_param_.augmentation_param().mean().execute() != AugmentationMeanParameter_Execute_NEVER) {
		has_mean_file_ = this->layer_param_.augmentation_param().mean().has_mean_file();
		if (has_mean_file_) {
			CHECK_EQ(this->layer_param_.augmentation_param().mean().mean_value_size(), 0)
					<<"Cannot specify mean_file and mean_value at the same time";
			const string& mean_file = this->layer_param_.augmentation_param().mean().mean_file();
			BlobProto blob_proto;
			ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
			mean_data_.FromProto(blob_proto);
			// mean_.Reshape(1,channels_,height_,width_);
			// caffe_copy(total_,mean_data_.cpu_data(),mean_.mutable_cpu_data());
		}
		if (this->layer_param_.augmentation_param().mean().mean_value_size() > 0) {
			CHECK(this->layer_param_.augmentation_param().mean().has_mean_file() == false) <<
				"Cannot specify mean_file and mean_value at the same time";
			for (int c = 0; c < this->layer_param_.augmentation_param().mean().mean_value_size(); ++c) {
				mean_values_.push_back(this->layer_param_.augmentation_param().mean().mean_value(c));
			}
		}
	}

	////////////////////////////ROTATION//////////////////////////
	if (this->layer_param_.augmentation_param().rotation().execute() != AugmentationRotationParameter_Execute_NEVER) {
		rotation_max_degree_ = this->layer_param_.augmentation_param().rotation().max_degree();
		rotation_max_degree_ *= 0.01745;
	}

	////////////////////////////LENS DISTORTION//////////////////////////
	if (this->layer_param_.augmentation_param().lens_distortion().execute() != AugmentationLensDistortionParameter_Execute_NEVER) {
		lens_max_parama_ = this->layer_param_.augmentation_param().lens_distortion().max_parama();
		lens_max_paramb_ = this->layer_param_.augmentation_param().lens_distortion().max_paramb();
		lens_paramc_ = this->layer_param_.augmentation_param().lens_distortion().paramc();
		lens_paramd_ = this->layer_param_.augmentation_param().lens_distortion().paramd();
	}

	////////////////////////////SIN WARP//////////////////////////
	if (this->layer_param_.augmentation_param().sin_warp().execute() != AugmentationSinWarpParameter_Execute_NEVER) {
		sin_warp_max_am_ = this->layer_param_.augmentation_param().sin_warp().max_am();
		sin_warp_min_am_ = this->layer_param_.augmentation_param().sin_warp().min_am();
		sin_warp_max_hz_ = this->layer_param_.augmentation_param().sin_warp().max_hz();
		sin_warp_min_hz_ = this->layer_param_.augmentation_param().sin_warp().min_hz();
	}

	////////////////////////////RESCALE//////////////////////////
	if (this->layer_param_.augmentation_param().rescale().execute() != AugmentationRescaleParameter_Execute_NEVER) {
		width_min_scale_ = this->layer_param_.augmentation_param().rescale().width_min_scale();
		width_max_scale_ = this->layer_param_.augmentation_param().rescale().width_max_scale();
		height_min_scale_ = this->layer_param_.augmentation_param().rescale().height_min_scale();
		height_max_scale_ = this->layer_param_.augmentation_param().rescale().height_max_scale();
		fixed_ratio_ = this->layer_param_.augmentation_param().rescale().fixed_ratio();
	}

	////////////////////////////CROP//////////////////////////
	if (this->layer_param_.augmentation_param().crop().execute() != AugmentationCropParameter_Execute_NEVER) {
		crop_size_ = this->layer_param_.augmentation_param().crop().crop_size();
		out_height_ = crop_size_;
		out_width_ = crop_size_;
	} else {
		out_height_ = height_;
		out_width_ = width_;
	}

	////////////////////////////VARIABLE CROP//////////////////////////
	if (this->layer_param_.augmentation_param().variable_crop().execute() != AugmentationVariableCropParameter_Execute_NEVER) {
		CHECK_EQ(num_,1) << "Only one input could be processed due to a variable output size.";
		CHECK_EQ(this->layer_param_.augmentation_param().rescale().execute(),AugmentationRescaleParameter_Execute_NEVER)
										 <<	"Variable crop can not be executed simultaneously with rescale.";
		CHECK_EQ(this->layer_param_.augmentation_param().crop().execute(),AugmentationCropParameter_Execute_NEVER)
										 <<	"Variable crop can not be executed simultaneously with crop.";

		width_min_scale_ = this->layer_param_.augmentation_param().variable_crop().width_min_scale();
		width_max_scale_ = this->layer_param_.augmentation_param().variable_crop().width_max_scale();
		height_min_scale_ = this->layer_param_.augmentation_param().variable_crop().height_min_scale();
		height_max_scale_ = this->layer_param_.augmentation_param().variable_crop().height_max_scale();
		fixed_ratio_ = this->layer_param_.augmentation_param().variable_crop().fixed_ratio();
		size_protection_ = this->layer_param_.augmentation_param().variable_crop().size_protection();
		max_size_ = this->layer_param_.augmentation_param().variable_crop().max_size();
		crop_scale_ = this->layer_param_.augmentation_param().variable_crop().crop_scale();
		output_crop_ = this->layer_param_.augmentation_param().variable_crop().output_crop();

		if (output_crop_)
			CHECK_EQ(top.size(),2) << "Must have 2 top blobs when output crop is on.";
	}
}

template <typename Dtype>
void AugmentationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
	if (output_crop_) {
		// for (int top_id = 0; top_id < top.size()-1; top_id++) {
		// 	top[top_id]->Reshape(num_, channels_, out_height_, out_width_);
		// }
		top[0]->Reshape(num_, channels_, out_height_, out_width_);
		top[1]->Reshape(1,1,1,7); //crop,scale,mirror
	} else {
		for (int top_id = 0; top_id < top.size(); ++top_id) {
			top[top_id]->Reshape(num_, channels_, out_height_, out_width_);
		}
	}
}

template <typename Dtype>
void AugmentationLayer<Dtype>::InitRand() {
  const unsigned int rng_seed = caffe_rng_rand();
  rng_.reset(new Caffe::RNG(rng_seed));
}

template <typename Dtype>
int AugmentationLayer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

template <typename Dtype>
void AugmentationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void AugmentationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(AugmentationLayer);
#endif

INSTANTIATE_CLASS(AugmentationLayer);
REGISTER_LAYER_CLASS(Augmentation);

}  // namespace caffe
